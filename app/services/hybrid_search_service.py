"""混合检索服务 - 向量检索 + BM25 稀疏检索融合。"""

from typing import Dict, List, Tuple

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.bm25_service import bm25_service
from app.services.vector_store_manager import vector_store_manager


class HybridSearchService:
    """混合检索：向量相似度 + BM25 词法匹配。

    融合策略:
        final_score = vector_weight * vector_score + bm25_weight * bm25_score
        两路 score 均已在各自模块做了 Min-Max 归一化 (0~1)。
    """

    def __init__(self) -> None:
        self.vector_weight = config.rag_vector_weight
        self.bm25_weight = config.rag_bm25_weight
        logger.info(
            f"混合检索服务初始化完成: "
            f"向量权重={self.vector_weight}, BM25权重={self.bm25_weight}"
        )

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """用于去重的文档唯一键。优先用 metadata._source + 内容签名。"""
        meta = doc.metadata or {}
        source = meta.get("_source", "")
        file_name = meta.get("_file_name", "")
        identifier = source or file_name or doc.page_content[:60]
        return identifier

    def _vector_search(
        self, query: str, top_k: int
    ) -> List[Tuple[Document, float]]:
        """向量检索：返回 (Document, 归一化得分 0~1)。"""
        try:
            docs = vector_store_manager.similarity_search(query, k=top_k)
        except Exception as e:
            logger.error(f"向量检索调用失败: {e}")
            return []

        if not docs:
            return []

        # 由于 similarity_search 未直接返回分数，采用 L2 距离近似计算：
        # 调用 Milvus 原生 search 拿到距离，再转换为相似度。
        try:
            from app.services.vector_embedding_service import vector_embedding_service
            from app.core.milvus_client import milvus_manager

            query_vector = vector_embedding_service.embed_query(query)
            collection = milvus_manager.get_collection()
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

            hits: List[Tuple[Document, float]] = []
            for hit in results[0]:
                content = hit.entity.get("content", "")
                if not content:
                    continue
                metadata = hit.entity.get("metadata", {}) or {}
                distance = float(hit.distance)
                # L2 距离转相似度: sim = 1 / (1 + distance)，范围 (0,1]
                similarity = 1.0 / (1.0 + distance)
                hits.append((Document(page_content=content, metadata=metadata), similarity))

            if not hits:
                return []

            # 在 top-k 内做一次 Min-Max 归一化
            max_s = max(s for _, s in hits)
            min_s = min(s for _, s in hits)
            rng = max_s - min_s
            normalized: List[Tuple[Document, float]] = []
            for doc, s in hits:
                ns = (s - min_s) / rng if rng > 0 else 1.0
                normalized.append((doc, ns))
            logger.info(f"向量检索: 命中 {len(normalized)} 篇")
            return normalized
        except Exception as e:
            logger.warning(f"向量 Milvus search 失败，回退到无分数结果: {e}")
            # 回退：用占位符分数 1.0，顺序保持
            return [(doc, 1.0 - i * 0.05) for i, doc in enumerate(docs)]

    def search(self, query: str) -> List[Document]:
        """执行混合检索，返回 TOP rag_hybrid_top_k 篇文档。"""
        vector_k = config.rag_vector_top_k
        bm25_k = config.rag_bm25_top_k
        hybrid_k = config.rag_hybrid_top_k

        logger.info(
            f"混合检索开始: query='{query}', "
            f"向量 TOP {vector_k} + BM25 TOP {bm25_k} -> 融合 TOP {hybrid_k}"
        )

        # 1. 两路检索
        vector_results = self._vector_search(query, vector_k)
        bm25_results = bm25_service.search(query, top_k=bm25_k)

        # 2. 用字典合并：键 = 文档标识，值 = (final_score, doc, has_vector, has_bm25)
        merged: Dict[str, Dict] = {}

        for doc, score in vector_results:
            key = self._doc_key(doc)
            if key not in merged:
                merged[key] = {"doc": doc, "v": 0.0, "b": 0.0}
            merged[key]["v"] = max(merged[key]["v"], score)

        for doc, score in bm25_results:
            key = self._doc_key(doc)
            if key not in merged:
                merged[key] = {"doc": doc, "v": 0.0, "b": 0.0}
            merged[key]["b"] = max(merged[key]["b"], score)

        # 3. 加权融合
        scored: List[Tuple[Document, float]] = []
        for key, info in merged.items():
            # 对只在单一路径出现的文档，给另一个路径补一个较小的保底分
            v_score = info["v"] if info["v"] > 0 else 0.05
            b_score = info["b"] if info["b"] > 0 else 0.05
            final = self.vector_weight * v_score + self.bm25_weight * b_score
            scored.append((info["doc"], final))

        # 4. 排序取 Top（保留融合分数到 metadata，供自动触发网络检索判断）
        scored.sort(key=lambda x: x[1], reverse=True)
        top_docs = []
        for doc, score in scored[:hybrid_k]:
            new_meta = dict(doc.metadata) if doc.metadata else {}
            new_meta["_hybrid_score"] = score
            top_docs.append(Document(page_content=doc.page_content, metadata=new_meta))

        logger.info(
            f"混合检索完成: 向量命中 {len(vector_results)} 篇, "
            f"BM25 命中 {len(bm25_results)} 篇, 去重融合后 -> {len(top_docs)} 篇"
        )
        return top_docs


hybrid_search_service = HybridSearchService()