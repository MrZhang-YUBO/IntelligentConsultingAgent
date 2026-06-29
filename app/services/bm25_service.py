"""BM25 稀疏检索服务 - 词法匹配检索，与向量检索互补。"""

import re
from typing import List, Tuple

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager


_PUNCT_RE = re.compile(
    r"[\s\u3000-\u303f\uff00-\uffef,.\!\/_,$%^*(+\"\']+|[+---!,\.?\~@#$%^&*()]+",
)


def _tokenize(text: str) -> List[str]:
    """简易中英文分词：中文逐字，英文按词。"""
    tokens: List[str] = []
    segments = _PUNCT_RE.split(text.lower().strip())
    for seg in segments:
        if not seg:
            continue
        current_word: List[str] = []
        for ch in seg:
            if "\u4e00" <= ch <= "\u9fff":
                if current_word:
                    tokens.append("".join(current_word))
                    current_word = []
                tokens.append(ch)
            else:
                current_word.append(ch)
        if current_word:
            tokens.append("".join(current_word))
    return [t for t in tokens if t]


class BM25Service:
    """BM25 稀疏检索服务。

    从 Milvus 拉取 rag_bm25_corpus_size 条文档构建语料池，
    在池内计算 BM25 得分。语料池首次调用时惰性构建。
    """

    def __init__(self) -> None:
        self._corpus_docs: List[Document] = []
        self._bm25 = None
        logger.info("BM25 服务初始化完成（语料延迟加载）")

    def _ensure_corpus(self) -> None:
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RuntimeError(
                "缺少 rank-bm25 依赖，请执行: pip install rank-bm25"
            ) from e

        logger.info(
            f"构建 BM25 语料池: 从 Milvus 拉取最多 {config.rag_bm25_corpus_size} 条文档"
        )
        try:
            collection = milvus_manager.get_collection()
        except Exception as e:
            logger.warning(f"获取 Milvus collection 失败，BM25 暂不可用: {e}")
            return

        try:
            batch_size = min(config.rag_bm25_corpus_size, 100)
            docs: List[Document] = []
            count = 0
            it = collection.query_iterator(
                expr="id != ''",
                output_fields=["id", "content", "metadata"],
                batch_size=batch_size,
            )
            try:
                while True:
                    batch = it.next()
                    if not batch:
                        break
                    for row in batch:
                        content = row.get("content", "")
                        if not content:
                            continue
                        metadata = row.get("metadata", {}) or {}
                        docs.append(Document(page_content=content, metadata=metadata))
                        count += 1
                        if count >= config.rag_bm25_corpus_size:
                            break
                    if count >= config.rag_bm25_corpus_size:
                        break
            finally:
                try:
                    it.close()
                except Exception:
                    pass

            if not docs:
                logger.warning("BM25 语料池为空，无法构建索引")
                return

            self._corpus_docs = docs
            tokenized = [_tokenize(doc.page_content) for doc in docs]
            self._bm25 = BM25Okapi(tokenized)
            logger.info(
                f"BM25 语料池构建完成: 共 {len(docs)} 篇文档, "
                f"平均 {sum(len(t) for t in tokenized) / len(tokenized):.0f} tokens/篇"
            )
        except Exception as e:
            logger.error(f"构建 BM25 语料池失败: {e}")

    def rebuild_corpus(self) -> int:
        self._corpus_docs = []
        self._bm25 = None
        self._ensure_corpus()
        return len(self._corpus_docs)

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """BM25 检索，返回 (Document, 归一化得分 0~1) 列表。"""
        if top_k is None:
            top_k = config.rag_bm25_top_k

        self._ensure_corpus()

        if self._bm25 is None or not self._corpus_docs:
            return []

        try:
            query_tokens = _tokenize(query)
            if not query_tokens:
                return []

            scores = self._bm25.get_scores(query_tokens)
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_items = indexed[:top_k]

            max_score = max((s for _, s in top_items), default=0.0)
            min_score = min((s for _, s in top_items), default=0.0)
            score_range = max_score - min_score

            results: List[Tuple[Document, float]] = []
            for doc_idx, raw in top_items:
                if raw <= 0:
                    continue
                if score_range > 0:
                    norm = (raw - min_score) / score_range
                else:
                    norm = 1.0
                results.append((self._corpus_docs[doc_idx], norm))

            logger.info(
                f"BM25 检索完成: query='{query}', "
                f"tokens={len(query_tokens)}, 命中 {len(results)} 篇"
            )
            return results
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []


bm25_service = BM25Service()