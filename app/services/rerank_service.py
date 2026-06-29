"""重排服务 - 基于 DashScope Rerank API 的文档重排。"""

from typing import List

from langchain_core.documents import Document
from loguru import logger

from app.config import config


class RerankService:
    """文档重排服务。

    使用 DashScope Rerank API (OpenAI 兼容模式) 对候选文档重新排序，
    选出与查询最相关的 TOP rag_rerank_top_k 篇。
    """

    def __init__(self) -> None:
        self.api_key = config.dashscope_api_key
        self.model = config.rag_rerank_model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        logger.info(
            f"重排服务初始化完成: model={self.model}, "
            f"启用={config.rag_enable_rerank}, TOP_K={config.rag_rerank_top_k}"
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> List[Document]:
        """对候选文档进行重排。

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 最终返回数量，默认使用 config.rag_rerank_top_k

        Returns:
            List[Document]: 重排并截断后的文档列表。
                如果 API 不可用或未启用，则原序返回。
        """
        if top_k is None:
            top_k = config.rag_rerank_top_k

        if not config.rag_enable_rerank:
            logger.debug("重排已禁用，直接返回原文档")
            return documents[:top_k]

        if not documents:
            return []

        if len(documents) <= top_k:
            logger.debug(f"候选文档({len(documents)}) <= TOP_K({top_k})，跳过重排")
            return documents

        logger.info(
            f"开始重排: query='{query}', candidates={len(documents)}, TOP_K={top_k}"
        )

        if not self.api_key:
            logger.warning("未配置 dashscope_api_key，跳过重排")
            return documents[:top_k]

        try:
            import requests
        except ImportError:
            logger.warning("缺少 requests 依赖，跳过重排")
            return documents[:top_k]

        try:
            # 准备文档文本
            documents_text = [
                doc.page_content[:2000] for doc in documents
            ]

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "query": query,
                "documents": documents_text,
                "top_n": top_k,
            }

            logger.debug(
                f"调用 DashScope Rerank API: {self.base_url}/rerank, model={self.model}"
            )

            response = requests.post(
                f"{self.base_url}/rerank",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                logger.warning("Rerank API 返回空结果，使用原文档前 TOP_K")
                return documents[:top_k]

            # 按返回的 index 和 relevance_score 重建顺序
            reranked: List[Document] = []
            for item in results:
                idx = item.get("index")
                score = item.get("relevance_score", 0.0)
                if idx is None or 0 > idx >= len(documents):
                    continue
                doc = documents[idx]
                # 把重排得分写入 metadata 方便后续调试
                new_meta = dict(doc.metadata) if doc.metadata else {}
                new_meta["_rerank_score"] = score
                reranked.append(
                    Document(page_content=doc.page_content, metadata=new_meta)
                )

            logger.info(
                f"重排完成: {len(documents)} -> {len(reranked)} 篇, "
                f"top_score={results[0].get('relevance_score', 'N/A')}"
            )
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"重排失败，回退到原文档排序: {e}")
            return documents[:top_k]


rerank_service = RerankService()