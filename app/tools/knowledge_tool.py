"""知识检索工具 - 混合检索 (向量 + BM25) + 重排 + 网络检索自动触发

注意：不使用 response_format="content_and_artifact"，避免 ToolMessage.artifact
导致 LangGraph MemorySaver 的 msgpack 序列化失败。
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.hybrid_search_service import hybrid_search_service
from app.services.rerank_service import rerank_service


@tool
def retrieve_knowledge(query: str) -> str:
    """从知识库中检索相关信息来回答问题

    采用混合检索策略：向量检索 + BM25 词法检索 -> 融合 -> 重排 -> TOP 10。
    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    当知识库检索得分低于阈值时，会自动触发网络检索补充信息。

    Args:
        query: 用户的问题或查询

    Returns:
        str: 格式化的知识检索结果文本
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # Step 1: 混合检索 - 向量 TOP 10 + BM25 TOP 10 -> 融合 TOP 20
        candidate_docs = hybrid_search_service.search(query)

        if not candidate_docs:
            logger.warning("混合检索未找到文档，尝试回退到纯向量检索")
            try:
                from app.services.vector_store_manager import vector_store_manager
                candidate_docs = vector_store_manager.similarity_search(
                    query, k=config.rag_rerank_top_k
                )
            except Exception as fallback_err:
                logger.error(f"回退向量检索也失败: {fallback_err}")

            # 知识库完全无结果，尝试网络检索兜底
            if not candidate_docs and config.web_search_enabled and config.web_search_auto_trigger_enabled:
                logger.info("知识库无结果，触发网络检索兜底")
                return _web_search_fallback(query)

            if not candidate_docs:
                return "没有找到相关信息。"

        # Step 2: 重排 - TOP rag_rerank_top_k (默认 10)
        final_docs = rerank_service.rerank(query, candidate_docs)

        if not final_docs:
            logger.warning("重排后无文档，使用混合检索前 TOP_K")
            final_docs = candidate_docs[: config.rag_rerank_top_k]

        # Step 3: 检查是否需要自动触发网络检索
        web_docs: List[Document] = []
        if config.web_search_enabled and config.web_search_auto_trigger_enabled:
            top_score = _get_top_score(final_docs)
            if top_score < config.web_search_auto_trigger_threshold:
                logger.info(
                    f"知识库检索最高得分低于阈值 "
                    f"({top_score:.3f} < {config.web_search_auto_trigger_threshold})，"
                    f"自动触发网络检索"
                )
                try:
                    from app.services.web_search_service import web_search_service
                    web_docs = web_search_service.search(query)
                except Exception as e:
                    logger.error(f"自动触发网络检索失败: {e}")
                    web_docs = []

        # Step 4: 合并结果（如有网络检索结果）
        if web_docs:
            final_docs = _merge_results(final_docs, web_docs)

        # Step 5: 格式化文档为上下文
        context = format_docs(final_docs)

        logger.info(
            f"知识检索完成: 混合候选 {len(candidate_docs)} 篇 -> "
            f"重排后 {len(final_docs) - len(web_docs)} 篇"
            + (f" + 网络检索 {len(web_docs)} 篇" if web_docs else "")
        )
        return context

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}"


def _web_search_fallback(query: str) -> str:
    """知识库完全无结果时的网络检索兜底

    Args:
        query: 查询文本

    Returns:
        str: 格式化的检索结果文本
    """
    try:
        from app.services.web_search_service import web_search_service
        web_docs = web_search_service.search(query)

        if web_docs:
            # 标记所有文档为网络来源
            for doc in web_docs:
                if doc.metadata:
                    doc.metadata["_result_source"] = "web_search"
                else:
                    doc.metadata = {"_result_source": "web_search"}

            context = format_docs(web_docs)
            logger.info(f"网络检索兜底: 返回 {len(web_docs)} 篇文档")
            return context

        return "知识库和网络检索均未找到相关信息。"

    except Exception as e:
        logger.error(f"网络检索兜底失败: {e}")
        return "没有找到相关信息。"


def _get_top_score(docs: List[Document]) -> float:
    """获取文档列表中的最高得分

    优先取 _rerank_score（更准确），无则取 _hybrid_score。
    都没有则默认 1.0（表示得分足够高，不触发网络检索）。

    Args:
        docs: 文档列表

    Returns:
        float: 最高得分
    """
    if not docs:
        return 0.0

    max_score = 0.0
    for doc in docs:
        metadata = doc.metadata or {}
        score = metadata.get("_rerank_score", metadata.get("_hybrid_score", None))
        if score is not None:
            max_score = max(max_score, float(score))
        else:
            # 无分数信息，保守返回 1.0（不触发网络检索）
            return 1.0

    return max_score


def _merge_results(
    kb_docs: List[Document], web_docs: List[Document]
) -> List[Document]:
    """合并知识库和网络检索结果

    策略：知识库结果在前（主），网络结果追加在后（辅）。
    每条文档添加 _result_source metadata 以区分来源。

    Args:
        kb_docs: 知识库文档列表
        web_docs: 网络检索文档列表

    Returns:
        合并后的文档列表
    """
    merged: List[Document] = []

    # 知识库文档：标记来源
    for doc in kb_docs:
        new_meta = dict(doc.metadata) if doc.metadata else {}
        new_meta.setdefault("_result_source", "knowledge_base")
        merged.append(
            Document(page_content=doc.page_content, metadata=new_meta)
        )

    # 网络检索文档：已自带 _result_source="web_search"，直接追加
    merged.extend(web_docs)

    logger.info(
        f"结果合并: 知识库 {len(kb_docs)} 篇 + 网络 {len(web_docs)} 篇 = {len(merged)} 篇"
    )
    return merged


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata or {}
        result_source = metadata.get("_result_source", "knowledge_base")

        # 区分来源显示
        if result_source == "web_search":
            # 网络来源
            title = metadata.get("_web_title", "未知标题")
            url = metadata.get("_web_url", "")
            source = f"🌐 {title}"
            if url:
                source += f" ({url})"
        else:
            # 知识库来源
            source = metadata.get("_file_name", "未知来源")

            # 提取标题信息 (如果有)
            headers = []
            for key in ["h1", "h2", "h3"]:
                if key in metadata and metadata[key]:
                    headers.append(metadata[key])

            header_str = " > ".join(headers) if headers else ""
            if header_str:
                source = f"{header_str} ({source})"

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)