"""网络检索工具 - 基于 Tavily Search API 的 LangChain Tool

供 Agent 自主调用，当用户明确要求网络搜索或知识库检索不足时使用。
注意：不使用 response_format="content_and_artifact"，避免 ToolMessage.artifact
导致 LangGraph MemorySaver 的 msgpack 序列化失败。
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.web_search_service import web_search_service


@tool
def web_search(query: str) -> str:
    """搜索网络获取实时信息。当用户问当前事件、新闻、最新发展，或知识库检索结果不足，或用户明确要求网络搜索时使用此工具。

    Args:
        query: 搜索查询文本

    Returns:
        str: 格式化的网络检索结果文本
    """
    try:
        if not config.web_search_enabled:
            logger.debug("网络检索功能未启用，跳过")
            return "网络检索功能未启用。"

        logger.info(f"网络检索工具被调用: query='{query}'")

        # 调用网络检索服务（内部已包含安全过滤 + 摘要压缩）
        documents = web_search_service.search(query)

        if not documents:
            logger.info("网络检索未找到结果")
            return "网络检索未找到相关信息。"

        # 格式化文档为上下文
        context = format_web_docs(documents)

        logger.info(f"网络检索完成: 返回 {len(documents)} 篇文档")
        return context

    except Exception as e:
        logger.error(f"网络检索工具调用失败: {e}")
        return f"网络检索时发生错误: {str(e)}"


def format_web_docs(docs: List[Document]) -> str:
    """格式化网络检索文档列表为上下文文本

    Args:
        docs: 网络检索文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        title = metadata.get("_web_title", "未知标题")
        url = metadata.get("_web_url", "")

        formatted = f"【网络资料 {i}】"
        formatted += f"\n标题: {title}"
        if url:
            formatted += f"\n来源: {url}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)