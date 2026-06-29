"""网络检索服务 - 基于 Tavily Search API

核心流程：Tavily API 检索 → 安全过滤 → 摘要压缩 → 返回 Document 列表

设计原则：
  - 遵循项目中其他服务的单例 + fail-safe 模式
  - 使用同步 TavilyClient（因为 LangGraph @tool 在同步上下文执行）
  - 延迟初始化客户端（避免启动时因 API Key 缺失而崩溃）
  - 失败时返回空列表，绝不中断主对话
"""

import asyncio
from textwrap import dedent
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config
from app.services.content_safety_service import content_safety_service


class WebSearchService:
    """网络检索服务

    集成 Tavily Search API，提供：
    1. 网络检索（Tavily）
    2. 内容安全过滤（规则引擎 + LLM 审核）
    3. 摘要压缩（qwen-turbo）
    """

    def __init__(self):
        self._tavily_client = None
        self._summarization_model: Optional[ChatQwen] = None

    # ── 客户端管理 ────────────────────────────────────────────────

    def _ensure_client(self):
        """延迟初始化 Tavily 客户端（同步）

        使用同步 TavilyClient，因为 LangGraph 的 @tool 在同步上下文执行。
        延迟初始化确保 API Key 未配置时不会在导入阶段崩溃。
        """
        if self._tavily_client is not None:
            return

        try:
            from tavily import TavilyClient

            if not config.tavily_api_key:
                raise ValueError("TAVILY_API_KEY 未配置")

            self._tavily_client = TavilyClient(api_key=config.tavily_api_key)
            logger.info("Tavily 客户端初始化成功")

        except ImportError as e:
            raise RuntimeError(
                "缺少 tavily-python 依赖，请执行: pip install tavily-python"
            ) from e

    def _get_summarization_model(self) -> ChatQwen:
        """延迟初始化摘要模型"""
        if self._summarization_model is None:
            self._summarization_model = ChatQwen(
                model=config.web_search_summarization_model,
                api_key=config.dashscope_api_key,
                temperature=0.3,
                streaming=False,
            )
            logger.info(f"网络检索摘要模型初始化完成: {config.web_search_summarization_model}")
        return self._summarization_model

    # ── 核心检索接口 ──────────────────────────────────────────────

    def search(self, query: str) -> List[Document]:
        """执行网络检索（同步接口，供 @tool 调用）

        流程：
        1. 调用 Tavily API
        2. 转换为 Document 列表
        3. 安全过滤（规则引擎 + LLM 审核）
        4. 摘要压缩
        5. 返回过滤/压缩后的文档列表

        失败时返回空列表，绝不抛异常。

        Args:
            query: 搜索查询

        Returns:
            List[Document]: 网络检索结果文档列表
        """
        if not config.web_search_enabled:
            logger.debug("网络检索功能未启用")
            return []

        try:
            self._ensure_client()
        except Exception as e:
            logger.error(f"Tavily 客户端初始化失败: {e}")
            return []

        try:
            logger.info(f"开始网络检索: query='{query}'")

            # 1. 调用 Tavily API
            raw_results = self._tavily_client.search(
                query=query,
                max_results=config.web_search_max_results,
                search_depth=config.web_search_search_depth,
            )

            results = raw_results.get("results", [])
            if not results:
                logger.info("Tavily 搜索无结果")
                return []

            logger.info(f"Tavily 搜索返回 {len(results)} 条结果")

            # 2. 转换为 Document 列表
            documents = self._convert_to_documents(results, query)

            # 3. 安全过滤 - 第一层：规则引擎
            if config.web_search_safety_enabled:
                documents = content_safety_service.filter_by_rules(documents)

            if not documents:
                logger.info("规则引擎过滤后无安全文档")
                return []

            # 4. 安全过滤 - 第二层：LLM 审核
            if config.web_search_llm_review_enabled:
                try:
                    documents = self._run_async_from_sync(
                        content_safety_service.review_with_llm(query, documents),
                        timeout=30,
                    )
                except Exception as e:
                    logger.warning(f"LLM 安全审核失败（fail-open）: {e}")

            if not documents:
                logger.info("安全过滤后无文档")
                return []

            # 5. 摘要压缩
            if config.web_search_summarization_enabled:
                try:
                    documents = self._run_async_from_sync(
                        self._summarize_results(query, documents),
                        timeout=30,
                    )
                except Exception as e:
                    logger.warning(f"摘要压缩失败，使用原文档: {e}")

            logger.info(f"网络检索完成: 返回 {len(documents)} 篇文档")
            return documents

        except Exception as e:
            logger.error(f"网络检索失败: {e}")
            return []

    async def search_async(self, query: str) -> List[Document]:
        """执行网络检索（异步接口，供异步上下文调用）

        与 search() 流程相同，但使用异步 LLM 调用。
        """
        if not config.web_search_enabled:
            return []

        try:
            self._ensure_client()
        except Exception as e:
            logger.error(f"Tavily 客户端初始化失败: {e}")
            return []

        try:
            logger.info(f"开始网络检索(async): query='{query}'")

            raw_results = self._tavily_client.search(
                query=query,
                max_results=config.web_search_max_results,
                search_depth=config.web_search_search_depth,
            )

            results = raw_results.get("results", [])
            if not results:
                return []

            documents = self._convert_to_documents(results, query)

            # 安全过滤
            if config.web_search_safety_enabled:
                documents = content_safety_service.filter_by_rules(documents)
            if documents and config.web_search_llm_review_enabled:
                documents = await content_safety_service.review_with_llm(query, documents)

            # 摘要压缩
            if documents and config.web_search_summarization_enabled:
                documents = await self._summarize_results(query, documents)

            logger.info(f"网络检索完成(async): 返回 {len(documents)} 篇文档")
            return documents

        except Exception as e:
            logger.error(f"网络检索失败(async): {e}")
            return []

    # ── 同步上下文中运行异步代码的辅助方法 ─────────────────────────

    @staticmethod
    def _run_async_from_sync(coro, timeout: int = 30):
        """在同步上下文中安全运行异步协程

        策略：
          1. 尝试获取当前运行中的事件循环（如 FastAPI 的 uvloop）
             - 如果存在，在新线程中用 asyncio.run() 执行（避免冲突）
          2. 如果没有运行中的事件循环（如 @tool 在 LangGraph 线程池中），
             直接用 asyncio.run() 执行
          3. 兜底：创建新线程运行 asyncio.run()

        Args:
            coro: 异步协程对象
            timeout: 超时秒数

        Returns:
            协程的返回值
        """
        try:
            loop = asyncio.get_running_loop()
            # 已在事件循环中（如 FastAPI），不能直接 run_until_complete
            # 在新线程中运行 asyncio.run()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=timeout)
        except RuntimeError:
            # 没有运行中的事件循环，直接用 asyncio.run()
            return asyncio.run(coro)

    # ── 内部方法 ──────────────────────────────────────────────────

    def _convert_to_documents(
        self, results: list, query: str
    ) -> List[Document]:
        """将 Tavily 结果转换为 LangChain Document 列表

        每条结果的 metadata 包含：
        - _source: "web_search" （标识来源）
        - _web_url: 结果 URL
        - _web_title: 结果标题
        - _web_score: Tavily 相关度分数
        - _search_type: "web"
        - _result_source: "web_search" （与知识库结果统一标识）
        """
        documents: List[Document] = []
        for item in results:
            content = item.get("content", "")
            if not content or not content.strip():
                continue

            # 截断过长内容
            if len(content) > config.web_search_max_content_length:
                content = content[: config.web_search_max_content_length] + "\n...(已截断)"

            metadata = {
                "_source": "web_search",
                "_web_url": item.get("url", ""),
                "_web_title": item.get("title", ""),
                "_web_score": item.get("score", 0.0),
                "_search_type": "web",
                "_result_source": "web_search",
            }

            documents.append(
                Document(page_content=content, metadata=metadata)
            )

        return documents

    async def _summarize_results(
        self, query: str, documents: List[Document]
    ) -> List[Document]:
        """压缩网络检索结果

        策略：对每条结果独立压缩至约 300 字，保留关键事实。
        原始内容存入 metadata._original_content 备查。

        失败时返回原文档列表。
        """
        if not documents:
            return documents

        model = self._get_summarization_model()

        system_prompt = dedent("""
            你是一个"网络检索结果压缩员"，任务是把一段网页搜索结果压缩成简明摘要。
            压缩规则：
            1. 用中文输出，字数控制在 200~300 字之间
            2. 保留关键信息：核心事实、数据、结论
            3. 移除广告、导航、页脚等无信息价值的内容
            4. 保留专有名词和数字的准确性
            5. 不要输出多余的解释
            输出格式：直接输出摘要文本，不需要任何前缀。
        """).strip()

        summarized: List[Document] = []
        for doc in documents:
            try:
                user_prompt = dedent(f"""
                    用户查询: {query}
                    网页标题: {doc.metadata.get('_web_title', '无标题')}
                    网页内容:
                    ---
                    {doc.page_content}
                    ---
                    请压缩以上内容为简明摘要。
                """).strip()

                response = await model.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])

                summary_text = response.content if hasattr(response, "content") else str(response)
                summary_text = summary_text.strip()

                if not summary_text:
                    summarized.append(doc)
                    continue

                # 保留原始内容在 metadata 中
                new_meta = dict(doc.metadata) if doc.metadata else {}
                new_meta["_original_content"] = doc.page_content

                summarized.append(
                    Document(page_content=summary_text, metadata=new_meta)
                )

            except Exception as e:
                logger.warning(f"压缩单条网络结果失败，保留原文: {e}")
                summarized.append(doc)

        logger.info(f"摘要压缩完成: {len(documents)} 篇 -> {len(summarized)} 篇")
        return summarized


# 全局单例
web_search_service = WebSearchService()
