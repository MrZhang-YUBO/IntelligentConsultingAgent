"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS
from app.services.content_safety_service import content_safety_service
from app.agent.mcp_client import (
    get_mcp_client_with_retry,
    load_mcp_tools_safe,
    format_exception_chain,
    suggest_mcp_transport,
)
from app.agent.summary_agent import SummaryAgent, summary_agent
from app.agent.intent_agent import intent_agent, intent_tracker, IntentRecognitionResult
from app.core.metrics import timed_metric

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    【兜底机制】简单修剪消息历史——只在消息特别多时才触发

    说明：
    正常情况下由 RagAgentService._summarize_and_update() 负责"总结式压缩"，
    本函数是兜底方案：当消息数量超过 threshold 时（例如工具调用链导致消息积累过多），
    简单丢弃最旧的消息以避免上下文窗口溢出。

    策略：
    - 保留系统消息（包括总结消息也会被保留）
    - 保留最近的 10 条消息（5 轮对话）
    - 当消息少于等于 12 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 12:
        return None

    # 提取系统消息（包括总结消息）
    system_msgs = [
        msg for msg in messages if isinstance(msg, SystemMessage)
    ]

    # 提取非系统消息，保留最近的 10 条
    non_system_msgs = [
        msg for msg in messages if not isinstance(msg, SystemMessage)
    ]
    recent_non_system = non_system_msgs[-10:]

    # 构建新的消息列表
    new_messages = list(system_msgs) + list(recent_non_system)

    logger.warning(
        f"【兜底修剪】消息数 {len(messages)} -> {len(new_messages)} 条"
    )

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具（与 AIOps Planner/Executor 使用同一套默认本地工具）
        self.tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        for name, server in config.mcp_servers.items():
            hint = suggest_mcp_transport(
                str(server.get("url", "")),
                str(server.get("transport", "")),
            )
            if hint:
                logger.warning(f"MCP 配置 [{name}]: {hint}")

        mcp_client = await get_mcp_client_with_retry()
        mcp_tools, mcp_err = await load_mcp_tools_safe(mcp_client)
        if mcp_err:
            logger.warning(
                f"MCP 工具加载失败，将仅使用本地工具继续运行:\n{mcp_err}"
            )
            self.mcp_tools = []
        else:
            self.mcp_tools = mcp_tools
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        all_tools = self.tools + self.mcp_tools

        self.agent = create_agent(
            self.model, # 模型实例
            tools=all_tools, # 可用工具列表
            checkpointer=self.checkpointer, # 会话持久化
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    # ── 总结式记忆压缩：核心辅助方法 ────────────────────────────

    def _get_thread_config(self, session_id: str) -> Dict[str, Any]:
        """构造 checkpointer 需要的 config dict

        LangGraph 1.0+ 的 MemorySaver.put() 要求 configurable 中包含 checkpoint_ns，
        否则抛出 KeyError。根命名空间使用空字符串 ""。
        """
        return {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}

    def _read_checkpoint_messages(self, session_id: str) -> List[BaseMessage]:
        """从 checkpointer 读取指定会话的所有消息

        说明：
            checkpointer.get(config) 直接返回 Checkpoint dict，
            其结构为 {"channel_values": {"messages": [...], ...}, "v": ..., ...}

        Returns:
            消息列表（可能为空列表，表示该会话尚无历史）
        """
        config_obj = self._get_thread_config(session_id)
        checkpoint_data = self.checkpointer.get(config_obj)

        if not checkpoint_data:
            return []

        # checkpoint_data 就是一个 dict，直接从中取 messages
        if isinstance(checkpoint_data, dict):
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
        else:
            # 兜底：如果 checkpoint_data 是对象（某些版本的 LangGraph）
            messages = getattr(checkpoint_data, "messages", [])

        messages = list(messages) if messages else []

        # ── [Agent 安全 层 4] 记忆投毒检查 ──
        if getattr(config, "agent_safety_enabled", True) and messages:
            try:
                mem_result = content_safety_service.check_memory_messages(messages, session_id)
                if not mem_result.is_safe:
                    # 发现疑似被投毒的消息 —— 构造"干净"的 messages：
                    # 保留所有未被命中的消息，被命中的消息替换为空内容
                    cleaned_messages = []
                    suspicious_set = set(mem_result.suspicious_message_indices)
                    for idx, msg in enumerate(messages):
                        if idx in suspicious_set:
                            # 保留消息的类型结构，但把内容替换为空
                            new_msg = type(msg)(content="")
                            for attr in ("additional_kwargs", "response_metadata"):
                                if hasattr(msg, attr):
                                    setattr(new_msg, attr, getattr(msg, attr, {}))
                            cleaned_messages.append(new_msg)
                        else:
                            cleaned_messages.append(msg)
                    logger.warning(
                        f"[安全-记忆] 会话 {session_id}: 已清理 {len(suspicious_set)} 条疑似被投毒的消息"
                    )
                    messages = cleaned_messages
            except Exception as e:
                logger.error(f"[安全-记忆] 检查异常（fail-open）: {e}")

        return messages

    def _write_checkpoint_messages(
        self, session_id: str, new_messages: List[BaseMessage]
    ) -> bool:
        """把新消息列表写回 checkpointer

        关键：必须正确使用 LangGraph API
            - checkpointer.get_tuple(config) 返回 CheckpointTuple namedtuple
            - CheckpointTuple 有 .checkpoint（dict）和 .metadata（dict）等字段
            - checkpointer.put(config, checkpoint, metadata, new_versions) 需要 4 个参数

        Args:
            session_id: 会话 ID
            new_messages: 新的消息列表

        Returns:
            是否写入成功
        """
        try:
            config_obj = self._get_thread_config(session_id)

            # 用 get_tuple() 获取完整的 checkpoint 信息（包含 metadata 和 versions）
            cp_tuple = self.checkpointer.get_tuple(config_obj)

            if not cp_tuple:
                # 该会话尚无 checkpoint，不需要写回（下次调用 agent 时会自动创建）
                logger.info(
                    f"[会话 {session_id}] 无现有 checkpoint，跳过写回"
                )
                return True

            # 从 CheckpointTuple 中提取 checkpoint dict 和 metadata
            # 优先用属性名访问（CheckpointTuple 是 namedtuple）
            if hasattr(cp_tuple, "checkpoint"):
                old_checkpoint = cp_tuple.checkpoint
                old_metadata = getattr(cp_tuple, "metadata", {})
            elif isinstance(cp_tuple, tuple) and len(cp_tuple) >= 2:
                old_checkpoint = cp_tuple[0]
                old_metadata = cp_tuple[1]
            else:
                logger.warning(
                    f"[会话 {session_id}] 无法解析 checkpoint_tuple 结构: {type(cp_tuple)}"
                )
                return False

            # 确保 old_checkpoint 是 dict
            if not isinstance(old_checkpoint, dict):
                if hasattr(old_checkpoint, "__dict__"):
                    old_checkpoint = vars(old_checkpoint)
                else:
                    logger.warning(
                        f"[会话 {session_id}] checkpoint 不是 dict: {type(old_checkpoint)}"
                    )
                    return False

            # 构造新的 checkpoint（复制旧的，只替换 messages）
            new_checkpoint = dict(old_checkpoint)
            if "channel_values" not in new_checkpoint or not isinstance(
                new_checkpoint.get("channel_values"), dict
            ):
                new_checkpoint["channel_values"] = {}
            new_checkpoint["channel_values"]["messages"] = list(new_messages)

            # 为 messages channel 计算新的 version（每次修改都要递增）
            current_versions = new_checkpoint.get("channel_versions", {})
            if isinstance(current_versions, dict):
                current_msg_version = current_versions.get("messages", None)
            else:
                current_msg_version = None

            try:
                new_msg_version = self.checkpointer.get_next_version(
                    current_msg_version, None
                )
            except (TypeError, AttributeError):
                # 兜底：简单递增
                try:
                    new_msg_version = str(int(str(current_msg_version or "0")) + 1)
                except (ValueError, TypeError):
                    new_msg_version = "1"

            new_versions = dict(current_versions) if isinstance(current_versions, dict) else {}
            new_versions["messages"] = new_msg_version

            # 调用 put() 写回（4 个参数：config, checkpoint, metadata, new_versions）
            self.checkpointer.put(config_obj, new_checkpoint, old_metadata, new_versions)
            logger.info(
                f"[会话 {session_id}] checkpointer 已更新，新消息数={len(new_messages)}"
            )
            return True

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] 写回 checkpointer 失败: {format_exception_chain(e)}"
            )
            return False

    def _should_summarize(self, messages: List[BaseMessage]) -> bool:
        """判断当前消息列表是否需要触发总结

        规则：
        - 非系统消息（Human + AI + Tool）累计达到 N * 2 条（每轮 2 条）
        - 触发阈值由 config.summary_trigger_rounds 控制（默认 5 轮 = 10 条）
        - 每次触发后，旧消息会被压缩成 1 条总结消息 + 保留最近 5 轮
        - 因此下次触发是：总结消息 + 又积累 10 条非系统消息

        Args:
            messages: 当前 checkpointer 中的完整消息列表

        Returns:
            True 表示需要触发总结
        """
        non_system_count = sum(
            1 for msg in messages
            if not isinstance(msg, SystemMessage)
        )
        threshold = config.summary_trigger_rounds * 2  # 每轮 = 用户消息 + 助手消息
        return non_system_count >= threshold

    async def _summarize_and_update(self, session_id: str) -> bool:
        """执行总结并更新消息历史

        流程：
        1. 读取 checkpointer 中的当前消息
        2. 判断是否需要总结
        3. 需要则：提取已有总结 + 提取要压缩的消息
        4. 调用 SummaryAgent 生成新总结
        5. 构建新消息列表（新总结消息 + 最近 N 轮对话）
        6. 写回 checkpointer

        Args:
            session_id: 会话 ID

        Returns:
            True 表示执行了总结更新，False 表示未触发总结或执行失败
        """
        try:
            messages = self._read_checkpoint_messages(session_id)

            if not self._should_summarize(messages):
                return False

            logger.info(
                f"[会话 {session_id}] 触发总结："
                f"当前消息数={len(messages)}，"
                f"其中非系统消息={sum(1 for m in messages if not isinstance(m, SystemMessage))}"
            )

            # 分离已有总结和普通消息
            existing_summary, non_summary_messages = (
                SummaryAgent.extract_existing_summary(messages)
            )

            # 从非总结消息中：按时间顺序，前面的拿去总结，最近 N 轮保留
            # N = summary_trigger_rounds（即保留最近 5 轮 = 10 条非系统消息）
            keep_count = config.summary_trigger_rounds * 2

            non_system_msgs = [
                msg for msg in non_summary_messages
                if not isinstance(msg, SystemMessage)
            ]
            # 用户在对话中设置的系统提示（SystemMessage）也要保留
            system_msgs = [
                msg for msg in non_summary_messages
                if isinstance(msg, SystemMessage)
            ]

            if len(non_system_msgs) <= keep_count:
                # 理论上 _should_summarize 已经过滤了，这里做一个二次判断
                logger.info(
                    f"[会话 {session_id}] 实际非系统消息数 {len(non_system_msgs)} "
                    f"未达阈值 {keep_count}，跳过总结"
                )
                return False

            # 前面的消息（需要被总结压缩掉的）
            to_summarize = non_system_msgs[:-keep_count]
            # 后面的消息（保留的最近对话）
            to_keep = non_system_msgs[-keep_count:]

            logger.info(
                f"[会话 {session_id}] 将压缩 {len(to_summarize)} 条消息，"
                f"保留最近 {len(to_keep)} 条消息"
            )

            # 调用 SummaryAgent（支持增量总结）
            summary_text = await summary_agent.summarize(
                messages=to_summarize,
                existing_summary=existing_summary,
            )

            # 构建新消息列表：系统提示 + 总结消息 + 保留的最近对话
            summary_msg = summary_agent.build_summary_message(summary_text)
            new_messages = list(system_msgs) + [summary_msg] + list(to_keep)

            # 写回 checkpointer
            success = self._write_checkpoint_messages(session_id, new_messages)

            if success:
                logger.info(
                    f"[会话 {session_id}] 总结完成："
                    f"{len(messages)} 条 -> {len(new_messages)} 条"
                )
            return success

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] 总结过程异常: {format_exception_chain(e)}"
            )
            return False

    async def _recognize_intent(
        self, question: str, session_id: str
    ) -> Optional[IntentRecognitionResult]:
        """识别本轮意图并记录到意图轨迹

        位置：在 _summarize_and_update 之后调用——此时历史已被压缩，
        意图识别只接收「有界上下文」，与对话总长度无关（长上下文应对策略）：
          1. 最近 N 条消息（config.intent_recent_message_window）
          2. 已有总结文本（更早对话的语义压缩）
          3. 最近 2 条意图（来自 IntentTracker，判断 intent_shift / 依赖）

        受 config.intent_recognition_enabled 开关控制；关闭时返回 None，
        不产生任何额外 LLM 调用。失败时返回 None（IntentAgent 内部已保证
        recognize 不抛异常并返回 unknown 兜底，这里是再一层保险）。
        """
        if not config.intent_recognition_enabled:
            return None

        try:
            # 1. 取最近若干条对话消息（有界窗口）
            all_messages = self._read_checkpoint_messages(session_id)
            recent_messages = (
                all_messages[-config.intent_recent_message_window :]
                if all_messages
                else []
            )

            # 2. 取已有总结（若已触发过压缩），作为更早对话的语义压缩
            existing_summary, _ = SummaryAgent.extract_existing_summary(all_messages)

            # 3. 取最近 2 条意图，供判断 intent_shift 与跨轮依赖
            previous_intents = intent_tracker.get_recent(session_id, k=2)

            # 4. 调用意图识别
            result = await intent_agent.recognize(
                query=question,
                recent_messages=recent_messages,
                previous_intents=previous_intents,
                existing_summary=existing_summary,
            )

            # 5. 记录轨迹（供下一轮 shift 判断 + 会话历史接口返回）
            intent_tracker.record(session_id, result)

            logger.info(
                f"[会话 {session_id}] 意图识别: "
                f"{result.primary_intent.intent_type.value}, "
                f"多意图={result.is_multi_intent}, 复杂={result.is_complex}, "
                f"切换={result.intent_shift}"
            )
            return result
        except Exception as e:
            logger.error(
                f"[会话 {session_id}] 意图识别流程异常: {format_exception_chain(e)}"
            )
            return None

    def _build_system_prompt(self, intent_context: Optional[str] = None) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Args:
            intent_context: 本轮意图识别结果文本块（来自 IntentRecognitionResult.format_for_prompt）。
                传入时会追加到系统提示词末尾，帮助 Agent 选择工具、处理多意图/复杂意图；
                不传或为空则生成不带意图段的基础提示词。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        base_prompt = dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

        # 意图识别段是「每轮临时」注入的，不作为独立消息持久化进 checkpointer，
        # 仅通过本系统提示词传递给本轮 Agent，避免污染/堆积对话历史。
        if intent_context and intent_context.strip():
            base_prompt = f"{base_prompt}\n\n{intent_context}"

        return base_prompt

    @timed_metric(service="rag", method="query")
    async def query(
        self,
        question: str,
        session_id: str,
        enable_web_search: bool = False,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        流程：
          1. 总结压缩 + 意图识别
          2. 若为多意图/复杂意图：走编排路径（拆分子任务分别回答再汇总）
          3. 否则：走原 Agent 路径

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            enable_web_search: 是否启用网络搜索（用户手动触发）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            # ── [Agent 安全 层 1] 输入安全检查（用户提问） ──
            if getattr(config, "agent_safety_enabled", True):
                input_result = content_safety_service.check_user_input(question, session_id)
                if not input_result.is_safe:
                    logger.warning(
                        f"[会话 {session_id}] 用户输入被安全系统拦截: "
                        f"reason={input_result.reason}, keywords={input_result.blocked_keywords}"
                    )
                    return "抱歉，你的问题包含不安全内容，我无法提供帮助。请换一个问题试试。"

            # ── 关键步骤：对话历史总结压缩 ──
            await self._summarize_and_update(session_id)

            # ── 关键步骤：多轮意图识别 ──
            intent_result = await self._recognize_intent(question, session_id)

            # ── 新增：意图驱动自动分解编排 ──
            # 延迟 import 避免循环依赖（orchestrator 内部会导入 rag_agent_service）
            from app.agent.orchestrator import orchestrator

            if orchestrator.should_orchestrate(intent_result) and intent_result:
                logger.info(
                    f"[会话 {session_id}] 命中编排条件，进入意图驱动分解编排"
                )

                # 汇总阶段的 content 事件不带 subtask_index，把它们累积起来作为最终答案
                final_parts: List[str] = []
                fallback = False
                async for evt in orchestrator.orchestrate(
                    intent_result,
                    original_question=question,
                    session_id=session_id,
                    enable_web_search=enable_web_search,
                    streaming=True,
                ):
                    evt_type = evt.get("type")
                    if evt_type == "content":
                        # 子任务流式内容带 subtask_index；汇总阶段流式内容不带
                        if evt.get("subtask_index") is None:
                            final_parts.append(evt.get("data", "") or "")
                    elif evt_type == "orchestration_summary":
                        data = evt.get("data") or {}
                        if data.get("fallback") or data.get("summary_mode") == "fallback":
                            fallback = True

                if not fallback and final_parts:
                    final_answer = "".join(final_parts).strip()
                    if final_answer:
                        # ── [Agent 安全 层 5] 输出安全检查（最终回答） ──
                        if getattr(config, "agent_safety_enabled", True):
                            output_result = content_safety_service.check_output(final_answer, session_id)
                            if not output_result.is_safe:
                                logger.warning(
                                    f"[会话 {session_id}] 编排路径回答被安全系统拦截: "
                                    f"reason={output_result.reason}"
                                )
                                final_answer = output_result.sanitized_answer
                        logger.info(
                            f"[会话 {session_id}] 编排路径返回最终回答（长度 {len(final_answer)}）"
                        )
                        return final_answer

                logger.warning(
                    f"[会话 {session_id}] 编排未能产出有效回答，回退到原 Agent"
                )

            # ── 原路径：单意图/简单问题，直接交给 Agent 处理 ──
            intent_context = (
                intent_result.format_for_prompt() if intent_result else None
            )
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            if enable_web_search and config.web_search_enabled:
                web_search_instruction = (
                    "\n【网络搜索】用户已启用网络搜索，请在检索知识库的同时也使用 web_search 工具搜索网络信息，"
                    "综合两方面的结果进行回答。"
                )
                intent_context = (intent_context or "") + web_search_instruction

            system_prompt = self._build_system_prompt(intent_context)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                # ── [Agent 安全 层 5] 输出安全检查（最终回答） ──
                if getattr(config, "agent_safety_enabled", True):
                    output_result = content_safety_service.check_output(answer, session_id)
                    if not output_result.is_safe:
                        logger.warning(
                            f"[会话 {session_id}] 原 Agent 路径回答被安全系统拦截: "
                            f"reason={output_result.reason}"
                        )
                        answer = output_result.sanitized_answer

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（非流式）: "
                f"{format_exception_chain(e)}"
            )
            raise

    @timed_metric(service="rag", method="query_stream")
    async def query_stream(
        self,
        question: str,
        session_id: str,
        enable_web_search: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        流程：
          1. 总结压缩 + 意图识别，先 yield 意图事件
          2. 若为多意图/复杂意图：走编排路径（子任务进度 -> 汇总 -> 最终答案流式）
          3. 否则：走原 Agent 路径

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "intent" | "orchestration_step" | "orchestration_summary" |
                          "content" | "complete" | "error" | "safety_blocked"
        """
        try:
            await self._initialize_agent()

            # ── [Agent 安全 层 1] 输入安全检查（用户提问） ──
            if getattr(config, "agent_safety_enabled", True):
                input_result = content_safety_service.check_user_input(question, session_id)
                if not input_result.is_safe:
                    logger.warning(
                        f"[会话 {session_id}] 用户输入被安全系统拦截: "
                        f"reason={input_result.reason}, keywords={input_result.blocked_keywords}"
                    )
                    yield {
                        "type": "safety_blocked",
                        "data": {
                            "reason": input_result.reason,
                            "blocked_keywords": input_result.blocked_keywords,
                            "hint": "请换一个问题试试",
                        }
                    }
                    return

            # ── 关键步骤：对话历史总结压缩 ──
            await self._summarize_and_update(session_id)

            # ── 关键步骤：多轮意图识别 ──
            intent_result = await self._recognize_intent(question, session_id)
            intent_context = (
                intent_result.format_for_prompt() if intent_result else None
            )
            if intent_result:
                yield {"type": "intent", "data": intent_result.to_dict()}

            # ── 新增：意图驱动自动分解编排（流式） ──
            from app.agent.orchestrator import orchestrator

            if orchestrator.should_orchestrate(intent_result) and intent_result:
                logger.info(
                    f"[会话 {session_id}] 流式命中编排条件，进入意图驱动分解编排"
                )

                # orchestrator 产生的事件：
                #  - orchestration_start: 编排开始与子任务清单
                #  - orchestration_step: 某子任务 running/done/failed
                #  - content (subtask_index=N): 某子任务流式生成的内容
                #  - content (无 subtask_index): 汇总阶段 LLM 生成的最终回答
                #  - orchestration_summary: 汇总模式与耗时，fallback=True 需回退
                need_fallback = False
                got_summary_content = False
                async for evt in orchestrator.orchestrate(
                    intent_result,
                    original_question=question,
                    session_id=session_id,
                    enable_web_search=enable_web_search,
                    streaming=True,
                ):
                    evt_type = evt.get("type")
                    if evt_type == "orchestration_summary":
                        data = evt.get("data") or {}
                        if data.get("fallback") or data.get("summary_mode") == "fallback":
                            need_fallback = True
                        # 把 summary 作为信息事件也透传给前端，便于展示
                        yield evt
                    elif evt_type == "content":
                        # 子任务流式内容（带 subtask_index）与
                        # 汇总阶段流式内容（无 subtask_index）都原样透传
                        if evt.get("subtask_index") is None:
                            got_summary_content = True
                        yield evt
                    else:
                        # orchestration_start / orchestration_step 等
                        yield evt

                if got_summary_content and not need_fallback:
                    # ── [Agent 安全 层 5] 输出安全检查：重新从 orchestrator 收集内容并检查 ──
                    # 为确保安全，我们重新跑一次 orchestrator 收集完整答案，然后检查后再流式输出
                    # （如果不需要安全检查，上面的流式 content 已经直接输出了）
                    if getattr(config, "agent_safety_enabled", True):
                        logger.info(
                            f"[会话 {session_id}] 编排路径做输出安全检查（重新收集答案）"
                        )
                        re_parts: List[str] = []
                        async for evt2 in orchestrator.orchestrate(
                            intent_result, original_question=question,
                            session_id=session_id,
                            enable_web_search=enable_web_search, streaming=True,
                        ):
                            if evt2.get("type") == "content" and evt2.get("subtask_index") is None:
                                re_parts.append(evt2.get("data", "") or "")
                        collected = "".join(re_parts).strip()
                        if collected:
                            output_result = content_safety_service.check_output(collected, session_id)
                            if not output_result.is_safe:
                                logger.warning(
                                    f"[会话 {session_id}] 编排路径回答被安全系统拦截，替换为兜底回答"
                                )
                                collected = output_result.sanitized_answer
                            # 检查通过后，以流式方式输出最终回答
                            yield {"type": "content", "data": collected}
                            yield {"type": "complete"}
                            return
                    logger.info(
                        f"[会话 {session_id}] 编排路径完成，最终回答已流式输出"
                    )
                    yield {"type": "complete"}
                    return

                logger.warning(
                    f"[会话 {session_id}] 编排未能产出有效回答，回退到原 Agent 流式"
                )

            # ── 原路径：单意图/简单问题，直接交给 Agent 流式处理 ──
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            if enable_web_search and config.web_search_enabled:
                web_search_instruction = (
                    "\n【网络搜索】用户已启用网络搜索，请在检索知识库的同时也使用 web_search 工具搜索网络信息，"
                    "综合两方面的结果进行回答。"
                )
                intent_context = (intent_context or "") + web_search_instruction

            system_prompt = self._build_system_prompt(intent_context)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # ── 策略：先收集完整 answer，做安全检查后再流式输出 ──
            # （如果 agent_safety_enabled=False，则直接流式输出，保持性能）
            if getattr(config, "agent_safety_enabled", True):
                # 安全模式：收集完整回答 -> 检查 -> 输出
                collected_parts: List[str] = []
                async for token, metadata in self.agent.astream(
                    input=agent_input,
                    config=config_dict,
                    stream_mode="messages",
                ):
                    message_type = type(token).__name__
                    if message_type in ("AIMessageChunk",):
                        content_blocks = getattr(token, 'content_blocks', None)
                        if content_blocks and isinstance(content_blocks, list):
                            for block in content_blocks:
                                if isinstance(block, dict) and block.get('type') == 'text':
                                    collected_parts.append(block.get('text', ''))
                collected_answer = "".join(collected_parts).strip()
                # [Agent 安全 层 5] 输出安全检查
                if collected_answer:
                    output_result = content_safety_service.check_output(collected_answer, session_id)
                    if not output_result.is_safe:
                        logger.warning(
                            f"[会话 {session_id}] 原 Agent 路径回答被安全系统拦截: "
                            f"reason={output_result.reason}"
                        )
                        collected_answer = output_result.sanitized_answer
                # 以 chunk 方式模拟流式输出（约 80 字符一段）
                for i in range(0, len(collected_answer), 80):
                    yield {"type": "content", "data": collected_answer[i:i + 80]}
            else:
                # 非安全模式：直接流式输出
                async for token, metadata in self.agent.astream(
                    input=agent_input,
                    config=config_dict,
                    stream_mode="messages",
                ):
                    node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                    message_type = type(token).__name__

                    logger.info(
                        f"[流式调试] type={message_type}, node={node_name}, "
                        f"content={repr(getattr(token, 'content', None)[:200]) if hasattr(token, 'content') and isinstance(getattr(token, 'content', str), str) else repr(getattr(token, 'content', None))}, "
                        f"tool_calls={getattr(token, 'tool_calls', None)}, "
                        f"tool_call_chunks={getattr(token, 'tool_call_chunks', None)}"
                    )

                    if message_type in ("AIMessageChunk",):
                        content_blocks = getattr(token, 'content_blocks', None)

                        if content_blocks and isinstance(content_blocks, list):
                            for block in content_blocks:
                                if isinstance(block, dict) and block.get('type') == 'text':
                                    text_content = block.get('text', '')
                                    if text_content:
                                        yield {
                                            "type": "content",
                                            "data": text_content,
                                            "node": node_name
                                        }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            detail = format_exception_chain(e)
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（流式）: {detail}"
            )
            yield {"type": "error", "data": detail}

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        说明：
            - 普通 SystemMessage（Agent 内部的系统提示）会被跳过
            - "对话历史总结"消息会被保留并标记 role="summary"
            - 用户消息和助手消息正常返回

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{
                "role": "user|assistant|summary",
                "content": "...",
                "timestamp": "..."
            }]
        """
        try:
            messages = self._read_checkpoint_messages(session_id)

            if not messages:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # 判断是否为总结消息
                if (
                    isinstance(msg, SystemMessage)
                    and isinstance(content, str)
                    and content.startswith(SummaryAgent.SUMMARY_PREFIX)
                ):
                    # 总结消息：去掉前缀标记后返回
                    summary_content = content[len(SummaryAgent.SUMMARY_PREFIX):].strip()
                    timestamp = getattr(msg, 'timestamp', None)
                    history.append({
                        "role": "summary",
                        "content": summary_content,
                        "timestamp": timestamp or ""
                    })
                    continue

                # 普通 SystemMessage：跳过（Agent 内部系统提示，不需要暴露给前端）
                if isinstance(msg, SystemMessage):
                    continue

                # 用户消息 / 助手消息
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(
                f"获取会话历史: {session_id}, 消息数量: {len(history)}"
                f"（其中 {sum(1 for h in history if h['role'] == 'summary')} 条为总结消息）"
            )
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def get_session_intents(self, session_id: str) -> list:
        """获取某会话的意图识别轨迹（供会话历史接口返回，前端可展示）

        Args:
            session_id: 会话 ID

        Returns:
            list: 意图结果字典列表（按时间顺序），每项为 IntentRecognitionResult.to_dict()
        """
        try:
            intents = intent_tracker.get_all(session_id)
            return [r.to_dict() for r in intents]
        except Exception as e:
            logger.error(f"获取会话意图轨迹失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            # 使用 checkpointer 的 delete_thread 方法删除该 thread 的所有检查点
            self.checkpointer.delete_thread(session_id)
            # 同步清空意图识别轨迹（与对话历史同生命周期）
            intent_tracker.clear(session_id)

            logger.info(f"已清除会话历史: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)