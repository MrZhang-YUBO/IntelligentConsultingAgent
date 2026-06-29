"""多轮意图识别 Agent - 在每轮对话前识别用户意图，辅助 Agent 选择工具与处理多意图/复杂意图

设计上严格复刻 `app/agent/summary_agent.py` 的"独立辅助 Agent"模式：
- 独立的轻量模型（qwen-turbo，低温度，不开流式）
- 在 `rag_agent_service.query / query_stream` 中、`_summarize_and_update` 之后被调用
- 失败时返回兜底结果，绝不中断主对话

与 SummaryAgent 的区别：
- SummaryAgent 输出纯文本摘要并写回 checkpointer；
- IntentAgent 输出结构化 `IntentRecognitionResult`（Pydantic），不写回 checkpointer，
  而是注入「本轮系统提示词」+ 存入内存 `IntentTracker`，避免污染对话历史。

结构化输出复刻 `app/agent/aiops/planner.py` 的 `ChatQwen(...).with_structured_output(Pydantic)` 用法。
"""

import json
import re
import threading
from collections import deque
from enum import Enum
from textwrap import dedent
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config


# ── 意图分类体系（8 类，映射到现有工具/能力）──────────────────────────────

class IntentType(str, Enum):
    """意图类型枚举（str 继承便于 JSON 序列化与 prompt 中直接使用）"""
    KNOWLEDGE_QA = "knowledge_qa"            # 知识库问答
    DOCUMENT_OP = "document_op"              # 文档上传/列表/删除
    AIOPS_DIAGNOSE = "aiops_diagnose"        # 运维诊断/告警排查
    WEB_SEARCH = "web_search"                # 网络搜索（用户明确要求搜索网络）
    COMPARISON = "comparison"                # 对比分析
    MULTI_STEP_TASK = "multi_step_task"      # 多步复杂任务
    CLARIFICATION = "clarification"          # 澄清/追问（依赖上文）
    CHITCHAT = "chitchat"                    # 闲聊/问候
    UNKNOWN = "unknown"                      # 无法判定


class SubIntent(BaseModel):
    """单个子意图"""
    intent_type: IntentType = Field(description="意图类型，必须是 IntentType 中的值")
    description: str = Field(description="该子意图想做什么（一句话）")
    entities: List[str] = Field(default_factory=list, description="关键实体，如产品名/指标名/文档名")
    suggested_tools: List[str] = Field(default_factory=list, description="建议使用的工具名，无则留空")
    depends_on: Optional[str] = Field(
        default=None,
        description="复杂意图：依赖上文哪一轮或哪个意图，如 '上一轮的产品A对比'；无依赖则 null",
    )


class IntentRecognitionResult(BaseModel):
    """一轮意图识别的完整结果"""
    primary_intent: SubIntent = Field(description="主意图")
    secondary_intents: List[SubIntent] = Field(
        default_factory=list, description="多意图时的其余子意图"
    )
    is_multi_intent: bool = Field(default=False, description="是否包含多个意图")
    is_complex: bool = Field(default=False, description="是否复杂意图（条件/依赖/多步）")
    context_references: List[str] = Field(
        default_factory=list, description="引用了上文哪些内容（代词消解线索，如 '产品A'）"
    )
    intent_shift: bool = Field(default=False, description="相对上一轮是否发生意图切换")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度 0~1")
    reasoning: str = Field(default="", description="简短判断理由（可观测）")

    def format_for_prompt(self) -> str:
        """生成本轮意图的精简文本块，用于注入 Agent 系统提示词。"""
        lines: List[str] = ["【本轮意图识别（供你参考以选择工具、处理多意图/复杂意图）】"]
        p = self.primary_intent
        lines.append(
            f"主意图: {p.intent_type.value}（{p.description}）| 置信度: {self.confidence:.2f}"
        )
        if p.entities:
            lines.append(f"  实体: {', '.join(p.entities)}")
        if p.suggested_tools:
            lines.append(f"  建议工具: {', '.join(p.suggested_tools)}")
        if self.is_multi_intent and self.secondary_intents:
            lines.append(f"多意图: 是（共 {1 + len(self.secondary_intents)} 个）")
            for i, s in enumerate(self.secondary_intents, start=2):
                dep = f" — 依赖: {s.depends_on}" if s.depends_on else ""
                lines.append(f"  - 子意图{i}: {s.intent_type.value}（{s.description}）{dep}")
        else:
            lines.append("多意图: 否")
        lines.append(
            f"复杂意图: {'是' if self.is_complex else '否'} | "
            f"意图切换: {'是' if self.intent_shift else '否'}"
        )
        if self.context_references:
            lines.append(f"引用上文: {', '.join(self.context_references)}")
        if self.reasoning:
            lines.append(f"理由: {self.reasoning}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化为可放入 SSE / JSON 响应的纯字典（枚举转字符串）。"""
        return self.model_dump(mode="json")


# ── 意图识别 Agent ──────────────────────────────────────────────────────

class IntentAgent:
    """多轮意图识别 Agent"""

    # 单条消息截断长度（与 SummaryAgent 一致，防止长上下文撑爆识别模型）
    _MAX_MSG_CHARS = 1500

    def __init__(self):
        self.model = ChatQwen(
            model=config.intent_model,
            api_key=config.dashscope_api_key,
            temperature=0,        # 意图识别需要确定性，温度设为 0
            streaming=False,
        )
        self.system_prompt = self._build_system_prompt()
        logger.info(f"意图 Agent 初始化完成，模型={config.intent_model}")

    @staticmethod
    def _build_system_prompt() -> str:
        return dedent("""
            你是一个"多轮意图识别器"，任务是根据用户当前问题与对话上下文，识别用户的真实意图。

            ## 意图分类（只能从以下 9 类中选择）
            - knowledge_qa：知识库问答（查阅文档/概念解释/原理说明）
            - document_op：文档操作（上传/列出/删除文档）
            - aiops_diagnose：运维诊断（告警排查/根因分析/指标查询）
            - web_search：网络搜索（用户明确要求搜索网络/查询最新信息/实时数据）
            - comparison：对比分析（多个实体的比较）
            - multi_step_task：多步复杂任务（需多步或多工具组合完成）
            - clarification：澄清/追问（承接上文，如"它""那个""继续"）
            - chitchat：闲聊/问候
            - unknown：信息不足，无法判定

            ## 识别要求
            1. 多意图：一句话含多个诉求时，主诉求放 primary_intent，其余放 secondary_intents，并置 is_multi_intent=true。
            2. 复杂意图：含条件分支/跨轮依赖/多步骤时置 is_complex=true，并用 depends_on 标注依赖上文哪个意图或哪一轮。
            3. 多轮衔接：结合"最近识别过的意图"判断 intent_shift（是否切换话题）；用 context_references 记录引用了上文哪些实体或内容（用于代词消解，如"它"指代什么）。
            4. entities：抽取关键实体（产品名/指标名/文档名等）。
            5. suggested_tools：按意图给出建议工具名，可选值：retrieve_knowledge / web_search / get_current_time / query_prometheus_alerts / mcp_cls / mcp_monitor；无需工具则留空。
            6. confidence：0~1 置信度；reasoning：一句话理由。
        """).strip()

    @classmethod
    def _format_history(cls, messages: List[BaseMessage]) -> str:
        """格式化最近对话（每条截断到 _MAX_MSG_CHARS），供识别模型参考。"""
        if not messages:
            return "（无）"
        lines: List[str] = []
        for idx, msg in enumerate(messages, start=1):
            if isinstance(msg, HumanMessage):
                role_str = "用户"
            elif isinstance(msg, AIMessage):
                role_str = "助手"
            else:
                continue  # 跳过 SystemMessage / ToolMessage，意图识别只关心对话本身
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, str) and len(content) > cls._MAX_MSG_CHARS:
                content = content[: cls._MAX_MSG_CHARS] + "\n...(已截断)..."
            lines.append(f"[{idx}] {role_str}: {content}")
        return "\n".join(lines) if lines else "（无）"

    @staticmethod
    def _format_previous_intents(intents: List["IntentRecognitionResult"]) -> str:
        """把最近识别过的意图紧凑序列化，供判断 intent_shift 与依赖。"""
        if not intents:
            return "（无）"
        lines: List[str] = []
        for i, r in enumerate(intents, start=1):
            p = r.primary_intent
            shift = "切换" if r.intent_shift else "延续"
            multi = f"，含{len(r.secondary_intents)}个副意图" if r.is_multi_intent else ""
            lines.append(
                f"[近{i}] {p.intent_type.value}（{p.description}）| {shift}{multi}"
            )
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        query: str,
        existing_summary: Optional[str],
        recent_messages: List[BaseMessage],
        previous_intents: List["IntentRecognitionResult"],
    ) -> str:
        return dedent(f"""
            当前用户问题：
            {query}

            已有对话总结（若有，是更早对话的压缩摘要）：
            {existing_summary or "（无）"}

            最近对话（最近 {config.intent_recent_message_window} 条）：
            {self._format_history(recent_messages)}

            最近识别过的意图（若有，用于判断意图切换与依赖）：
            {self._format_previous_intents(previous_intents)}

            请识别本轮意图。
        """).strip()

    def _unknown_result(self, query: str) -> IntentRecognitionResult:
        """兜底结果：识别失败或信息不足时返回，确保主对话不中断。"""
        return IntentRecognitionResult(
            primary_intent=SubIntent(
                intent_type=IntentType.UNKNOWN,
                description="意图识别失败或信息不足",
            ),
            confidence=0.0,
            reasoning="识别异常或上下文不足，已降级",
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从模型输出中提取 JSON 对象（兼容 ```json 代码块包裹）。"""
        if not isinstance(text, str):
            raise ValueError(f"非字符串输出: {type(text)}")
        # 优先取 ```json ... ``` 代码块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = m.group(1) if m else text
        # 否则取第一个 {...} 片段
        if not m:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start : end + 1]
        return json.loads(candidate)

    async def recognize(
        self,
        query: str,
        recent_messages: List[BaseMessage],
        previous_intents: Optional[List[IntentRecognitionResult]] = None,
        existing_summary: Optional[str] = None,
    ) -> IntentRecognitionResult:
        """识别本轮意图

        Args:
            query: 当前用户问题
            recent_messages: 最近若干条对话消息（已由调用方裁剪到有界窗口）
            previous_intents: 最近识别过的意图（来自 IntentTracker）
            existing_summary: 已有对话总结文本（来自 SummaryAgent）

        Returns:
            IntentRecognitionResult：失败时返回 unknown 兜底，绝不抛异常
        """
        if not query or not query.strip():
            return self._unknown_result(query)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_user_prompt(
                query, existing_summary, recent_messages, previous_intents or []
            )),
        ]

        # 第一优先：结构化输出（function calling / tool schema）
        try:
            chain = self.model.with_structured_output(IntentRecognitionResult)
            result = await chain.ainvoke(messages)
            if isinstance(result, IntentRecognitionResult):
                logger.info(
                    f"意图识别完成: 主意图={result.primary_intent.intent_type.value}, "
                    f"多意图={result.is_multi_intent}, 复杂={result.is_complex}, "
                    f"置信度={result.confidence:.2f}"
                )
                return result
            # 某些版本可能返回 dict
            return IntentRecognitionResult.model_validate(result)
        except Exception as e:
            logger.warning(f"意图识别结构化输出失败，尝试 JSON 兜底: {e}")

        # 第二优先：让模型直接输出 JSON 再解析
        try:
            json_messages = messages + [
                SystemMessage(content="请严格只输出一个 JSON 对象，不要任何解释或代码块标记。")
            ]
            resp = await self.model.ainvoke(json_messages)
            data = self._parse_json(resp.content if hasattr(resp, "content") else str(resp))
            result = IntentRecognitionResult.model_validate(data)
            logger.info(
                f"意图识别完成(JSON兜底): 主意图={result.primary_intent.intent_type.value}"
            )
            return result
        except Exception as e2:
            logger.error(f"意图识别彻底失败，返回 unknown: {e2}")
            return self._unknown_result(query)


# ── 意图轨迹追踪（内存，按会话隔离，与 MemorySaver 同生命周期）──────────────

class IntentTracker:
    """按 session_id 维护意图识别轨迹的内存存储

    与 MemorySaver 一样是纯内存态，服务重启即丢失——这是有意的，保持与现有记忆机制一致。
    使用 threading.Lock 保护，多线程安全（Consumer 后台线程与请求线程可能并发）。
    """

    def __init__(self):
        self._max_size = config.intent_history_size
        self._store: dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, session_id: str, result: IntentRecognitionResult) -> None:
        """记录一轮意图识别结果"""
        with self._lock:
            dq = self._store.get(session_id)
            if dq is None:
                dq = deque(maxlen=self._max_size)
                self._store[session_id] = dq
            dq.append(result)

    def get_recent(
        self, session_id: str, k: int = 2
    ) -> List[IntentRecognitionResult]:
        """取最近 k 条意图（供下一轮识别判断 intent_shift）"""
        if k <= 0:
            return []
        with self._lock:
            dq = self._store.get(session_id)
            if not dq:
                return []
            return list(dq)[-k:]

    def get_all(self, session_id: str) -> List[IntentRecognitionResult]:
        """取该会话的全部意图轨迹（供会话历史接口返回）"""
        with self._lock:
            dq = self._store.get(session_id)
            return list(dq) if dq else []

    def clear(self, session_id: str) -> None:
        """清空某会话的意图轨迹（与 clear_session 联动）"""
        with self._lock:
            self._store.pop(session_id, None)


# 全局单例（与 summary_agent 风格一致）
intent_agent = IntentAgent()
intent_tracker = IntentTracker()
