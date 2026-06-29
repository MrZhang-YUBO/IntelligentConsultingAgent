"""总结对话 Agent - 当对话超过 N 轮时对历史对话进行总结压缩"""

from textwrap import dedent
from typing import List, Tuple, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config


class SummaryAgent:
    """对话历史总结 Agent"""

    SUMMARY_PREFIX = "【对话历史总结】"

    def __init__(self):
        self.model = ChatQwen(
            model=config.summary_model,
            api_key=config.dashscope_api_key,
            temperature=0.3,
            streaming=False,
        )
        self.system_prompt = self._build_system_prompt()
        logger.info(f"总结 Agent 初始化完成，模型={config.summary_model}")

    @staticmethod
    def _build_system_prompt() -> str:
        return dedent("""
            你是一个"对话历史压缩员"，任务是把一段多轮对话总结成一段简明摘要。
            总结规则：
            1. 用中文输出，字数控制在 300~500 字之间
            2. 保留关键信息：用户的核心诉求、Agent 得出的关键结论、已使用的工具及其返回的关键数据
            3. 忽略问候语、重复确认、闲聊等无信息价值的内容
            4. 以第三人称客观描述
            5. 不要输出多余的解释、不要分点
            输出格式：直接输出摘要文本，不需要任何前缀。
        """).strip()

    @staticmethod
    def _format_conversation(messages: List[BaseMessage]) -> str:
        lines: List[str] = []
        for idx, msg in enumerate(messages, start=1):
            if isinstance(msg, HumanMessage):
                role_str = "用户"
            elif isinstance(msg, AIMessage):
                role_str = "助手"
            else:
                role_str = type(msg).__name__
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, str) and len(content) > 1500:
                content = content[:1500] + "\n...(已截断)..."
            lines.append(f"[{idx}] {role_str}: {content}")
        return "\n".join(lines)

    async def summarize(
        self,
        messages: List[BaseMessage],
        existing_summary: Optional[str] = None,
    ) -> str:
        if not messages:
            logger.warning("SummaryAgent.summarize 收到空消息列表")
            return ""
        conversation_text = self._format_conversation(messages)
        if existing_summary and existing_summary.strip():
            user_prompt = dedent(f"""
                以下是已有的对话历史总结：
                ---
                {existing_summary}
                ---
                以下是之后发生的新对话：
                ---
                {conversation_text}
                ---
                请把"已有总结"和"新对话"合并，生成一个新的完整总结。
                直接输出总结文本。
            """).strip()
        else:
            user_prompt = dedent(f"""
                以下是一段对话历史，请生成总结：
                ---
                {conversation_text}
                ---
                直接输出总结文本。
            """).strip()
        try:
            logger.info(
                f"调用总结 Agent，消息数={len(messages)}，"
                f"是否增量={'是' if existing_summary else '否'}"
            )
            response = await self.model.ainvoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ])
            summary_text = response.content if hasattr(response, "content") else str(response)
            summary_text = summary_text.strip()
            logger.info(f"总结完成，总结文本长度={len(summary_text)} 字符")
            return summary_text
        except Exception as e:
            logger.error(f"总结 Agent 调用失败: {e}")
            return f"此前共进行了 {len(messages)} 条消息的对话，因异常未能生成详细总结。"

    def build_summary_message(self, summary_text: str) -> SystemMessage:
        return SystemMessage(content=f"{self.SUMMARY_PREFIX}\n{summary_text}")

    @staticmethod
    def extract_existing_summary(messages: List[BaseMessage]) -> Tuple[str, List[BaseMessage]]:
        summary_text = ""
        remaining: List[BaseMessage] = []
        for msg in messages:
            if (
                isinstance(msg, SystemMessage)
                and isinstance(msg.content, str)
                and msg.content.startswith(SummaryAgent.SUMMARY_PREFIX)
            ):
                summary_text = msg.content[len(SummaryAgent.SUMMARY_PREFIX):].strip()
                continue
            remaining.append(msg)
        return summary_text, remaining


summary_agent = SummaryAgent()