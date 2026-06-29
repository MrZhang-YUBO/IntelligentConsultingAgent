"""多意图编排器（MultiIntentOrchestrator）— v2.0

当意图识别结果表明是多意图（is_multi_intent=True）或复杂意图
（is_complex=True）时，本模块把原问题拆解成 N 个子任务，
**每个子任务仅执行工具调用（不再调用 LLM 生成）**，
全部并行执行，最后用一个轻量 LLM 把工具结果流式地汇总成最终回答。

与 v1.x 的核心区别：
  v1.x: 每个子任务走完整 Agent 链路（意图识别→工具→生成）→ 有 N+1 次 LLM
  v2.0: 每个子任务只执行工具（知识库检索/网络检索/时间/告警） → 只有 1 次汇总 LLM
        → 延迟显著降低

事件协议（更简洁）：
  - orchestration_start:  进入编排，宣告子任务总数与清单（前端只显示一行进度）
  - orchestration_summary:  编排完成，汇总模式与耗时
  - content (无 subtask_index):  汇总阶段 LLM 正在生成的最终回答 token
"""

import asyncio
import time
from textwrap import dedent
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.agent.intent_agent import IntentRecognitionResult, SubIntent
from app.config import config
from app.services.content_safety_service import content_safety_service
from app.tools import (
    get_current_time,
    query_prometheus_alerts,
    retrieve_knowledge,
    web_search,
)


# ── 工具名字符串 → 实际可调用函数（与 intent_agent prompt 中 suggested_tools 取值对齐） ──
TOOL_REGISTRY: Dict[str, Any] = {
    "retrieve_knowledge": retrieve_knowledge,
    "web_search": web_search,
    "get_current_time": get_current_time,
    "query_prometheus_alerts": query_prometheus_alerts,
    # 以下为 MCP 工具名的兜底映射——若意图识别输出这些字符串，我们在子任务里忽略它们
    # （因为 MCP 工具需要 MCP 客户端，这里宁可把它当成"无工具"，依赖主 Agent 兜底）
    "mcp_cls": None,
    "mcp_monitor": None,
}


class SubTask(BaseModel):
    index: int
    intent_type: str = Field(description="子意图类型")
    entities: List[str] = Field(default_factory=list)
    question: str = Field(description="该子任务的独立问题（用作工具 query）")
    suggested_tools: List[str] = Field(default_factory=list, description="来自意图识别的建议工具")
    tool_outputs: Dict[str, str] = Field(default_factory=dict, description="每个工具的返回文本，key=工具名")
    error: Optional[str] = Field(default=None)
    execution_time_ms: float = Field(default=0.0)

    @property
    def has_any_tool_result(self) -> bool:
        return bool(self.tool_outputs) and any(
            (v or "").strip() not in ("", "没有找到相关信息。") for v in self.tool_outputs.values()
        )

    def format_for_prompt(self) -> str:
        """用于汇总 LLM 阅读：把子任务的意图、用到的工具、工具输出紧凑地呈现。"""
        lines = [f"【子任务 {self.index}】[意图={self.intent_type}] {self.question}"]
        if self.entities:
            lines.append(f"  关键实体: {', '.join(self.entities)}")
        if self.suggested_tools:
            lines.append(f"  建议工具: {', '.join(self.suggested_tools)}")
        if self.error:
            lines.append(f"  状态: 失败（{self.error}）")
        elif self.tool_outputs:
            lines.append("  工具输出:")
            for tool_name, text in self.tool_outputs.items():
                body = (text or "").strip()
                if not body:
                    body = "（空）"
                # 控制单个工具输出的上限长度，避免上下文爆掉
                if len(body) > 1800:
                    body = body[:1800] + "\n...(过长已截断)..."
                lines.append(f"    * [{tool_name}] {body}")
        else:
            lines.append("  状态: 未调用工具（无需检索，直接依赖意图描述）")
        lines.append(f"  耗时: {self.execution_time_ms:.0f}ms")
        return "\n".join(lines)


class OrchestrationResult(BaseModel):
    triggered: bool = True
    sub_tasks: List[SubTask] = Field(default_factory=list)
    final_answer: str = ""
    summary_mode: str = ""
    total_time_ms: float = 0.0


class MultiIntentOrchestrator:
    def __init__(self):
        # 只需要一个"汇总"用的轻量模型 —— 子任务不再调用 LLM
        self.summary_model = ChatQwen(
            model=config.intent_orchestration_model,
            api_key=config.dashscope_api_key,
            temperature=0.2,
        )
        logger.info(
            f"多意图编排器（v2）初始化完成，汇总模型={config.intent_orchestration_model}"
        )

    # ── 判断与子任务构造 ───────────────────────────────────
    def should_orchestrate(self, result: Optional[IntentRecognitionResult]) -> bool:
        if not config.intent_orchestration_enabled:
            return False
        if result is None or not isinstance(result, IntentRecognitionResult):
            return False

        has_multi = bool(result.is_multi_intent)
        has_complex = bool(result.is_complex)
        sub_intent_count = len(result.secondary_intents or []) + 1
        enough_sub = sub_intent_count >= config.intent_orchestration_min_sub_intents

        should = has_multi or has_complex or enough_sub
        logger.info(
            f"编排判断: is_multi_intent={has_multi}, is_complex={has_complex}, "
            f"sub_intent_count={sub_intent_count}(阈值{config.intent_orchestration_min_sub_intents}), "
            f"结果={'进入编排' if should else '走原 Agent'}"
        )
        return should

    def build_sub_tasks(
        self, result: IntentRecognitionResult, original_question: str
    ) -> List[SubTask]:
        tasks: List[SubTask] = []
        all_sub_intents: List[SubIntent] = (
            [result.primary_intent] + list(result.secondary_intents or [])
        )

        for idx, si in enumerate(all_sub_intents, start=1):
            desc = (si.description or "").strip()
            entity_hint = ("、".join(si.entities)) if si.entities else ""
            if desc:
                question = f"{desc}"
                if entity_hint:
                    question += f"（涉及：{entity_hint}）"
            else:
                question = (
                    f"针对【{si.intent_type}】回答以下问题中相关的部分：{original_question}"
                )

            tasks.append(
                SubTask(
                    index=idx,
                    intent_type=str(si.intent_type),
                    entities=list(si.entities or []),
                    question=question,
                    suggested_tools=list(si.suggested_tools or []),
                )
            )

        logger.info(
            f"编排器构造了 {len(tasks)} 个子任务："
            + " ; ".join(
                f"#{t.index}[{t.intent_type}] tools={t.suggested_tools}" for t in tasks
            )
        )
        return tasks

    # ── 工具调度（根据 suggested_tools 选择函数并调用） ─────────
    def _invoke_tool_sync(self, tool_name: str, query: str) -> str:
        """**同步**调用一个工具；被 to_thread 包到线程池里异步执行。"""
        fn = TOOL_REGISTRY.get(tool_name)

        # 未注册或显式 None 的工具（如 MCP 工具）
        if fn is None:
            return f"（工具 {tool_name} 未在编排器中注册，已跳过）"

        try:
            if tool_name == "get_current_time":
                # 无参数
                return fn.func() if hasattr(fn, "func") else fn()
            if tool_name == "query_prometheus_alerts":
                # 无参数
                return fn.func() if hasattr(fn, "func") else fn()
            # 需要 query 的工具：retrieve_knowledge / web_search
            # LangChain @tool 装饰后的函数有 .func 属性指向原始函数，我们直接调原始函数，
            # 避免 LangChain 工具包装的副作用
            if hasattr(fn, "func"):
                return fn.func(query)
            return fn(query)
        except Exception as e:
            logger.error(f"[编排] 子任务工具 {tool_name} 调用失败: {e}")
            return f"（工具 {tool_name} 调用失败: {e}）"

    # ── 空内容判断（用于"知识库没结果就自动触发网络搜索"） ────────
    def _is_empty_output(self, text: str) -> bool:
        if not text:
            return True
        stripped = text.strip()
        return not stripped or stripped in (
            "没有找到相关信息。",
            "知识库和网络检索均未找到相关信息。",
        )

    # ── 单个子任务执行（仅工具，无 LLM） ─────────────────
    async def _run_one_subtask_tool_only(self, task: SubTask, enable_web_search: bool = False):
        """**并行 worker**：
        - 若 suggested_tools 非空 → 每个工具调用一次（并行）
        - 若 suggested_tools 为空 → 标记为"无需工具"
        - 【新逻辑 1】 enable_web_search=True 时，
            无论 suggested_tools 有没有 web_search，都会强制追加 web_search。
        - 【新逻辑 2】 retrieve_knowledge 输出为空/"没有找到相关信息"时，
            自动在本子任务里补一个 web_search（不依赖 knowledge_tool 内部的 auto trigger）。
        """
        start = time.perf_counter()
        logger.info(
            f"[编排 v2] 子任务 {task.index} 开始: {task.question} "
            f"suggested_tools={task.suggested_tools}, enable_web_search={enable_web_search}"
        )

        # ── [Agent 安全 层 3] 工具调用安全检查 ──
        effective_tools: List[str] = list(task.suggested_tools or [])
        if getattr(config, "agent_safety_enabled", True) and getattr(config, "agent_safety_tool_check", True):
            allowed_tools: List[str] = []
            for tool_name in effective_tools:
                try:
                    tool_result = content_safety_service.check_tool_call(tool_name, {"query": task.question})
                    if tool_result.is_safe:
                        allowed_tools.append(tool_name)
                    else:
                        logger.warning(
                            f"[编排 v2-安全] 子任务 {task.index} 工具 {tool_name} "
                            f"被安全系统拦截: {tool_result.reason}"
                        )
                except Exception as e:
                    logger.error(f"[编排 v2-安全] 工具检查异常（放行）: {e}")
                    allowed_tools.append(tool_name)
            effective_tools = allowed_tools

        # 【修复 1】用户手动启用网络搜索时，强制给每个有工具的子任务加 web_search
        if enable_web_search and effective_tools:
            if "web_search" not in effective_tools:
                effective_tools.append("web_search")
                logger.info(
                    f"[编排 v2] 子任务 {task.index} 启用网络搜索后附加 web_search 工具"
                )
        elif enable_web_search and not effective_tools:
            effective_tools = ["web_search"]
            logger.info(
                f"[编排 v2] 子任务 {task.index} 无 suggested_tools，但启用网络搜索；改为 web_search 子任务"
            )

        if not effective_tools:
            logger.info(f"[编排 v2] 子任务 {task.index} 无建议工具，跳过工具调用")
            task.execution_time_ms = (time.perf_counter() - start) * 1000
            return

        # ── 第一步：并行调用 effective_tools 中所有工具 ──
        try:
            results: List[Any] = await asyncio.gather(
                *[
                    asyncio.to_thread(self._invoke_tool_sync, tool_name, task.question)
                    for tool_name in effective_tools
                ],
                return_exceptions=True,
            )

            for tool_name, res in zip(effective_tools, results):
                if isinstance(res, Exception):
                    task.tool_outputs[tool_name] = f"（异常: {res}）"
                else:
                    content = str(res)
                    # ── [Agent 安全 层 2] 文档投毒检查（知识库检索和网络检索结果） ──
                    if (getattr(config, "agent_safety_enabled", True) and
                            tool_name in ("retrieve_knowledge", "web_search") and content):
                        try:
                            doc_result = content_safety_service.check_document(content, source=tool_name)
                            if not doc_result.is_safe:
                                logger.warning(
                                    f"[编排 v2-安全] 子任务 {task.index} 工具 {tool_name} "
                                    f"输出疑似被投毒，已替换为安全摘要"
                                )
                                content = doc_result.sanitized_content
                        except Exception as e:
                            logger.error(f"[编排 v2-安全] 文档检查异常（保留原文）: {e}")
                    task.tool_outputs[tool_name] = content
        except Exception as e:
            task.error = str(e)
            logger.error(f"[编排 v2] 子任务 {task.index} 整体异常: {e}")

        # ── 【修复 2】知识库检索为空时，自动补一次 web_search ──
        # 只在子任务里出现 retrieve_knowledge 但没出现 web_search 时才补；
        # 如果已经有 web_search 调用了（或用户强制启用并附加了），则不再补。
        if "retrieve_knowledge" in task.tool_outputs:
            kb_output = task.tool_outputs.get("retrieve_knowledge", "")
            kb_empty = self._is_empty_output(kb_output)
            already_has_web = "web_search" in task.tool_outputs
            if kb_empty and not already_has_web:
                logger.warning(
                    f"[编排 v2] 子任务 {task.index} 的 retrieve_knowledge 为空，"
                    f"自动追加一次 web_search 作为兜底"
                )
                try:
                    web_res = await asyncio.to_thread(
                        self._invoke_tool_sync, "web_search", task.question
                    )
                    if isinstance(web_res, Exception):
                        task.tool_outputs["web_search"] = f"（异常: {web_res}）"
                    else:
                        web_content = str(web_res)
                        if getattr(config, "agent_safety_enabled", True):
                            try:
                                doc_result = content_safety_service.check_document(web_content, source="web_search")
                                if not doc_result.is_safe:
                                    web_content = doc_result.sanitized_content
                            except Exception as e2:
                                logger.error(f"[编排 v2-安全] 兜底 web_search 文档检查异常: {e2}")
                        task.tool_outputs["web_search"] = web_content
                except Exception as e:
                    logger.error(f"[编排 v2] 兜底 web_search 失败: {e}")
                    task.tool_outputs["web_search"] = f"（兜底 web_search 失败: {e}）"

        task.execution_time_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"[编排 v2] 子任务 {task.index} 完成: "
            f"工具数={len(task.tool_outputs)}, 耗时={task.execution_time_ms:.0f}ms, "
            f"实际调用={list(task.tool_outputs.keys())}"
        )

    # ── 汇总：LLM 流式综合（基于工具结果） ──────────────────
    async def synthesize_stream(
        self,
        tasks: List[SubTask],
        original_question: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """汇总：
        1. 若至少一个子任务有工具结果 → 调一次 LLM 流式综合
        2. 所有子任务都没有工具结果（全失败/全为"无工具意图"）→ fallback 到原 Agent
        """
        has_any = any(t.has_any_tool_result for t in tasks)

        if not has_any:
            logger.warning(
                "[编排 v2] 所有子任务均未产出有效工具结果，fallback=true"
            )
            yield {
                "type": "_summary_internal",
                "summary_mode": "fallback",
                "final_answer": "",
            }
            return

        tasks_block = "\n\n".join(t.format_for_prompt() for t in tasks)

        system_prompt = dedent("""
            你是一个"结果汇总员"。给定原始用户问题，以及 N 个子任务独立执行工具检索得到的结果，
            请把这些结果整合成一段流畅、逻辑清晰的最终回答。

            整合原则：
              1. 用中文输出，直接面向最终用户
              2. 按逻辑顺序组织（先介绍背景，再给出对比 / 结论）
              3. 必要时使用小标题、列表分段
              4. **只使用工具给出的信息**，不编造、不扩展、不输出思考过程
              5. 如果某个子任务工具结果为空/失败，不要在回答中显式提它
              6. 不要输出"以下是根据各个子任务得到的回答"等元信息，直接给出回答
        """).strip()

        user_prompt = dedent(f"""
            原始用户问题：
            {original_question}

            子任务工具结果（按意图顺序，可能存在重复信息）：
            {tasks_block}

            请基于上述信息，输出一个自然、完整的最终回答。
        """).strip()

        collected: List[str] = []
        try:
            logger.info(
                f"[编排 v2] 调用汇总 LLM 流式；有工具结果的子任务="
                f"{sum(1 for t in tasks if t.has_any_tool_result)}/{len(tasks)}"
            )
            # langchain_qwq 的 astream 返回 AIMessageChunk 序列
            async for chunk in self.summary_model.astream(
                [SystemMessage(content=system_prompt),
                 HumanMessage(content=user_prompt)]
            ):
                text = ""
                if hasattr(chunk, "content"):
                    c = getattr(chunk, "content")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text += item.get("text", "")
                            elif isinstance(item, str):
                                text += item
                if text:
                    collected.append(text)
                    yield {
                        "type": "content",
                        "data": text,
                        "node": "orchestration_synthesis",
                    }
        except Exception as e:
            logger.error(f"[编排 v2] 汇总 LLM 失败: {e}，退化为拼接")
            # 降级：直接把各子任务的工具输出原样拼接
            parts: List[str] = []
            for t in tasks:
                if not t.has_any_tool_result:
                    continue
                header = f"【{t.index}. {t.question}】"
                body_chunks = []
                for tool_name, text in t.tool_outputs.items():
                    body_chunks.append(f"[{tool_name}] {text}")
                parts.append(header + "\n" + "\n".join(body_chunks))

            if not parts:
                yield {
                    "type": "_summary_internal",
                    "summary_mode": "fallback",
                    "final_answer": "",
                }
                return

            final = "\n\n".join(parts)
            # 降级结果也做一下流式吐出，让 UI 有一致体验
            for i in range(0, len(final), 150):
                yield {
                    "type": "content",
                    "data": final[i:i + 150],
                    "node": "orchestration_synthesis",
                }
                await asyncio.sleep(0.01)
            yield {
                "type": "_summary_internal",
                "summary_mode": "concat_all",
                "final_answer": final,
            }
            return

        final = "".join(collected)
        if final.strip():
            yield {
                "type": "_summary_internal",
                "summary_mode": "llm_synthesis",
                "final_answer": final,
            }
        else:
            # LLM 返回空 → 兜底
            yield {
                "type": "_summary_internal",
                "summary_mode": "fallback",
                "final_answer": "",
            }

    # ── 主入口：并行工具执行 + 一次流式汇总 LLM ───────────────
    async def orchestrate(
        self,
        result: IntentRecognitionResult,
        original_question: str,
        session_id: str,
        enable_web_search: bool = False,
        streaming: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        t0 = time.perf_counter()
        tasks = self.build_sub_tasks(result, original_question)

        # 1) 推送"编排开始"事件（前端只需要总数做一个简单进度提示）
        yield {
            "type": "orchestration_start",
            "data": {
                "total": len(tasks),
                "subtasks": [
                    {"index": t.index,
                     "question": t.question,
                     "intent_type": t.intent_type,
                     "suggested_tools": t.suggested_tools}
                    for t in tasks
                ],
            },
        }

        # 2) 并行执行 N 个子任务（全部只是工具调用，没有 LLM）
        #    enable_web_search 会透传到子任务里：用户勾选"启用网络搜索"时，每个子任务强制附加 web_search
        worker_tasks = [
            asyncio.create_task(
                self._run_one_subtask_tool_only(t, enable_web_search=enable_web_search)
            )
            for t in tasks
        ]
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        # 3) 汇总阶段：一次 LLM 流式生成最终回答
        summary_mode = "fallback"
        async for summary_evt in self.synthesize_stream(tasks, original_question):
            if summary_evt["type"] == "_summary_internal":
                summary_mode = summary_evt["summary_mode"]
            else:
                # content：汇总 LLM 的流式 token → 原样透传
                yield summary_evt

        total_ms = (time.perf_counter() - t0) * 1000
        success_count = sum(1 for t in tasks if t.has_any_tool_result)
        logger.info(
            f"[编排 v2] 完成: 有工具结果={success_count}/{len(tasks)}, "
            f"模式={summary_mode}, 总耗时={total_ms:.0f}ms"
        )

        yield {
            "type": "orchestration_summary",
            "data": {
                "summary_mode": summary_mode,
                "total_count": len(tasks),
                "succeeded_count": success_count,
                "elapsed_ms": total_ms,
                "fallback": summary_mode == "fallback",
            },
        }


orchestrator = MultiIntentOrchestrator()