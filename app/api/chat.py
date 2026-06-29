"""对话接口

提供基于 RAG Agent 的普通对话和流式对话接口
"""

import json
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from app.models.request import ChatRequest, ClearRequest
from app.models.response import SessionInfoResponse, ApiResponse
from app.agent.mcp_client import format_exception_chain
from app.services.rag_agent_service import rag_agent_service
from loguru import logger

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """快速对话接口
    {
        "code": 200,
        "message": "success",
        "data": {
            "success": true,
            "answer": "回答内容",
            "errorMessage": null
        }
    }

    Args:
        request: 对话请求

    Returns:
        统一格式的对话响应
    """
    try:
        logger.info(f"[会话 {request.id}] 收到快速对话请求: {request.question}")
        answer = await rag_agent_service.query(
            request.question,
            session_id=request.id,
            enable_web_search=request.enable_web_search,
        )

        # 取本轮意图识别结果（query 内部已记录到轨迹，取最后一条即本轮）
        intents = rag_agent_service.get_session_intents(request.id)
        latest_intent = intents[-1] if intents else None

        logger.info(f"[会话 {request.id}] 快速对话完成")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,
                "intent": latest_intent,
                "errorMessage": None
            }
        }

    except Exception as e:
        logger.error(f"对话接口错误: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e)
            }
        }


@router.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口（基于 RAG Agent，SSE）

    返回 SSE 格式，data 字段为 JSON：

    工具调用事件:
    event: message
    data: {"type":"tool_call","data":{"tool":"工具名","status":"start|end","input":{...}}}

    内容流式事件:
    event: message
    data: {"type":"content","data":"内容块"}

    完成事件:
    event: message
    data: {"type":"done","data":{"answer":"完整答案","tool_calls":[...]}}

    Args:
        request: 对话请求

    Returns:
        SSE 事件流
    """
    logger.info(f"[会话 {request.id}] 收到流式对话请求: {request.question}")

    async def event_generator():
        try:
            async for chunk in rag_agent_service.query_stream(
                request.question, session_id=request.id,
                enable_web_search=request.enable_web_search,
            ):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                logger.info(
                    f"[流式调试] yield type={chunk.get('type')}, data_preview={str(chunk.get('data', ''))[:200]}")
                # 处理调试类型消息（新增）
                if chunk_type == "debug":
                    # 调试信息，可以选择发送或忽略
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "debug",
                            "node": chunk.get("node", "unknown"),
                            "message_type": chunk.get("message_type", "unknown")
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "intent":
                    # 意图识别结果（在内容流之前到达，前端可展示"系统理解到的意图"）
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "intent",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "orchestration_start":
                    # 编排开始：宣告子任务总数与清单
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "orchestration_start",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "orchestration_step":
                    # 编排进度：子任务开始/结束
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "orchestration_step",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "orchestration_summary":
                    # 编排汇总结果
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "orchestration_summary",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "tool_call":
                    # 发送工具调用事件（可选，前端可以显示工具调用状态）
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "tool_call",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "search_results":
                    # 发送检索结果（可选，前端可以忽略）
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "search_results",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "web_search":
                    # 网络检索触发通知
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "web_search",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "content":
                    # 发送内容块；如带 subtask_index，代表"某子任务的流式内容"
                    out = {
                        "type": "content",
                        "data": chunk_data
                    }
                    subtask_idx = chunk.get("subtask_index")
                    if subtask_idx is not None:
                        out["subtask_index"] = subtask_idx
                    node = chunk.get("node")
                    if node is not None:
                        out["node"] = node
                    yield {
                        "event": "message",
                        "data": json.dumps(out, ensure_ascii=False)
                    }
                elif chunk_type == "complete":
                    # 发送完成信号
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "done",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "error":
                    # 发送错误信息
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "error",
                            "data": str(chunk_data)
                        }, ensure_ascii=False)
                    }

            logger.info(f"[会话 {request.id}] 流式对话完成")

        except Exception as e:
            logger.error(f"流式对话接口错误: {format_exception_chain(e)}")
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "data": str(e)
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    """清空会话历史

    Args:
        request: 清空请求

    Returns:
        操作结果
    """
    try:
        success = rag_agent_service.clear_session(request.session_id)
        logger.info(f"清空会话: {request.session_id}, 结果: {success}")

        return ApiResponse(
            status="success" if success else "error",
            message="会话已清空" if success else "清空会话失败",
            data=None
        )

    except Exception as e:
        logger.error(f"清空会话错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """查询会话历史

    Args:
        session_id: 会话 ID

    Returns:
        会话信息
    """
    try:
        history = rag_agent_service.get_session_history(session_id)
        intents = rag_agent_service.get_session_intents(session_id)

        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history,
            intents=intents
        )

    except Exception as e:
        logger.error(f"获取会话信息错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))