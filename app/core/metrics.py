"""业务指标模块 - 基于 Prometheus

负责：
1. 定义全局 Prometheus 指标（Counter / Histogram / Gauge）
2. 提供 FastAPI 中间件 MetricsMiddleware，统计每个 HTTP 请求的耗时
3. 提供装饰器 @timed_metric，用于统计任意业务函数的耗时

使用方式：
    from app.core.metrics import MetricsMiddleware, timed_metric

    # FastAPI 挂载中间件
    app.add_middleware(MetricsMiddleware)

    # 装饰任意函数（同步 or 异步）
    @timed_metric(service="rag", method="query")
    async def query(...):
        ...
"""

import asyncio
import functools
import time
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.routing import Match

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - 缺失依赖时的兜底
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None
    Counter = None
    Gauge = None
    Histogram = None


from loguru import logger


# ---------------------------------------------------------------------------
# 全局 CollectorRegistry（避免和其他进程注册表冲突）
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    REGISTRY = CollectorRegistry()

    # 1) HTTP 请求总数
    HTTP_REQUEST_COUNT = Counter(
        "http_request_count",
        "Total HTTP requests handled by the service",
        ["method", "path", "status_code"],
        registry=REGISTRY,
    )

    # 2) HTTP 请求耗时直方图（默认分桶：0.005s ~ 10s）
    HTTP_REQUEST_LATENCY = Histogram(
        "http_request_latency_seconds",
        "HTTP request latency in seconds",
        ["method", "path"],
        registry=REGISTRY,
    )

    # 3) 正在处理中的请求数（Gauge）
    HTTP_REQUEST_IN_PROGRESS = Gauge(
        "http_request_in_progress",
        "Number of HTTP requests currently being processed",
        ["method", "path"],
        registry=REGISTRY,
    )

    # 4) HTTP 错误总数（4xx / 5xx）
    HTTP_ERROR_COUNT = Counter(
        "http_error_count",
        "Total HTTP error responses (4xx / 5xx)",
        ["method", "path", "status_code"],
        registry=REGISTRY,
    )

    # 5) 业务层耗时直方图（供服务内部使用）
    SERVICE_LATENCY = Histogram(
        "service_request_latency_seconds",
        "Service-level request latency in seconds",
        ["service", "method"],
        registry=REGISTRY,
    )

    # 6) 业务层错误计数
    SERVICE_ERROR_COUNT = Counter(
        "service_error_count",
        "Total service-level errors",
        ["service", "method"],
        registry=REGISTRY,
    )
else:  # pragma: no cover
    REGISTRY = None
    HTTP_REQUEST_COUNT = None
    HTTP_REQUEST_LATENCY = None
    HTTP_REQUEST_IN_PROGRESS = None
    HTTP_ERROR_COUNT = None
    SERVICE_LATENCY = None
    SERVICE_ERROR_COUNT = None


def is_prometheus_available() -> bool:
    """返回 prometheus_client 是否已安装。"""
    return PROMETHEUS_AVAILABLE


# ---------------------------------------------------------------------------
# FastAPI 中间件
# ---------------------------------------------------------------------------
class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus 指标中间件

    对每个 HTTP 请求：
    - 自增 in_progress gauge（请求开始）
    - 记录耗时（请求结束）
    - 根据状态码更新 error_count / request_count
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not PROMETHEUS_AVAILABLE:
            return await call_next(request)

        method = request.method
        path = _resolve_path(request)  # 归一化，避免路径参数导致 cardinality 爆炸
        start_time = time.perf_counter()

        # 更新 in_progress
        HTTP_REQUEST_IN_PROGRESS.labels(method=method, path=path).inc()

        status_code = 500  # 默认，处理中异常时也能记录
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            HTTP_REQUEST_IN_PROGRESS.labels(method=method, path=path).dec()
            HTTP_REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
            HTTP_REQUEST_COUNT.labels(
                method=method, path=path, status_code=str(status_code)
            ).inc()

            if status_code >= 400:
                HTTP_ERROR_COUNT.labels(
                    method=method, path=path, status_code=str(status_code)
                ).inc()


# ---------------------------------------------------------------------------
# 业务函数耗时装饰器（兼容同步 & 异步）
# ---------------------------------------------------------------------------
def timed_metric(service: str, method: str) -> Callable:
    """装饰一个函数/方法，自动统计它的耗时与错误数。

    兼容同步函数和 `async def` 协程。

    Args:
        service: 业务模块名，例如 "rag" / "vector_index" / "aiops"
        method:  具体方法名，例如 "query" / "index_file"

    Example:
        @timed_metric(service="rag", method="query")
        async def query(...):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return await func(*args, **kwargs)
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            except Exception:
                SERVICE_ERROR_COUNT.labels(service=service, method=method).inc()
                raise
            finally:
                SERVICE_LATENCY.labels(service=service, method=method).observe(
                    time.perf_counter() - start
                )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            except Exception:
                SERVICE_ERROR_COUNT.labels(service=service, method=method).inc()
                raise
            finally:
                SERVICE_LATENCY.labels(service=service, method=method).observe(
                    time.perf_counter() - start
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# 辅助：从 FastAPI route 中解析归一化路径（避免 "/chat/{session_id}" 产生不同 label）
# ---------------------------------------------------------------------------
def _resolve_path(request: Request) -> str:
    """把真实 path 匹配到已注册的路由 path template。"""
    app = getattr(request, "app", None)
    if app is None:
        return request.url.path

    for route in getattr(app, "routes", []):
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", request.url.path)

    return request.url.path


# ---------------------------------------------------------------------------
# /metrics 响应内容生成（供 FastAPI 路由直接调用）
# ---------------------------------------------------------------------------
def get_metrics_response_bytes() -> tuple[bytes, str]:
    """生成 prometheus 最新指标文本。

    Returns:
        (bytes_content, content_type)
    """
    if not PROMETHEUS_AVAILABLE:
        msg = (
            b"# prometheus_client is not installed. "
            b"Please install `prometheus-client` to enable metrics."
        )
        return msg, "text/plain; version=0.0.4; charset=utf-8"

    data = generate_latest(REGISTRY)
    return data, CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# 启动日志：打印模块是否就绪
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    logger.info("[Metrics] prometheus_client 已加载，指标注册完成")
else:  # pragma: no cover
    logger.warning(
        "[Metrics] prometheus_client 未安装，指标功能将跳过。"
        "请执行: pip install prometheus-client"
    )