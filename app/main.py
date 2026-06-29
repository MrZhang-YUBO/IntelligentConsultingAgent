"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from contextlib import asynccontextmanager
import os

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops
from app.core.milvus_client import milvus_manager
from app.core.metrics import MetricsMiddleware, get_metrics_response_bytes
from app.events.document_event_handler import (
    start_document_event_consumer,
    stop_document_event_consumer,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")

    # 启动 Kafka Consumer（文档变更事件处理
    logger.info("📨 正在启动 Kafka Consumer...")
    start_document_event_consumer()
    logger.info("✅ Kafka Consumer 启动完成")

    logger.info("=" * 60)

    yield

    # 关闭时执行
    logger.info("🔌 正在停止 Kafka Consumer...")
    stop_document_event_consumer()
    logger.info("✅ Kafka Consumer 已停止")

    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置 Prometheus 指标中间件（必须在 CORS 之后注册，先执行外层）
if config.metrics_enabled:
    app.add_middleware(MetricsMiddleware)
    logger.info(f"[Metrics] 已启用指标中间件，暴露路径: {config.metrics_path}")

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 暴露 Prometheus /metrics 端点
if config.metrics_enabled:
    @app.get(config.metrics_path, include_in_schema=True, tags=["监控"])
    async def metrics():
        """Prometheus 指标抓取端点。"""
        content, content_type = get_metrics_response_bytes()
        return Response(content=content, media_type=content_type)

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )