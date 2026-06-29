"""文档变更事件处理器

Kafka Consumer 端逻辑入口：
1. 接收 DOCUMENT_CREATED / UPDATED / DELETED 事件
2. 调用 vector_index_service.process_event() 处理
3. 处理异常并记录日志
"""

import threading

from loguru import logger

from app.core.kafka_client import DocumentEventConsumer
from app.models.document import DocumentChangeEvent
from app.services.vector_index_service import vector_index_service


_consumer_instance: DocumentEventConsumer | None = None
_started: bool = False
_lock = threading.Lock()


def _event_handler(event: DocumentChangeEvent) -> None:
    """Kafka Consumer 回调：将事件分发给 vector_index_service"""
    try:
        vector_index_service.process_event(event)
    except Exception as e:
        logger.error(f"事件处理器异常: event_id={event.event_id}, err={e}")


def start_document_event_consumer() -> None:
    """启动文档变更事件 Consumer（全局单例）

    在应用启动时调用（FastAPI lifespan 中）
    """
    global _consumer_instance, _started

    with _lock:
        if _started:
            logger.warning("Document event consumer 已启动")
            return

        try:
            consumer = DocumentEventConsumer(handler=_event_handler)
            consumer.start()
            _consumer_instance = consumer
            _started = True
            logger.info("Document event consumer 启动成功")
        except Exception as e:
            logger.error(f"启动 Document event consumer 失败: {e}")
            _started = False


def stop_document_event_consumer() -> None:
    """停止文档变更事件 Consumer

    在应用关闭时调用（FastAPI lifespan 中）
    """
    global _consumer_instance, _started

    with _lock:
        if _consumer_instance is not None:
            try:
                _consumer_instance.stop()
                logger.info("Document event consumer 已停止")
            except Exception as e:
                logger.warning(f"停止 Consumer 异常: {e}")
            finally:
                _consumer_instance = None
                _started = False


def is_consumer_running() -> bool:
    """返回 Consumer 是否在运行"""
    return _started and _consumer_instance is not None