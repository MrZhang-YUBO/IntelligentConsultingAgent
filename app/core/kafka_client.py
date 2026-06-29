"""Kafka 客户端 - 文档变更事件的生产者与消费者

使用 confluent-kafka 或 kafka-python（按实际依赖选择）：
- Producer: 发送 DocumentChangeEvent 消息到 topic
- Consumer: 从 topic 消费消息，处理文档索引更新

消息格式（JSON）:
{
    "event_id": "evt_xxx",
    "event_type": "DOCUMENT_CREATED" | "DOCUMENT_UPDATED" | "DOCUMENT_DELETED",
    "document_id": "doc_xxx",
    "file_path": "/uploads/manual.pdf",
    "content_hash": "sha256:xxx",
    "old_hash": "sha256:yyy" | null,
    "timestamp": 1718524900
}
"""

import json
import threading
import time
from typing import Callable, Optional

from loguru import logger

from app.config import config
from app.models.document import DocumentChangeEvent


# ---------------------------------------------------------------------------
# 辅助函数：确保 topic 存在
# ---------------------------------------------------------------------------
_topic_ensure_lock = threading.Lock()
_topics_ensured: set[str] = set()


def _ensure_topic_exists(bootstrap_servers: str, topic: str) -> None:
    """如果 topic 不存在则创建它（线程安全，去重）"""
    with _topic_ensure_lock:
        if topic in _topics_ensured:
            return

        try:
            from confluent_kafka import admin as confluent_admin

            admin_client = confluent_admin.AdminClient(
                {"bootstrap.servers": bootstrap_servers}
            )

            # 先检查 topic 是否已存在
            cluster_metadata = admin_client.list_topics(timeout=5.0)
            if topic in cluster_metadata.topics:
                _topics_ensured.add(topic)
                admin_client.close()
                return

            # 创建 topic（单分区，单副本，适合开发；生产请自行调整）
            new_topic = confluent_admin.NewTopic(
                topic=topic,
                num_partitions=1,
                replication_factor=1,
                config={
                    "cleanup.policy": "delete",
                    "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 天
                },
            )
            future = admin_client.create_topics([new_topic])
            for _, f in future.items():
                f.result(timeout=10.0)

            _topics_ensured.add(topic)
            admin_client.close()
            logger.info(f"Kafka topic 已创建: {topic}")

        except Exception as e:
            logger.debug(f"确保 Kafka topic 存在失败（可能由 broker 自动创建或已存在）: {e}")
        finally:
            _topics_ensured.add(topic)  # 避免重复尝试影响启动速度


# ---------------------------------------------------------------------------
# 生产者
# ---------------------------------------------------------------------------
class DocumentEventProducer:
    """文档变更事件生产者

    发送事件到 Kafka topic。
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
    ):
        self.bootstrap_servers = bootstrap_servers or config.kafka_bootstrap_servers
        self.topic = topic or config.kafka_topic_document_changes
        self._producer = None
        self._lock = threading.Lock()

        try:
            from confluent_kafka import Producer as ConfluentProducer

            self._producer = ConfluentProducer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "acks": "all",
                    "retries": config.kafka_max_retries,
                    "retry.backoff.ms": config.kafka_retry_backoff_ms,
                    "compression.type": "snappy",
                    "queue.buffering.max.messages": 100000,
                }
            )
            self._backend = "confluent-kafka"
            logger.info(
                f"Kafka Producer 初始化成功 (confluent-kafka): "
                f"{self.bootstrap_servers}, topic={self.topic}"
            )
            _ensure_topic_exists(self.bootstrap_servers, self.topic)

        except ImportError:
            # 兜底：使用 kafka-python
            try:
                from kafka import KafkaProducer as KafkaPyProducer

                self._producer = KafkaPyProducer(
                    bootstrap_servers=[self.bootstrap_servers],
                    value_serializer=lambda v: v,
                    retries=config.kafka_max_retries,
                    retry_backoff_ms=config.kafka_retry_backoff_ms,
                    compression_type="snappy",
                )
                self._backend = "kafka-python"
                logger.info(
                    f"Kafka Producer 初始化成功 (kafka-python): "
                    f"{self.bootstrap_servers}, topic={self.topic}"
                )

            except ImportError:
                self._backend = "memory"
                logger.warning(
                    "未安装 kafka 客户端库 (confluent-kafka / kafka-python), "
                    "将使用内存模拟 (仅适合开发调试)"
                )
                self._pending_events: list[DocumentChangeEvent] = []

    # ------------------------------------------------------------------
    # 发送接口
    # ------------------------------------------------------------------
    def send(self, event: DocumentChangeEvent) -> bool:
        """发送一个事件到 Kafka

        Args:
            event: 文档变更事件

        Returns:
            True 表示发送成功，False 表示失败
        """
        try:
            payload = event.to_json_bytes()

            if self._backend == "confluent-kafka":
                assert self._producer is not None
                # 异步发送，使用回调报告结果
                def _delivery_report(err, msg):
                    if err is not None:
                        logger.error(
                            f"Kafka 消息发送失败: {err}, "
                            f"event_id={event.event_id}"
                        )
                    else:
                        logger.debug(
                            f"Kafka 消息发送成功: event_id={event.event_id}, "
                            f"topic={msg.topic()}, partition={msg.partition()}, "
                            f"offset={msg.offset()}"
                        )

                self._producer.produce(
                    self.topic, value=payload, on_delivery=_delivery_report
                )
                self._producer.poll(0)  # 触发回调

            elif self._backend == "kafka-python":
                assert self._producer is not None
                future = self._producer.send(self.topic, value=payload)
                future.get(timeout=10)  # 同步等待发送完成

            else:  # memory
                self._pending_events.append(event)
                logger.debug(
                    f"[Memory] 事件已入队: event_id={event.event_id}, "
                    f"type={event.event_type}, 待处理={len(self._pending_events)}"
                )

            logger.info(
                f"事件已发送: type={event.event_type}, "
                f"document_id={event.document_id}, "
                f"event_id={event.event_id}"
            )
            return True

        except Exception as e:
            logger.error(f"发送 Kafka 事件失败: event_id={event.event_id}, err={e}")
            return False

    def flush(self, timeout: float = 5.0) -> None:
        """确保所有待发送消息都已发出"""
        try:
            if self._backend == "confluent-kafka" and self._producer is not None:
                self._producer.flush(int(timeout * 1000))
            elif self._backend == "kafka-python" and self._producer is not None:
                self._producer.flush(timeout=timeout)
        except Exception as e:
            logger.warning(f"Kafka flush 异常: {e}")

    def close(self) -> None:
        """关闭生产者"""
        try:
            self.flush()
            if hasattr(self._producer, "close"):
                self._producer.close()
            logger.info("Kafka Producer 已关闭")
        except Exception as e:
            logger.warning(f"关闭 Kafka Producer 时异常: {e}")

    # 内存模式：暴露待处理事件供测试/调试使用
    @property
    def pending_events(self) -> list:
        return getattr(self, "_pending_events", [])


# ---------------------------------------------------------------------------
# 消费者
# ---------------------------------------------------------------------------
class DocumentEventConsumer:
    """文档变更事件消费者

    从 Kafka topic 消费消息，调用 handler 处理。
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        group_id: Optional[str] = None,
        handler: Optional[Callable[[DocumentChangeEvent], None]] = None,
    ):
        self.bootstrap_servers = bootstrap_servers or config.kafka_bootstrap_servers
        self.topic = topic or config.kafka_topic_document_changes
        self.group_id = group_id or config.kafka_consumer_group_id
        self._handler = handler
        self._consumer = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        try:
            from confluent_kafka import Consumer as ConfluentConsumer

            self._consumer = ConfluentConsumer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "group.id": self.group_id,
                    "auto.offset.reset": config.kafka_auto_offset_reset,
                    "enable.auto.commit": True,
                    "session.timeout.ms": config.kafka_session_timeout_ms,
                    "heartbeat.interval.ms": config.kafka_heartbeat_interval_ms,
                }
            )
            self._backend = "confluent-kafka"
            logger.info(
                f"Kafka Consumer 初始化成功 (confluent-kafka): "
                f"{self.bootstrap_servers}, topic={self.topic}, group={self.group_id}"
            )
            _ensure_topic_exists(self.bootstrap_servers, self.topic)

        except ImportError:
            try:
                from kafka import KafkaConsumer as KafkaPyConsumer

                self._consumer = KafkaPyConsumer(
                    self.topic,
                    bootstrap_servers=[self.bootstrap_servers],
                    group_id=self.group_id,
                    auto_offset_reset=config.kafka_auto_offset_reset,
                    enable_auto_commit=True,
                    session_timeout_ms=config.kafka_session_timeout_ms,
                    heartbeat_interval_ms=config.kafka_heartbeat_interval_ms,
                    value_deserializer=lambda m: m,
                )
                self._backend = "kafka-python"
                logger.info(
                    f"Kafka Consumer 初始化成功 (kafka-python): "
                    f"{self.bootstrap_servers}, topic={self.topic}, group={self.group_id}"
                )

            except ImportError:
                self._backend = "memory"
                logger.warning(
                    "未安装 kafka 客户端库，使用内存模拟 consumer"
                )

    # ------------------------------------------------------------------
    # 消费循环
    # ------------------------------------------------------------------
    def set_handler(self, handler: Callable[[DocumentChangeEvent], None]) -> None:
        """设置事件处理器"""
        self._handler = handler

    def start(self) -> None:
        """启动消费线程（后台）"""
        if self._thread and self._thread.is_alive():
            logger.warning("Consumer 已在运行中")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="kafka-consumer")
        self._thread.start()
        logger.info("Kafka Consumer 线程已启动")

    def stop(self) -> None:
        """停止消费线程"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        if self._consumer is not None and hasattr(self._consumer, "close"):
            try:
                self._consumer.close()
            except Exception as e:
                logger.warning(f"关闭 Kafka Consumer 异常: {e}")
        logger.info("Kafka Consumer 已停止")

    def _run_loop(self) -> None:
        """消费主循环"""
        logger.info("Kafka Consumer 循环开始")

        try:
            # confluent-kafka 需要显式订阅
            if self._backend == "confluent-kafka" and self._consumer is not None:
                self._consumer.subscribe([self.topic])

            while not self._stop_event.is_set():
                try:
                    event = self._consume_one()
                    if event is not None:
                        self._dispatch(event)
                except Exception as e:
                    logger.error(f"消费事件异常: {e}")
                    time.sleep(1.0)  # 异常时退避

        finally:
            logger.info("Kafka Consumer 循环结束")

    def _consume_one(self) -> Optional[DocumentChangeEvent]:
        """尝试消费一条消息"""
        if self._backend == "confluent-kafka" and self._consumer is not None:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                return None
            if msg.error():
                err = msg.error()
                err_code = getattr(err, "code", None)
                # 3 = UNKNOWN_TOPIC_OR_PART，通常是 topic 尚未创建的短暂状态
                if callable(err_code):
                    code_val = err_code()
                else:
                    code_val = err_code
                if code_val == 3:
                    logger.warning(
                        f"Kafka topic '{self.topic}' 尚未就绪，稍后重试: {err}"
                    )
                    time.sleep(2.0)
                    return None
                logger.error(f"Kafka 消费错误: {err}")
                return None

            try:
                payload = msg.value()
                if payload is None:
                    return None
                event = DocumentChangeEvent.from_json_bytes(
                    payload if isinstance(payload, bytes) else payload.encode("utf-8")
                )
                logger.debug(f"消费到事件: {event.event_id}")
                return event
            except Exception as e:
                logger.warning(f"解析 Kafka 消息失败: {e}")
                return None

        elif self._backend == "kafka-python" and self._consumer is not None:
            try:
                records = self._consumer.poll(timeout_ms=1000)
                for _, record_list in records.items():
                    for record in record_list:
                        try:
                            value = record.value
                            if value is None:
                                continue
                            event = DocumentChangeEvent.from_json_bytes(
                                value if isinstance(value, bytes) else value.encode("utf-8")
                            )
                            return event
                        except Exception as e:
                            logger.warning(f"解析 Kafka 消息失败: {e}")
            except Exception as e:
                logger.debug(f"kafka-python poll 异常: {e}")
            return None

        else:  # memory 模式
            # 从 producer 的 pending_events 轮询（调试用途）
            producer_events = getattr(_global_producer, "_pending_events", [])
            if producer_events:
                event = producer_events.pop(0)
                logger.debug(f"[Memory] 消费事件: {event.event_id}")
                return event
            time.sleep(1.0)
            return None

    def _dispatch(self, event: DocumentChangeEvent) -> None:
        """将事件分发给 handler"""
        if self._handler is None:
            logger.warning("Consumer handler 未设置，忽略事件")
            return
        try:
            self._handler(event)
        except Exception as e:
            logger.error(f"处理事件失败: event_id={event.event_id}, err={e}")


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
_global_producer = DocumentEventProducer()


def get_producer() -> DocumentEventProducer:
    """获取全局生产者"""
    return _global_producer


def send_event(event: DocumentChangeEvent) -> bool:
    """便捷函数：发送事件"""
    return _global_producer.send(event)