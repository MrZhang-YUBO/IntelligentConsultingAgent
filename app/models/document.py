"""文档相关数据模型

包含：
- DocumentChunk:       分片模型（保留原有）
- DocumentRecord:      文档注册表记录（document_id ↔ content_hash）
- DocumentChangeEvent: Kafka 事件模型（新增/更新/删除事件）
"""

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 分片模型（保留原有接口，兼容旧代码）
# ---------------------------------------------------------------------------
class DocumentChunk(BaseModel):
    """文档分片模型"""

    content: str = Field(..., description="分片内容")
    start_index: int = Field(..., description="分片在原文档中的起始位置")
    end_index: int = Field(..., description="分片在原文档中的结束位置")
    chunk_index: int = Field(..., description="分片索引（从0开始）")
    title: Optional[str] = Field(None, description="分片所属章节标题")

    class Config:
        """Pydantic 配置"""
        json_schema_extra = {
            "example": {
                "content": "这是一段文档内容...",
                "start_index": 0,
                "end_index": 100,
                "chunk_index": 0,
                "title": "第一章",
            }
        }


# ---------------------------------------------------------------------------
# 事件类型枚举
# ---------------------------------------------------------------------------
class EventType(str, Enum):
    """文档变更事件类型"""

    DOCUMENT_CREATED = "DOCUMENT_CREATED"  # 新文档首次索引
    DOCUMENT_UPDATED = "DOCUMENT_UPDATED"  # 已有文档内容变更
    DOCUMENT_DELETED = "DOCUMENT_DELETED"  # 文档被删除
    DOCUMENT_UNCHANGED = "DOCUMENT_UNCHANGED"  # 内容未变更（仅用于 API 响应）


class DocumentStatus(str, Enum):
    """文档在注册表中的状态"""

    ACTIVE = "active"          # 正常索引中
    PENDING = "pending"        # 已发事件，等待 Consumer 处理
    PROCESSING = "processing"  # Consumer 正在处理
    ERROR = "error"            # 处理失败
    DELETED = "deleted"        # 已标记删除


# ---------------------------------------------------------------------------
# 文档注册表记录
# ---------------------------------------------------------------------------
class DocumentRecord(BaseModel):
    """文档注册表记录

    存储 document_id ↔ file_path ↔ content_hash 的映射关系，
    用于：
    1. 检测文件内容是否变更（hash 比对）
    2. 通过 document_id 批量删除所有关联 chunk
    3. 追踪索引历史（更新时间、分片数量）
    """

    document_id: str = Field(..., description="文档唯一ID，格式：doc_<8位随机>")
    file_path: str = Field(..., description="文档文件的绝对/归一化路径")
    file_name: str = Field(..., description="文件名（含扩展名）")
    content_hash: str = Field(..., description="文档内容 SHA-256 hash，格式 sha256:<hex>")
    chunk_count: int = Field(default=0, description="该文档被切分后的 chunk 总数")
    status: DocumentStatus = Field(default=DocumentStatus.ACTIVE, description="文档状态")
    created_at: int = Field(default_factory=lambda: int(time.time()), description="首次索引时间戳（秒）")
    updated_at: int = Field(default_factory=lambda: int(time.time()), description="最近更新时间戳（秒）")
    last_event_id: Optional[str] = Field(None, description="最近关联的 Kafka 事件 ID")
    error_message: Optional[str] = Field(None, description="处理失败时的错误信息")
    extra: Optional[dict[str, Any]] = Field(None, description="预留扩展字段")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_a1b2c3d4",
                "file_path": "/uploads/ops-manual.pdf",
                "file_name": "ops-manual.pdf",
                "content_hash": "sha256:a1b2c3d4e5f6...",
                "chunk_count": 42,
                "status": "active",
                "created_at": 1718524800,
                "updated_at": 1718524900,
            }
        }

    def touch(self) -> None:
        """更新 updated_at 为当前时间"""
        self.updated_at = int(time.time())


# ---------------------------------------------------------------------------
# Kafka 事件模型
# ---------------------------------------------------------------------------
class DocumentChangeEvent(BaseModel):
    """Kafka 文档变更事件

    Producer（API 层）发送，Consumer（事件处理器）消费。

    典型消息：
    {
        "event_id": "evt_xyz789",
        "event_type": "DOCUMENT_UPDATED",
        "document_id": "doc_a1b2c3d4",
        "file_path": "/uploads/manual.pdf",
        "content_hash": "sha256:new...",
        "old_hash": "sha256:old...",
        "timestamp": 1718524900
    }
    """

    event_id: str = Field(
        default_factory=lambda: "evt_" + uuid.uuid4().hex[:12],
        description="事件唯一ID（幂等用）",
    )
    event_type: EventType = Field(..., description="事件类型")
    document_id: str = Field(..., description="文档唯一ID")
    file_path: str = Field(..., description="文档文件路径")
    content_hash: str = Field(..., description="当前文档内容 hash")
    old_hash: Optional[str] = Field(None, description="旧 hash（仅 DOCUMENT_UPDATED 时存在）")
    timestamp: int = Field(
        default_factory=lambda: int(time.time()),
        description="事件发生时间戳（秒）",
    )

    def to_json_bytes(self) -> bytes:
        """序列化为 JSON 字节串（发送到 Kafka）"""
        import json
        return json.dumps(self.model_dump(), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "DocumentChangeEvent":
        """从 JSON 字节串反序列化（从 Kafka 消费后解析）"""
        import json
        payload = json.loads(data.decode("utf-8"))
        return cls(**payload)