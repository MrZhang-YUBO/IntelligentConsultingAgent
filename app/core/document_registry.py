"""文档注册表

存储 document_id ↔ file_path ↔ content_hash 的映射关系。

持久化方式：JSON 文件（简单、无需额外数据库）
路径：由 config.document_registry_path 指定，默认 ./data/document_registry.json

核心接口：
- find_by_path(file_path)         -> DocumentRecord | None
- find_by_id(document_id)         -> DocumentRecord | None
- register_new(file_path, hash)   -> DocumentRecord （新文档）
- update_hash(document_id, hash)  -> DocumentRecord （更新 hash）
- mark_deleted(document_id)       -> None
- delete(document_id)             -> None （物理删除 registry 记录）
- list_all()                      -> list[DocumentRecord]
- is_content_changed(file, hash)  -> bool

线程安全：通过 threading.Lock 保护读写操作
"""

import json
import os
import random
import string
import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import config
from app.models.document import DocumentRecord, DocumentStatus


def _generate_document_id() -> str:
    """生成格式为 doc_<8位随机字母数字> 的文档 ID"""
    prefix = config.document_id_prefix
    length = config.document_id_length
    chars = string.ascii_lowercase + string.digits
    random_part = "".join(random.choices(chars, k=length))
    return f"{prefix}{random_part}"


class DocumentRegistry:
    """文档注册表（JSON 文件持久化）"""

    def __init__(self, registry_path: Optional[str] = None):
        self._registry_path = Path(
            registry_path or config.document_registry_path
        ).resolve()
        self._lock = threading.RLock()  # 可重入锁
        self._records: dict[str, DocumentRecord] = {}  # document_id -> record
        self._path_index: dict[str, str] = {}  # file_path -> document_id 反向索引

        # 确保父目录存在
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)

        # 从磁盘加载
        self._load_from_disk()
        logger.info(
            f"文档注册表初始化完成: {self._registry_path}, "
            f"已加载 {len(self._records)} 条记录"
        )

    # ------------------------------------------------------------------
    # 内部工具：磁盘读写
    # ------------------------------------------------------------------
    def _load_from_disk(self) -> None:
        """从 JSON 文件加载 registry"""
        if not self._registry_path.exists():
            logger.info(f"registry 文件不存在，将创建新文件: {self._registry_path}")
            return

        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, dict):
                logger.warning(f"registry 文件格式异常，将忽略: {self._registry_path}")
                return

            self._records.clear()
            self._path_index.clear()

            for doc_id, record_dict in raw_data.items():
                try:
                    record = DocumentRecord(**record_dict)
                    self._records[doc_id] = record
                    self._path_index[record.file_path] = doc_id
                except Exception as e:
                    logger.warning(f"跳过无效的 registry 记录 {doc_id}: {e}")

            logger.info(
                f"从磁盘加载 registry: {len(self._records)} 条记录"
            )

        except Exception as e:
            logger.error(f"加载 registry 失败: {e}")
            # 保留内存中的空状态，避免崩溃

    def _persist_to_disk(self) -> None:
        """将当前 registry 状态写入 JSON 文件"""
        try:
            # 临时文件写入后再 rename，保证原子性
            temp_path = self._registry_path.with_suffix(".json.tmp")
            data = {
                doc_id: record.model_dump()
                for doc_id, record in self._records.items()
            }

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Windows 下 os.replace 也支持覆盖存在的文件
            os.replace(temp_path, self._registry_path)
            logger.debug(f"registry 已持久化到磁盘: {len(self._records)} 条")

        except Exception as e:
            logger.error(f"持久化 registry 失败: {e}")
            # 清理临时文件
            try:
                if "temp_path" in dir() and Path(temp_path).exists():
                    Path(temp_path).unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def find_by_path(self, file_path: str | Path) -> Optional[DocumentRecord]:
        """根据文件路径查找记录

        Args:
            file_path: 文件路径（会被归一化为 Posix 绝对路径）
        """
        normalized = Path(str(file_path)).resolve().as_posix()

        with self._lock:
            doc_id = self._path_index.get(normalized)
            if doc_id is None:
                return None
            return self._records.get(doc_id)

    def find_by_id(self, document_id: str) -> Optional[DocumentRecord]:
        """根据 document_id 查找记录"""
        with self._lock:
            return self._records.get(document_id)

    def list_all(self, include_deleted: bool = False) -> list[DocumentRecord]:
        """列出所有记录

        Args:
            include_deleted: 是否包含已标记为 DELETED 的记录
        """
        with self._lock:
            records = list(self._records.values())
            if not include_deleted:
                records = [r for r in records if r.status != DocumentStatus.DELETED]
            # 按 updated_at 倒序
            records.sort(key=lambda r: r.updated_at, reverse=True)
            return records

    def is_content_changed(
        self,
        file_path: str | Path,
        new_content_hash: str,
    ) -> tuple[bool, Optional[DocumentRecord]]:
        """检查文件内容是否变更

        Returns:
            (是否变更, 已有记录或 None)
        """
        record = self.find_by_path(file_path)
        if record is None:
            return (True, None)  # 不存在视为变更（新文档）

        # 状态为 DELETED 的也视为变更
        if record.status == DocumentStatus.DELETED:
            return (True, record)

        # hash 不同则为变更
        if record.content_hash != new_content_hash:
            return (True, record)

        return (False, record)

    # ------------------------------------------------------------------
    # 变更接口
    # ------------------------------------------------------------------
    def register_new(
        self,
        file_path: str | Path,
        file_name: str,
        content_hash: str,
    ) -> DocumentRecord:
        """注册一个新文档

        Args:
            file_path: 文件路径
            file_name: 文件名
            content_hash: 内容 hash

        Returns:
            新创建的 DocumentRecord

        Raises:
            ValueError: 如果该路径已存在记录
        """
        path_obj = Path(str(file_path)).resolve()
        normalized = path_obj.as_posix()
        file_name = file_name or path_obj.name

        with self._lock:
            existing = self._path_index.get(normalized)
            if existing is not None:
                raise ValueError(
                    f"文件路径已注册: {normalized} (document_id={existing})"
                )

            document_id = _generate_document_id()
            # 确保 ID 不冲突（极小概率）
            while document_id in self._records:
                document_id = _generate_document_id()

            record = DocumentRecord(
                document_id=document_id,
                file_path=normalized,
                file_name=file_name,
                content_hash=content_hash,
                chunk_count=0,
                status=DocumentStatus.PENDING,
            )

            self._records[document_id] = record
            self._path_index[normalized] = document_id
            self._persist_to_disk()

            logger.info(
                f"注册新文档: {file_name} -> {document_id}, "
                f"hash={content_hash[:20]}..."
            )
            return record

    def update_hash(
        self,
        document_id: str,
        new_content_hash: str,
        chunk_count: Optional[int] = None,
    ) -> Optional[DocumentRecord]:
        """更新文档的 content_hash（文档内容变更时调用）

        Returns:
            更新后的 DocumentRecord，或 None（document_id 不存在）
        """
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                logger.warning(f"update_hash: document_id 不存在: {document_id}")
                return None

            old_hash = record.content_hash
            record.content_hash = new_content_hash
            if chunk_count is not None:
                record.chunk_count = chunk_count
            record.status = DocumentStatus.PENDING
            record.touch()
            self._persist_to_disk()

            logger.info(
                f"更新文档 hash: {document_id}, "
                f"{old_hash[:20]}... -> {new_content_hash[:20]}..."
            )
            return record

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
        last_event_id: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        """更新文档状态（Consumer 处理前后调用）"""
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                logger.warning(f"update_status: document_id 不存在: {document_id}")
                return None

            record.status = status
            if chunk_count is not None:
                record.chunk_count = chunk_count
            if error_message is not None:
                record.error_message = error_message
            if last_event_id is not None:
                record.last_event_id = last_event_id
            record.touch()
            self._persist_to_disk()
            logger.debug(f"更新文档状态: {document_id} -> {status}")
            return record

    def mark_deleted(self, document_id: str) -> Optional[DocumentRecord]:
        """标记文档为已删除（软删除，保留记录）"""
        return self.update_status(document_id, DocumentStatus.DELETED)

    def delete(self, document_id: str) -> bool:
        """物理删除 registry 记录（慎用，一般用 mark_deleted 软删除）

        Returns:
            True 表示删除成功，False 表示记录不存在
        """
        with self._lock:
            record = self._records.pop(document_id, None)
            if record is None:
                return False
            self._path_index.pop(record.file_path, None)
            self._persist_to_disk()
            logger.info(f"物理删除 registry 记录: {document_id} ({record.file_name})")
            return True

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        """返回统计信息"""
        with self._lock:
            stats_dict = {
                "total": len(self._records),
                "active": 0,
                "pending": 0,
                "processing": 0,
                "error": 0,
                "deleted": 0,
            }
            for record in self._records.values():
                key = record.status.value
                if key in stats_dict:
                    stats_dict[key] += 1
            return stats_dict


# 全局单例
document_registry = DocumentRegistry()