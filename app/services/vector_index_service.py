"""向量索引服务 - 动态更新方案

核心流程（事件驱动 + Hash 检测 + 先删后增：

【Producer 侧（同步，HTTP请求中）：
1. 计算文件 content_hash
2. 查询 document_registry
   - 不存在 → 新文档，生成 document_id
   - 存在但 hash 相同 → 跳过
   - 存在且 hash 不同 → 更新
3. 发送 Kafka 事件（DOCUMENT_CREATED / DOCUMENT_UPDATED）

【Consumer 侧（异步，后台线程）：
1. 接收 Kafka 事件
2. DOCUMENT_UPDATED → 用 document_id 删除旧 chunk
3. 重新切分文件
4. 重新入库（带 document_id）
5. 更新 registry（hash, chunk_count, status）
"""

import os
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.config import config
from app.core.document_registry import document_registry
from app.core.kafka_client import DocumentChangeEvent, send_event
from app.models.document import (
    DocumentStatus,
    DocumentRecord,
    EventType,
)
from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager
from app.utils.hash_utils import compute_file_hash


class IndexingResult:
    """索引结果（供 HTTP API 返回）"""

    def __init__(self) -> None:
        self.success: bool = False
        self.event_type: Optional[EventType] = None
        self.document_id: Optional[str] = None
        self.file_path: str = ""
        self.content_hash: str = ""
        self.chunk_count: int = 0
        self.elapsed_ms: int = 0
        self.error_message: Optional[str] = None
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "event_type": self.event_type.value if self.event_type else None,
            "document_id": self.document_id,
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "chunk_count": self.chunk_count,
            "elapsed_ms": self.elapsed_ms,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class VectorIndexService:
    """向量索引服务（动态更新版）"""

    def __init__(self):
        self.upload_path = config.upload_dir if hasattr(config, "upload_dir") else "./uploads"
        logger.info("向量索引服务初始化完成（事件驱动 + Hash 检测）")

    # ------------------------------------------------------------------
    # HTTP 侧 API：接收文件 -> 检测变更 -> 发事件（快速返回
    # ------------------------------------------------------------------
    def index_single_file(
        self,
        file_path: str,
        force: bool = False,
    ) -> IndexingResult:
        """索引单个文件（Producer 侧）

        同步逻辑：
        - 计算 hash → 比对 registry → 发事件
        - 立即返回（实际切分/入库由 Consumer 异步处理

        Args:
            file_path: 文件路径
            force: 是否强制重新索引（忽略 hash 比较）

        Returns:
            IndexingResult（含 event_type 和 document_id）
        """
        result = IndexingResult()
        start_time = time.time()
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            result.error_message = f"文件不存在: {file_path}"
            return result

        result.file_path = path.as_posix()

        try:
            # Step 1: 计算 content hash
            logger.info(f"开始索引文件: {path}")
            content_hash = compute_file_hash(path)
            result.content_hash = content_hash
            logger.debug(f"文件 hash 计算完成: {content_hash[:24]}...")

            # Step 2: 查询 registry，判断是否变更
            is_changed, existing_record = document_registry.is_content_changed(
                path.as_posix(), content_hash
            )

            # 强制模式：即使 hash 相同也视为变更
            if not is_changed and not force:
                # 内容未变更 → 跳过
                if existing_record:
                    result.success = True
                    result.event_type = EventType.DOCUMENT_UNCHANGED
                    result.document_id = existing_record.document_id
                    result.chunk_count = existing_record.chunk_count
                    logger.info(
                        f"文件内容未变更，跳过索引: {path.name} "
                        f"(document_id={existing_record.document_id})"
                    )
                else:
                    # 理论上不会走到这（registry 没有记录但 is_changed=False）
                    # 但保守处理：走新建流程
                    is_changed = True

            if is_changed or force:
                # Step 3: 生成 document_id（或更新
                document_id = self._resolve_document_id(
                    existing_record, path, content_hash
                )
                result.document_id = document_id

                # Step 4: 构造并发送 Kafka 事件
                event_type = (
                    EventType.DOCUMENT_UPDATED
                    if existing_record
                    else EventType.DOCUMENT_CREATED
                )
                result.event_type = event_type

                event = DocumentChangeEvent(
                    event_type=event_type,
                    document_id=document_id,
                    file_path=path.as_posix(),
                    content_hash=content_hash,
                    old_hash=(
                    existing_record.content_hash if existing_record else None
                ),
            )

            send_ok = send_event(event)
            if not send_ok:
                logger.warning(
                    f"Kafka 事件发送失败，将在 Consumer 离线处理: {document_id}"
                )
                # 兜底：本地直接执行（退化到同步模式）
                self._process_event_sync(event)
                result.metadata["processed_locally"] = True
            else:
                result.metadata["event_id"] = event.event_id

            result.success = True
            logger.info(
                f"文件已提交索引: {path.name} -> {event_type.value}, "
                f"document_id={document_id}"
            )

        except Exception as e:
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            result.success = False
            result.error_message = str(e)

        finally:
            result.elapsed_ms = int((time.time() - start_time) * 1000)

        return result

    def index_directory(
        self,
        directory_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """索引目录下所有支持的文件（Producer 侧

        遍历目录，逐个发送事件

        Args:
            directory_path: 目录路径

        Returns:
            汇总结果
        """
        summary = {
            "total": 0,
            "created": 0,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
            "failed_files": {},
        }

        target_path = Path(directory_path or self.upload_path).resolve()
        if not target_path.exists() or not target_path.is_dir():
            return {"error": f"目录不存在: {target_path}"}

        logger.info(f"开始索引目录: {target_path}")

        try:
            supported_suffixes = self._get_supported_suffixes()
            all_files = list(target_path.rglob("*"))
            files = [
                f for f in all_files
                if f.is_file() and f.suffix.lower() in supported_suffixes
            ]

            logger.info(f"找到 {len(files)} 个候选文件")

            for file_path in files:
                try:
                    result = self.index_single_file(str(file_path))
                    summary["total"] += 1

                    if result.success:
                        etype = result.event_type
                        if etype == EventType.DOCUMENT_CREATED:
                            summary["created"] += 1
                        elif etype == EventType.DOCUMENT_UPDATED:
                            summary["updated"] += 1
                        elif etype == EventType.DOCUMENT_UNCHANGED:
                            summary["unchanged"] += 1
                    else:
                        summary["failed"] += 1
                        summary["failed_files"][str(file_path)] = result.error_message
                except Exception as e:
                    summary["failed"] += 1
                    summary["failed_files"][str(file_path)] = str(e)

            logger.info(
                f"目录索引完成: total={summary['total']}, "
                f"created={summary['created']}, "
                f"updated={summary['updated']}, "
                f"unchanged={summary['unchanged']}, "
                f"failed={summary['failed']}"
            )
            return summary

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            return {"error": str(e)}

    def delete_document_by_id(self, document_id: str) -> bool:
        """删除某文档的所有索引

        1. 删除 Milvus 中所有 chunk
        2. registry 标记为 DELETED

        Args:
            document_id: 文档 ID

        Returns:
            True 表示成功
        """
        try:
            logger.info(f"删除文档索引: document_id={document_id}")

            deleted = vector_store_manager.delete_by_document_id(document_id)

            record = document_registry.find_by_id(document_id)
            if record is not None:
                document_registry.mark_deleted(document_id)

                # 发送删除事件（可选，让其他节点也能同步）
                event = DocumentChangeEvent(
                    event_type=EventType.DOCUMENT_DELETED,
                    document_id=document_id,
                    file_path=record.file_path,
                    content_hash=record.content_hash,
                    old_hash=None,
                )
                send_event(event)

            logger.info(f"文档删除完成: {document_id}, 删除 {deleted} 个 chunk")
            return True

        except Exception as e:
            logger.error(f"删除文档索引失败: {document_id}, err={e}")
            return False

    def list_documents(
        self,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """列出所有已索引文档"""
        records = document_registry.list_all(include_deleted=include_deleted)
        return [
            {
                "document_id": r.document_id,
                "file_path": r.file_path,
                "file_name": r.file_name,
                "content_hash": r.content_hash,
                "chunk_count": r.chunk_count,
                "status": r.status.value,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Consumer 侧：接收事件 -> 删旧 -> 切分 -> 入库 -> 更新 registry
    # ------------------------------------------------------------------
    def process_event(self, event: DocumentChangeEvent) -> None:
        """处理 Kafka 事件（Consumer 调用）"""
        start_time = time.time()
        document_id = event.document_id
        file_path = event.file_path
        content_hash = event.content_hash

        logger.info(
            f"[Consumer] 处理事件: type={event.event_type.value}, "
            f"document_id={document_id}, file={Path(file_path).name}"
        )

        # 更新状态为 PROCESSING
        document_registry.update_status(
            document_id, DocumentStatus.PROCESSING, last_event_id=event.event_id
        )

        try:
            if event.event_type == EventType.DOCUMENT_DELETED:
                # 删除事件：仅删除索引，registry 中已在 mark_deleted
                vector_store_manager.delete_by_document_id(document_id)
                document_registry.mark_deleted(document_id)
                logger.info(f"[Consumer] 删除事件处理完成: {document_id}")
                return

            # 其他事件类型（CREATED / UPDATED）通用处理）：先删后增
            # 1. 删除旧 chunk（UPDATED 时需要，CREATED 时也执行作为防御性删除，可能无数据）
            if event.event_type == EventType.DOCUMENT_UPDATED:
                deleted = vector_store_manager.delete_by_document_id(document_id)
                logger.info(
                    f"[Consumer] 删除旧 chunk: document_id={document_id}, 删除 {deleted} 个"
                )

            # 2. 切分文件
            documents = document_splitter_service.split_file(file_path)
            chunk_count = len(documents)

            if chunk_count == 0:
                logger.warning(f"[Consumer] 文件切分结果为空: {file_path}")
                document_registry.update_status(
                    document_id, DocumentStatus.ERROR, chunk_count=0, error_message="empty content"
                )
                return

            logger.info(f"[Consumer] 文档切分完成: {file_path} -> {chunk_count} 个分片")

            # 3. 入库（带 document_id）
            vector_store_manager.add_documents_with_doc_id(
                documents, document_id
            )

            # 4. 更新 registry（hash、chunk_count、状态
            document_registry.update_hash(
                document_id, content_hash, chunk_count=chunk_count
            )
            document_registry.update_status(
                document_id, DocumentStatus.ACTIVE, chunk_count=chunk_count
            )

            elapsed = int((time.time() - start_time) * 1000)
            logger.info(
                f"[Consumer] 事件处理完成: {document_id}, "
                f"chunk_count={chunk_count}, 耗时 {elapsed}ms"
            )

        except Exception as e:
            logger.error(f"[Consumer] 事件处理失败: {document_id}, err={e}")
            document_registry.update_status(
                document_id, DocumentStatus.ERROR, error_message=str(e)
            )

    def _process_event_sync(self, event: DocumentChangeEvent) -> None:
        """同步处理事件（Producer 发送失败时的兜底，用于调试）"""
        self.process_event(event)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _resolve_document_id(
        self,
        existing_record: Optional[DocumentRecord], file_path: Path, content_hash: str
    ) -> str:
        """获取或创建 document_id

        有旧记录：用旧 ID；没有则创建新记录"""
        if existing_record is not None:
            return existing_record.document_id

        # 新文档：在 registry 中创建新记录
        new_record = document_registry.register_new(
            file_path.as_posix(), file_path.name, content_hash
        )
        return new_record.document_id

    def _get_supported_suffixes(self) -> set[str]:
        """获取支持的文件扩展名集合"""
        try:
            return document_splitter_service.get_supported_types()
        except Exception:
            # 回退到常见格式
            return {".txt", ".md", ".pdf", ".docx", ".md"}


# 全局单例
vector_index_service = VectorIndexService()