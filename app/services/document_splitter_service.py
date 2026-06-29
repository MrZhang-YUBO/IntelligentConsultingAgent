"""文档分割服务（新版：基于处理器工厂）

统一入口：根据文件扩展名自动选择对应处理器。
此文件保留对旧接口的兼容，供 vector_index_service 调用。
"""

from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from app.processors import (
    get_supported_extensions,
    get_processor,
    process_file,
)


class DocumentSplitterService:
    """文档分割服务（新版）

    使用方法：
        service = DocumentSplitterService()
        docs = service.split_file("path/to/file.pdf")
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info("文档分割服务初始化完成（基于处理器工厂）")

    def get_supported_types(self) -> List[str]:
        """获取所有支持的文件扩展名"""
        return get_supported_extensions()

    def split_file(self, file_path: str) -> List[Document]:
        """处理并分割文件（推荐接口）

        Args:
            file_path: 文件路径

        Returns:
            分片后的 Document 列表
        """
        return process_file(
            file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    # ========== 兼容旧接口（保留以避免破坏现有代码） ==========

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """兼容旧接口：根据文件路径扩展名选择处理器

        注意：此方法假设内容已经是纯文本（适合 .txt / .md），
        对于 .pdf / .docx 请使用 split_file() 直接传入文件路径。
        """
        from pathlib import Path

        ext = Path(file_path).suffix.lower() if file_path else ".txt"
        processor = get_processor(
            ext, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        if processor is None:
            # 回退到通用文本处理器
            from app.processors.text_processor import TextProcessor

            processor = TextProcessor(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

        try:
            return processor.split_content(content, file_path)
        except Exception:
            # 任何解析失败都回退到通用文本分割
            from app.processors.text_processor import TextProcessor

            fallback = TextProcessor(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            return fallback.split_content(content, file_path)


# 全局单例
document_splitter_service = DocumentSplitterService()