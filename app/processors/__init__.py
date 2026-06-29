"""文档处理器工厂模块

根据文件扩展名自动选择合适的处理器进行文档读取和分片。
"""

from typing import Dict, List, Optional, Type

from langchain_core.documents import Document

from app.processors.base_processor import BaseProcessor
from app.processors.text_processor import TextProcessor
from app.processors.markdown_processor import MarkdownProcessor
from app.processors.pdf_processor import PdfProcessor
from app.processors.word_processor import WordProcessor


PROCESSOR_MAP: Dict[str, Type[BaseProcessor]] = {
    ".txt": TextProcessor,
    ".md": MarkdownProcessor,
    ".markdown": MarkdownProcessor,
    ".pdf": PdfProcessor,
    ".docx": WordProcessor,
}


def get_supported_extensions() -> List[str]:
    """获取所有支持的文件扩展名（不含点，小写）"""
    return list(set(ext.lstrip(".") for ext in PROCESSOR_MAP.keys()))


def get_processor_class(extension: str) -> Optional[Type[BaseProcessor]]:
    """根据扩展名获取处理器类"""
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    return PROCESSOR_MAP.get(ext)


def get_processor(extension: str, **kwargs) -> Optional[BaseProcessor]:
    """根据扩展名创建处理器实例"""
    cls = get_processor_class(extension)
    if cls is None:
        return None
    return cls(**kwargs)


def process_file(file_path: str, **kwargs) -> List[Document]:
    """处理单个文件并返回分片结果"""
    from pathlib import Path

    path = Path(file_path)
    ext = path.suffix.lower()

    processor = get_processor(ext, **kwargs)
    if processor is None:
        raise ValueError(
            f"不支持的文件类型: {ext}。支持的类型: {', '.join(get_supported_extensions())}"
        )

    return processor.process(file_path)