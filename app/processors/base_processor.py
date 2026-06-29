"""文档处理器基类

定义统一的 read_file / split_content / process 接口。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config


class BaseProcessor(ABC):
    """文档处理器抽象基类"""

    supported_extensions: List[str] = []

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs: Any,
    ):
        self.chunk_size = chunk_size or config.chunk_max_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    @abstractmethod
    def read_file(self, file_path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def split_content(self, content: Any, file_path: str) -> List[Document]:
        raise NotImplementedError

    def process(self, file_path: str) -> List[Document]:
        """完整处理流程：读取文件 -> 内容分片"""
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"处理器[{self.__class__.__name__}] 开始处理: {path}")

        try:
            content = self.read_file(str(path))
            normalized_path = path.as_posix()
            documents = self.split_content(content, normalized_path)

            for doc in documents:
                doc.metadata["_source"] = normalized_path
                doc.metadata["_file_name"] = path.name
                doc.metadata["_extension"] = path.suffix.lower()
                doc.metadata["_processor"] = self.__class__.__name__

            logger.info(
                f"处理器[{self.__class__.__name__}] 处理完成: {path} -> "
                f"{len(documents)} 个分片"
            )
            return documents

        except Exception as e:
            logger.error(
                f"处理器[{self.__class__.__name__}] 处理失败: {path}, 错误: {e}"
            )
            raise

    def _build_metadata(
        self,
        page: Optional[int] = None,
        block_type: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if page is not None:
            metadata["_page"] = page
        if block_type is not None:
            metadata["_block_type"] = block_type
        if extra:
            metadata.update(extra)
        return metadata

    def _secondary_split(
        self, documents: List[Document], min_size: Optional[int] = None
    ) -> List[Document]:
        """对已有分片进行二次分割（当单个分片过大时）"""
        if not documents:
            return []

        min_size = min_size or max(100, self.chunk_size // 3)

        final_docs: List[Document] = []
        for doc in documents:
            if len(doc.page_content) <= self.chunk_size:
                final_docs.append(doc)
                continue

            sub_docs = self._text_splitter.split_documents([doc])
            for sub_doc in sub_docs:
                sub_doc.metadata.update(doc.metadata)
            final_docs.extend(sub_docs)

        return self._merge_small_chunks(final_docs, min_size)

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """合并过小的分片"""
        if not documents:
            return []

        merged_docs: List[Document] = []
        current_doc: Optional[Document] = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                current_doc = Document(
                    page_content=doc.page_content, metadata=dict(doc.metadata)
                )
                continue

            if (
                doc_size < min_size
                and len(current_doc.page_content) + doc_size < self.chunk_size * 1.5
            ):
                current_doc.page_content += "\n\n" + doc.page_content
            else:
                merged_docs.append(current_doc)
                current_doc = Document(
                    page_content=doc.page_content, metadata=dict(doc.metadata)
                )

        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs