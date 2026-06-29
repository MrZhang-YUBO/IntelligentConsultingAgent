"""纯文本 (.txt) 处理器

最简单的处理器：直接读取文本内容，使用通用递归字符分割器分片。
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.processors.base_processor import BaseProcessor


class TextProcessor(BaseProcessor):
    """纯文本文件处理器"""

    supported_extensions = [".txt"]

    def read_file(self, file_path: str) -> str:
        """读取文本文件内容

        优先使用 utf-8 编码读取，如果失败则尝试 gbk 编码（兼容 Windows 中文 txt）。
        """
        path = Path(file_path)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="gbk", errors="replace")

    def split_content(self, content: str, file_path: str) -> List[Document]:
        """对纯文本内容进行分片

        策略：
          1. 使用 RecursiveCharacterTextSplitter 进行通用分割
          2. 对过小的分片进行合并
        """
        if not content or not content.strip():
            return []

        # 直接使用递归字符分割器
        documents = self._text_splitter.create_documents(
            texts=[content],
            metadatas=[
                self._build_metadata(
                    block_type="paragraph",
                    extra={"_source_file": file_path},
                )
            ],
        )

        # 合并过小的分片
        final_docs = self._merge_small_chunks(documents, min_size=200)
        return final_docs