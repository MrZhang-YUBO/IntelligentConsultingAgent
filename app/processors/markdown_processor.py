"""Markdown (.md, .markdown) 处理器

利用 LangChain 内置的 MarkdownHeaderTextSplitter 按标题层级分割，
保留文档的章节结构信息。
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.processors.base_processor import BaseProcessor


class MarkdownProcessor(BaseProcessor):
    """Markdown 文档处理器"""

    supported_extensions = [".md", ".markdown"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 按一级和二级标题分割，保留文档结构
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
            ],
            strip_headers=False,
        )

    def read_file(self, file_path: str) -> str:
        """读取 Markdown 文件（UTF-8 编码）"""
        path = Path(file_path)
        return path.read_text(encoding="utf-8")

    def split_content(self, content: str, file_path: str) -> List[Document]:
        """Markdown 分片策略（三阶段）：

        阶段 1：按 # / ## 标题分割为多个文档块（保留 h1/h2 标题到 metadata）
        阶段 2：对每个块使用 RecursiveCharacterTextSplitter 做二次分割
        阶段 3：合并过小的分片（<300 字符），避免碎片化
        """
        if not content or not content.strip():
            return []

        # 阶段 1：按标题分割
        md_docs = self._md_splitter.split_text(content)

        # 阶段 2：按大小进一步分割
        docs_after_split: List[Document] = []
        for doc in md_docs:
            if len(doc.page_content) <= self.chunk_size:
                docs_after_split.append(doc)
            else:
                sub_docs = self._text_splitter.split_documents([doc])
                for sub_doc in sub_docs:
                    sub_doc.metadata.update(doc.metadata)
                docs_after_split.extend(sub_docs)

        # 阶段 3：合并过小的分片
        final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

        # 为每个分片补充 _block_type 元数据
        for doc in final_docs:
            if "_block_type" not in doc.metadata:
                doc.metadata["_block_type"] = "paragraph"

        return final_docs