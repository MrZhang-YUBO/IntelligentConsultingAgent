"""PDF 文档处理器 (.pdf)

使用 PyMuPDF (fitz) 进行内容提取，核心特性：
  1. 逐页提取文本（保留换行和段落结构）
  2. 表格识别：使用 fitz.Page.find_tables() 检测表格，转 Markdown 表格格式
  3. 代码块：通过检测连续缩进行 + 代码特征关键字识别，包裹为 Markdown 代码块
  4. 图片提取：识别页面中的图片，记录位置和大小，作为 [图片 N] 占位符存入文本
  5. 分片策略：按"段落+表格+图片"分组，再用 RecursiveCharacterTextSplitter 二次分割
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger
import fitz
from app.processors.base_processor import BaseProcessor


class PdfProcessor(BaseProcessor):
    """PDF 文档处理器"""

    supported_extensions = [".pdf"]

    # 用于识别代码块的特征关键字
    _CODE_KEYWORDS = [
        "def ",
        "function",
        "class ",
        "import ",
        "return ",
        "if ",
        "for ",
        "while ",
        "const ",
        "let ",
        "var ",
        "console.log",
        "print(",
        "SELECT ",
        "CREATE TABLE",
        "```",
        "{",
        "}",
        ";",
    ]

    # 用于识别代码块的连续缩进行阈值
    _INDENT_THRESHOLD = 3  # 连续3行以上带缩进视为代码块

    def read_file(self, file_path: str) -> List[Dict]:
        """读取 PDF 文件，逐页提取结构化内容

        返回结构（按页）：
        [
            {
                "page_num": 1,
                "blocks": [
                    {"type": "paragraph", "content": "..."}
                    {"type": "table", "content": "| a | b |\n| --- | --- |\n| 1 | 2 |"}
                    {"type": "code", "content": "```python\ndef foo():...\n```"}
                    {"type": "image", "content": "[图片 i-j] (w x h)"}
                ]
            },
            ...
        ]
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF 未安装，请运行: pip install pymupdf"
            ) from e

        path = Path(file_path)
        doc = fitz.open(str(path))
        pages_data: List[Dict] = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = {
                    "page_num": page_num + 1,  # 页码从 1 开始
                    "blocks": self._extract_page_blocks(page, page_num),
                }
                pages_data.append(page_data)
        finally:
            doc.close()

        logger.info(
            f"PDF 读取完成: {file_path}, 共 {len(pages_data)} 页, "
            f"总块数: {sum(len(p['blocks']) for p in pages_data)}"
        )
        return pages_data

    def split_content(
        self, content: List[Dict], file_path: str
    ) -> List[Document]:
        """对 PDF 内容进行分片

        策略：
          1. 以"页"为基本单位，收集所有内容块的纯文本
          2. 表格作为完整单元，不被切开
          3. 使用 RecursiveCharacterTextSplitter 对长段落做二次分割
          4. 所有分片均携带 _page 元数据
        """
        documents: List[Document] = []

        for page_data in content:
            page_num = page_data["page_num"]
            blocks = page_data["blocks"]

            page_text_parts: List[str] = []
            has_table = False
            has_image = False
            has_code = False

            for block in blocks:
                block_type = block["type"]
                block_content = block["content"]

                if block_type == "table":
                    has_table = True
                elif block_type == "image":
                    has_image = True
                elif block_type == "code":
                    has_code = True

                page_text_parts.append(block_content)

            # 组装整页文本（段落之间用双换行分隔）
            page_text = "\n\n".join(page_text_parts).strip()
            if not page_text:
                continue

            # 构建该页的基础 metadata
            page_metadata = self._build_metadata(
                page=page_num,
                block_type="page",
                extra={
                    "_has_table": has_table,
                    "_has_image": has_image,
                    "_has_code": has_code,
                },
            )

            # 如果整页文本小于 chunk_size，直接作为一个分片
            if len(page_text) <= self.chunk_size:
                documents.append(
                    Document(page_content=page_text, metadata=dict(page_metadata))
                )
                continue

            # 否则按块进行分组，再分割
            block_based_docs = self._split_by_blocks(blocks, page_num, page_metadata)
            documents.extend(block_based_docs)

        # 合并过小的分片 + 二次分割
        return self._secondary_split(documents, min_size=250)

    # ============= 内部方法 =============

    def _extract_page_blocks(self, page, page_num: int) -> List[Dict]:
        """从单个 PDF 页面中提取结构化内容块

        提取顺序：文本 -> 表格 -> 图片
        注意：表格需要用专门的表格检测器识别，而非简单文本提取
        """
        blocks: List[Dict] = []
        image_counter = 0

        # Step 1: 提取表格（表格优先，因为表格区域会被 get_text 重复提取）
        table_regions: List[Tuple[fitz.Rect, str]] = []
        try:
            tables = page.find_tables()
            if tables and tables.tables:
                for idx, table in enumerate(tables.tables):
                    try:
                        table_data = table.extract()
                        if table_data and len(table_data) > 0:
                            markdown_table = self._table_to_markdown(table_data)
                            if markdown_table.strip():
                                blocks.append(
                                    {"type": "table", "content": markdown_table}
                                )
                                if table.bbox:
                                    table_regions.append(
                                        (fitz.Rect(table.bbox), markdown_table)
                                    )
                    except Exception as table_err:
                        logger.warning(
                            f"第 {page_num + 1} 页表格 {idx} 解析失败: {table_err}"
                        )
        except Exception as e:
            logger.debug(f"第 {page_num + 1} 页表格检测失败（可能是老版本 PyMuPDF）: {e}")

        # Step 2: 提取图片（记录图片位置，作为文字占位符）
        image_info_list = []
        try:
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                image_counter += 1
                # 尝试获取图片在页面上的位置
                try:
                    img_rect = page.get_image_rects(xref)
                    img_size = (
                        f"{int(img_rect[0].width)}x{int(img_rect[0].height)}"
                        if img_rect
                        else "unknown"
                    )
                except Exception:
                    img_size = "unknown"

                image_info_list.append(
                    f"[图片 {page_num + 1}-{image_counter}] (尺寸: {img_size})"
                )
        except Exception as e:
            logger.debug(f"第 {page_num + 1} 页图片提取失败: {e}")

        # Step 3: 提取普通文本（按"块"分组，保留段落结构）
        text_blocks = []
        try:
            raw_dict = page.get_text("dict")
            for block in raw_dict.get("blocks", []):
                if block.get("type") == 0:  # 0 表示文本块
                    block_text_lines = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        line_text = line_text.strip()
                        if line_text:
                            block_text_lines.append(line_text)
                    if block_text_lines:
                        text_blocks.append("\n".join(block_text_lines))
        except Exception as e:
            # 如果 dict 提取失败，回退到纯文本提取
            logger.debug(f"第 {page_num + 1} 页 dict 提取失败，回退到纯文本: {e}")
            fallback_text = page.get_text().strip()
            if fallback_text:
                text_blocks = [fallback_text]

        # 过滤掉落在表格区域内的文本块（避免表格被重复提取为普通文本）
        filtered_text_blocks = []
        for tb in text_blocks:
            # 简单启发式：如果文本块内容和某个表格内容高度重叠，则跳过
            is_duplicate = False
            for _, table_md in table_regions:
                # 取表格每一行的第一个非空单元格作为关键标识
                table_keywords = [
                    cell.strip()
                    for row in table_md.split("\n")[2:]
                    if row.strip() and not row.startswith("| ---")
                    for cell in [row.split("|")[1].strip() if "|" in row else ""]
                    if cell
                ]
                if table_keywords and any(kw and kw in tb for kw in table_keywords[:3]):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_text_blocks.append(tb)

        # Step 4: 识别代码块（连续缩进行 + 代码关键字特征）
        processed_blocks = self._detect_and_wrap_code_blocks(filtered_text_blocks)

        # Step 5: 按顺序组装（先图片占位符，再文本，再表格）
        # 注意：实际顺序更合理的应该是按页面坐标排序，但简化处理为：文本 -> 表格 -> 图片占位符

        for block_text in processed_blocks:
            if block_text.startswith("```"):
                blocks.append({"type": "code", "content": block_text})
            else:
                blocks.append({"type": "paragraph", "content": block_text})

        # 表格已经在 Step 1 加入（保留在 blocks 列表中，但需要确保顺序正确）
        # 简化处理：如果已存在 table 块，则保持原有顺序

        # 图片占位符加到该页末尾
        for img_info in image_info_list:
            blocks.append({"type": "image", "content": img_info})

        return blocks

    def _detect_and_wrap_code_blocks(self, text_blocks: List[str]) -> List[str]:
        """识别代码块并包裹为 Markdown 代码块

        识别逻辑：
          - 检测连续多行具有明显缩进特征（前导空格/tab）
          - 检测代码关键字（def/class/import/if/for 等）
        """
        if not text_blocks:
            return text_blocks

        processed: List[str] = []
        i = 0

        while i < len(text_blocks):
            block = text_blocks[i]

            # 检查是否为单行代码块（包含明显代码关键字）
            single_line_keywords = sum(
                1 for kw in self._CODE_KEYWORDS if kw in block
            )
            is_single_line_code = (
                single_line_keywords >= 2
                and len(block.split("\n")) <= 5
                and any(c in block for c in ["{", "}", ";", "(", ")", "="])
            )

            if is_single_line_code:
                processed.append(f"```text\n{block}\n```")
                i += 1
                continue

            # 检查是否为多行代码块（连续缩进）
            lines = block.split("\n")
            indent_count = 0
            code_start = -1

            for j, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                # 有缩进或包含代码特征
                has_indent = line.startswith("  ") or line.startswith("\t")
                has_code_feature = any(kw in line for kw in self._CODE_KEYWORDS)

                if has_indent or has_code_feature:
                    indent_count += 1
                    if code_start == -1:
                        code_start = j
                else:
                    indent_count = 0
                    code_start = -1

            # 如果多行中超过一半有代码特征，则视为代码块
            non_empty_lines = [l for l in lines if l.strip()]
            if (
                len(non_empty_lines) >= 3
                and sum(
                    1
                    for l in non_empty_lines
                    if any(kw in l for kw in self._CODE_KEYWORDS)
                )
                >= len(non_empty_lines) * 0.5
            ):
                processed.append(f"```text\n{block}\n```")
            else:
                processed.append(block)

            i += 1

        return processed

    def _split_by_blocks(
        self,
        blocks: List[Dict],
        page_num: int,
        base_metadata: Dict,
    ) -> List[Document]:
        """按内容块进行分组分片（表格保持完整不被切开）"""
        documents: List[Document] = []
        current_text: List[str] = []
        current_size = 0

        def flush_current():
            nonlocal current_text, current_size
            if current_text:
                text = "\n\n".join(current_text).strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata=dict(base_metadata),
                        )
                    )
                current_text = []
                current_size = 0

        for block in blocks:
            block_type = block["type"]
            block_content = block["content"]
            block_len = len(block_content)

            # 表格作为独立单元（不切开）
            if block_type == "table":
                flush_current()
                table_metadata = dict(base_metadata)
                table_metadata["_block_type"] = "table"
                documents.append(
                    Document(page_content=block_content, metadata=table_metadata)
                )
                continue

            # 图片作为独立单元
            if block_type == "image":
                image_metadata = dict(base_metadata)
                image_metadata["_block_type"] = "image"
                # 小图片说明追加到当前段落，或作为独立分片
                if current_size + block_len < self.chunk_size:
                    current_text.append(block_content)
                    current_size += block_len
                else:
                    flush_current()
                    documents.append(
                        Document(page_content=block_content, metadata=image_metadata)
                    )
                continue

            # 代码块作为独立单元
            if block_type == "code":
                flush_current()
                code_metadata = dict(base_metadata)
                code_metadata["_block_type"] = "code"
                # 如果代码太长，则用二次分割器
                if block_len <= self.chunk_size:
                    documents.append(
                        Document(page_content=block_content, metadata=code_metadata)
                    )
                else:
                    sub_docs = self._text_splitter.create_documents(
                        texts=[block_content], metadatas=[code_metadata]
                    )
                    documents.extend(sub_docs)
                continue

            # 普通段落：累积到当前分片
            if current_size + block_len > self.chunk_size and current_text:
                flush_current()

            current_text.append(block_content)
            current_size += block_len

        flush_current()
        return documents

    def _table_to_markdown(self, table_data: List[List[Optional[str]]]) -> str:
        """将二维表格数据转 Markdown 表格格式"""
        if not table_data or not table_data[0]:
            return ""

        # 清理 None 和空字符串
        cleaned = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table_data
        ]

        # 过滤完全空的行
        cleaned = [row for row in cleaned if any(row)]
        if not cleaned:
            return ""

        # 构建 Markdown 表格
        col_count = max(len(row) for row in cleaned)
        # 对齐每行列数
        for row in cleaned:
            while len(row) < col_count:
                row.append("")

        # 表头和分隔
        header = cleaned[0]
        separator = ["---"] * col_count
        rows = cleaned[1:]

        # 组装
        lines = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(separator) + " |")
        for row in rows:
            # 转义单元格中的 | 字符
            escaped = [c.replace("|", "\\|") for c in row]
            lines.append("| " + " | ".join(escaped) + " |")

        return "\n".join(lines)