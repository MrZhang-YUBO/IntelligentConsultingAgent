"""内容 hash 计算工具

用于检测文档内容是否变更：
- compute_file_hash(file_path) -> SHA-256 of file bytes
- compute_content_hash(text)    -> SHA-256 of text content

设计说明：
- 文件层面用二进制 hash，任何字节变化都能检测到
- 纯文本（如 txt/md）也用文件字节 hash，避免编码差异
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# 每次读取 64KB，大文件也不会 OOM
_CHUNK_READ_SIZE = 64 * 1024


def compute_file_hash(file_path: str | Path) -> str:
    """计算文件内容的 SHA-256 hash

    Args:
        file_path: 文件路径

    Returns:
        "sha256:<64位十六进制>" 格式的 hash 字符串

    Raises:
        FileNotFoundError: 文件不存在
        IOError: 读取失败
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    sha256 = hashlib.sha256()
    total_bytes = 0

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK_READ_SIZE)
                if not chunk:
                    break
                sha256.update(chunk)
                total_bytes += len(chunk)

        hex_digest = sha256.hexdigest()
        logger.debug(
            f"文件 hash 计算完成: {path.name} "
            f"({total_bytes} bytes) -> sha256:{hex_digest[:12]}..."
        )
        return f"sha256:{hex_digest}"

    except Exception as e:
        logger.error(f"计算文件 hash 失败: {path}, 错误: {e}")
        raise IOError(f"计算文件 hash 失败: {e}") from e


def compute_content_hash(text: str) -> str:
    """计算文本内容的 SHA-256 hash

    Args:
        text: 文本内容

    Returns:
        "sha256:<64位十六进制>" 格式的 hash 字符串
    """
    if not text:
        return "sha256:empty"

    # 统一用 UTF-8 编码，避免平台差异
    content_bytes = text.encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(content_bytes)
    hex_digest = sha256.hexdigest()

    logger.debug(f"文本 hash 计算完成 (长度 {len(text)}): sha256:{hex_digest[:12]}...")
    return f"sha256:{hex_digest}"


def compute_metadata_hash(metadata: dict[str, Any]) -> str:
    """计算结构化元数据的 hash（用于检测元数据变更）

    Args:
        metadata: 字典形式的元数据

    Returns:
        "sha256:<64位十六进制>" 格式的 hash 字符串
    """
    if not metadata:
        return "sha256:empty"

    # 用 sort_keys 保证字典 key 顺序不影响 hash
    canonical_json = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    sha256 = hashlib.sha256()
    sha256.update(canonical_json.encode("utf-8"))
    return f"sha256:{sha256.hexdigest()}"


def hashes_equal(hash_a: Optional[str], hash_b: Optional[str]) -> bool:
    """比较两个 hash 是否相等（安全的 None 处理）"""
    if hash_a is None or hash_b is None:
        return hash_a == hash_b
    return hash_a.strip().lower() == hash_b.strip().lower()