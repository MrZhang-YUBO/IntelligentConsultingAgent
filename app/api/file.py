"""文件上传接口模块（支持多格式：txt / md / pdf / docx）

POST /api/upload            -> 上传单个文件并自动建索引
POST /api/index_directory   -> 批量索引目录下的所有支持文件
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.vector_index_service import vector_index_service
from loguru import logger

router = APIRouter()

# 文件上传后存储的路径
UPLOAD_DIR = Path("./uploads")

# 单个文件支持最大大小
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _get_allowed_extensions() -> list:
    """从处理器工厂获取支持的扩展名（不含点号，小写）

    这样新增处理器时无需改此文件。
    """
    from app.processors import get_supported_extensions

    return get_supported_extensions()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件并自动创建向量索引

    支持的格式：txt / md / markdown / pdf / docx
    """
    try:
        # 1. 验证文件
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        # 2. 规范化文件名
        safe_filename = _sanitize_filename(file.filename)

        # 3. 验证文件扩展名（从处理器工厂获取支持列表）
        file_extension = _get_file_extension(safe_filename)
        allowed_extensions = _get_allowed_extensions()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"不支持的文件格式: .{file_extension}。"
                    f"仅支持: {', '.join('.' + ext for ext in allowed_extensions)}"
                ),
            )

        # 4. 创建上传目录
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 5. 保存文件
        file_path = UPLOAD_DIR / safe_filename

        if file_path.exists():
            logger.info(f"文件已存在，将覆盖: {file_path}")
            file_path.unlink()

        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制（最大 {MAX_FILE_SIZE} 字节，当前 {len(content)} 字节）",
            )

        file_path.write_bytes(content)
        logger.info(f"文件上传成功: {file_path} (大小: {len(content)} 字节)")

        # 6. 自动创建向量索引（调用新版 processor 流程）
        try:
            logger.info(f"开始为上传文件创建向量索引: {file_path}")
            vector_index_service.index_single_file(str(file_path))
            logger.info(f"向量索引创建成功: {file_path}")
        except Exception as e:
            logger.error(f"向量索引创建失败: {file_path}, 错误: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"文件上传成功，但索引创建失败: {e}",
            )

        # 7. 返回响应
        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "filename": safe_filename,
                    "file_path": str(file_path),
                    "size": len(content),
                    "extension": file_extension,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")


@router.post("/index_directory")
async def index_directory(directory_path: str = None):
    """索引指定目录下的所有支持文件"""
    try:
        logger.info(f"开始索引目录: {directory_path or 'uploads'}")
        result = vector_index_service.index_directory(directory_path)

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": result.to_dict(),
            },
        )

    except Exception as e:
        logger.error(f"索引目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引目录失败: {e}")


def _get_file_extension(filename: str) -> str:
    """获取文件扩展名（小写，不含点）"""
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].lower()
    return ""


def _sanitize_filename(filename: str) -> str:
    """规范化文件名，去除空格和特殊字符"""
    sanitized = filename.replace(" ", "_")
    for char in ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]:
        sanitized = sanitized.replace(char, "_")
    return sanitized