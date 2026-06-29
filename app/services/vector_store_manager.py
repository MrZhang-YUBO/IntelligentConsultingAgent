"""向量存储管理器 - 封装 Milvus VectorStore 操作

核心改造（动态更新方案）：
1. chunk_id = {document_id}_{chunk_index:04d}  → 可通过前缀定位某文档的所有 chunk
2. Milvus collection 有独立 document_id 字段  → 可用表达式批量删除
3. add_documents_with_doc_id(documents, doc_id) → 关联 ID 入库
4. delete_by_document_id(document_id)           → 按文档批量删除
"""

import time
from typing import List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


# 统一使用 biz collection
COLLECTION_NAME = "biz"


def _format_chunk_id(document_id: str, chunk_index: int) -> str:
    """生成格式为 {document_id}_{chunk_index:04d} 的 chunk ID"""
    return f"{document_id}_{chunk_index:04d}"


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store: Optional[Milvus] = None
        self.collection_name = COLLECTION_NAME
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore"""
        try:
            # 必须在 PyMilvus / langchain_milvus 访问 Collection 之前建立连接，
            # 否则会出现 ConnectionNotExistException: should create connection first.
            # （模块导入时就会执行此处，早于 FastAPI lifespan 中的 milvus_manager.connect）
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建 LangChain Milvus VectorStore
            # 使用 biz collection，字段映射：
            #   text_field -> content
            #   vector_field -> vector
            #   metadata_field -> metadata
            #   document_id  -> 独立字段（Milvus schema 中定义）
            # langchain_milvus 会自动处理非保留字段 document_id
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,  # 使用自定义 id
                drop_old=False,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    # ------------------------------------------------------------------
    # 新接口：带 document_id 的入库（推荐）
    # ------------------------------------------------------------------
    def add_documents_with_doc_id(
        self,
        documents: List[Document],
        document_id: str,
    ) -> List[str]:
        """将切分后的 chunk 入库，关联到 document_id

        每个 chunk 的主键 ID 格式为：{document_id}_{chunk_index:04d}
        每个 chunk 的 metadata 中会注入 document_id 字段。

        Args:
            documents: 切分后的 chunk 列表（Document 对象）
            document_id: 文档唯一 ID，来自 document_registry

        Returns:
            List[str]: 实际入库的 chunk ID 列表
        """
        if not documents:
            logger.warning(f"add_documents_with_doc_id: 空文档列表，跳过 ({document_id})")
            return []

        if self.vector_store is None:
            raise RuntimeError("VectorStore 未初始化")

        start_time = time.time()
        n = len(documents)

        # 1. 为每个 chunk 生成关联 ID，并在 metadata 中注入 document_id
        ids: list[str] = []
        enriched_docs: list[Document] = []
        for idx, doc in enumerate(documents):
            chunk_id = _format_chunk_id(document_id, idx)
            ids.append(chunk_id)

            # 克隆 metadata 并注入 document_id（避免修改原对象）
            new_metadata = dict(doc.metadata) if doc.metadata else {}
            new_metadata["document_id"] = document_id
            new_metadata["chunk_index"] = idx

            enriched_docs.append(
                Document(page_content=doc.page_content, metadata=new_metadata)
            )

        # 2. 调用底层 Milvus VectorStore 批量入库
        # langchain_milvus 会根据 schema 自动把 metadata 存到 JSON 字段，
        # 但 document_id 需要作为独立字段也能支持。这里我们确保两条路径都有值：
        #   a) metadata["document_id"]            → JSON 字段中
        #   b) Milvus 的独立 document_id 字段      → 若 langchain 支持字段透传
        try:
            # 使用底层 pymilvus collection 直接插入（字段映射更精确）
            collection = milvus_manager.get_collection()

            # 先拿到 embeddings
            texts = [doc.page_content for doc in enriched_docs]
            embeddings = vector_embedding_service.embed_documents(texts)

            # 准备插入数据（按 schema 顺序：id, document_id, vector, content, metadata）
            insert_data = [
                ids,  # id (primary)
                [document_id] * n,  # document_id
                embeddings,  # vector
                texts,  # content
                [doc.metadata for doc in enriched_docs],  # metadata (JSON)
            ]

            result = collection.insert(insert_data)
            inserted_count = getattr(result, "insert_count", n)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {inserted_count} 个 chunk 到 VectorStore 完成 "
                f"(document_id={document_id}), "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed / max(inserted_count, 1):.2f}秒/个"
            )
            return ids

        except TypeError as e:
            # 兜底：如果底层 pymilvus schema 字段顺序与预想不同，
            # 回退到 langchain 标准 API（metadata 中仍带有 document_id）
            logger.warning(
                f"底层 pymilvus 直接插入失败 ({e}), 回退到 langchain add_documents"
            )
            result_ids = self.vector_store.add_documents(enriched_docs, ids=ids)
            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(result_ids)} 个 chunk 到 VectorStore 完成 "
                f"(document_id={document_id}), 耗时: {elapsed:.2f}秒"
            )
            return result_ids

        except Exception as e:
            logger.error(
                f"添加文档失败 (document_id={document_id}): {e}"
            )
            raise

    # ------------------------------------------------------------------
    # 旧接口：保留兼容
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（兼容旧代码，自动生成 document_id）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        if not documents:
            return []

        # 从第一个文档的 metadata 中取 document_id（如果有）
        first_meta = documents[0].metadata or {}
        document_id = first_meta.get("document_id")

        if document_id:
            # 已经有 document_id：走新路径
            return self.add_documents_with_doc_id(documents, document_id)

        # 没有 document_id：生成一个临时的，保证向后兼容
        import uuid

        temp_doc_id = "legacy_" + uuid.uuid4().hex[:12]
        logger.info(
            f"add_documents (legacy): 未提供 document_id, "
            f"自动生成: {temp_doc_id}"
        )
        return self.add_documents_with_doc_id(documents, temp_doc_id)

    # ------------------------------------------------------------------
    # 删除接口
    # ------------------------------------------------------------------
    def delete_by_document_id(self, document_id: str) -> int:
        """
        删除指定 document_id 的所有 chunk（推荐方式）

        通过 Milvus 的 document_id 字段筛选，效率高且可靠。

        Args:
            document_id: 文档唯一 ID

        Returns:
            int: 删除的 chunk 数量
        """
        try:
            collection = milvus_manager.get_collection()

            # 通过独立 document_id 字段筛选（首选，索引友好）
            expr = f'document_id == "{document_id}"'
            logger.debug(f"执行删除表达式: {expr}")

            result = collection.delete(expr)
            deleted_count = getattr(result, "delete_count", 0)

            # 兜底：如果 document_id 字段不可用（旧数据），
            # 尝试通过 metadata JSON 字段删除
            if deleted_count == 0:
                expr_fallback = f'metadata["document_id"] == "{document_id}"'
                logger.debug(
                    f"document_id 字段未删除到数据，尝试 fallback: {expr_fallback}"
                )
                try:
                    result2 = collection.delete(expr_fallback)
                    deleted_count = getattr(result2, "delete_count", 0)
                except Exception as fallback_err:
                    logger.debug(f"fallback 删除也失败: {fallback_err}")

            logger.info(
                f"删除文档 chunk: document_id={document_id}, "
                f"删除数量={deleted_count}"
            )
            return deleted_count

        except Exception as e:
            logger.warning(
                f"删除文档 chunk 失败 (document_id={document_id}): {e}"
            )
            return 0

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件路径的所有文档（兼容旧代码）

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            collection = milvus_manager.get_collection()

            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            # _source 是文档的来源文件路径
            expr = f'metadata["_source"] == "{file_path}"'

            result = collection.delete(expr)
            deleted_count = getattr(result, "delete_count", 0)

            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        if self.vector_store is None:
            raise RuntimeError("VectorStore 未初始化")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        if self.vector_store is None:
            return []

        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()