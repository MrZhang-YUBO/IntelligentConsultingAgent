"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen3.7-plus"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # Kafka 配置（事件驱动 - 文档变更）
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_document_changes: str = "document_changes"
    kafka_consumer_group_id: str = "document-indexer-group"
    kafka_auto_offset_reset: str = "latest"  # latest | earliest
    kafka_max_retries: int = 3
    kafka_retry_backoff_ms: int = 1000
    kafka_session_timeout_ms: int = 30000
    kafka_heartbeat_interval_ms: int = 10000

    # 文档注册表配置
    document_registry_path: str = "./data/document_registry.json"
    document_id_prefix: str = "doc_"
    document_id_length: int = 8  # 随机部分长度

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen3.7-plus"  # 使用快速响应模型，不带扩展思考

    # ── 混合检索配置 ──────────────────────────
    rag_vector_top_k: int = 10          # 向量检索召回数量
    rag_bm25_top_k: int = 10            # BM25 检索召回数量
    rag_hybrid_top_k: int = 20          # 混合融合后候选数量
    rag_vector_weight: float = 0.6      # 向量检索权重
    rag_bm25_weight: float = 0.4        # BM25 权重
    rag_bm25_corpus_size: int = 200     # BM25 扫描文档池大小（从 Milvus 取多少文档构建语料）

    # ── 重排配置 ─────────────────────────────
    rag_enable_rerank: bool = True      # 是否启用重排
    rag_rerank_top_k: int = 10          # 重排后最终返回数量
    rag_rerank_model: str = "rerank-v1" # 重排模型 (DashScope)

    # ── 网络检索配置 ─────────────────────────
    web_search_enabled: bool = True                     # 总开关（默认关闭，需配置 TAVILY_API_KEY 后开启）
    tavily_api_key: str = ""                             # Tavily API Key
    web_search_max_results: int = 3                      # 每次搜索最大结果数
    web_search_search_depth: str = "basic"               # "basic" 或 "advanced"（advanced 更慢但更全面）
    web_search_auto_trigger_enabled: bool = True         # 是否启用自动触发
    web_search_auto_trigger_threshold: float = 0.3       # 自动触发阈值（混合检索最高分 < 此值时触发网络搜索）
    web_search_summarization_enabled: bool = True        # 是否压缩网络检索结果
    web_search_summarization_model: str = "qwen3.6-flash"   # 摘要压缩模型（用快速模型降低延迟）
    web_search_max_content_length: int = 2000            # 每条网络结果最大字符数（截断前）
    web_search_safety_enabled: bool = True               # 是否启用安全过滤
    web_search_safety_keyword_blacklist: str = ""         # 关键词黑名单（逗号分隔，如 "赌博,色情,诈骗"）
    web_search_safety_blocked_domains: str = ""           # 屏蔽域名（逗号分隔，如 "malicious.com,phishing.net"）
    web_search_llm_review_enabled: bool = True           # 是否启用 LLM 安全审核
    web_search_llm_review_model: str = "qwen3.5-flash"      # LLM 审核模型

    # ── Agent 安全（五层安全防护）─────────────────────
    agent_safety_enabled: bool = True                   # 总开关
    agent_safety_llm_check: bool = False                 # 是否启用 LLM 语义级审核（规则引擎始终运行）
    agent_safety_llm_model: str = "qwen-turbo"         # 安全审核用的轻量模型
    agent_safety_keyword_blacklist: str = ""             # 关键词黑名单（逗号分隔；空则使用内置默认 40+ 关键词）
    agent_safety_blocked_urls: str = ""                    # 屏蔽域名（逗号分隔）
    agent_safety_tool_check: bool = True                  # 是否启用工具调用安全检查（工具白名单 + 参数检查）
    agent_safety_tool_whitelist: str = "retrieve_knowledge,web_search,get_current_time,query_prometheus_alerts" # 工具白名单（逗号分隔；空表示不限制）

    # 对话记忆压缩（总结）配置
    summary_trigger_rounds: int = 5  # 每累计 N 轮对话触发一次总结
    summary_model: str = "qwen3.6-flash"  # 用更快更便宜的模型做总结

    # ── 多轮意图识别配置 ─────────────────────────
    intent_recognition_enabled: bool = True        # 是否启用意图识别（关闭则每轮不额外调用 LLM）
    intent_model: str = "qwen-turbo"               # 意图识别模型（用快速模型降低延迟）
    intent_recent_message_window: int = 3          # 传入识别器的最近消息条数（长上下文有界窗口）
    intent_history_size: int = 10                  # 每会话保留的意图轨迹条数（IntentTracker maxlen）
    intent_confidence_threshold: float = 0.5       # 置信度阈值（低于此值的意图可标记为不确定，留作扩展）

    # ── 意图驱动自动分解编排配置 ───────────────────────────
    intent_orchestration_enabled: bool = False      # 是否在多意图/复杂意图时触发编排
    intent_orchestration_model: str = "qwen3.6-flash" # 汇总回答用的轻量模型
    intent_orchestration_min_sub_intents: int = 2  # 至少 N 个子意图才触发编排（避免小题大做）

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置（transport: stdio | sse | streamable-http）
    # 腾讯云托管 MCP 的 URL 通常含 /sse/，需使用 sse；本地 FastMCP 使用 streamable-http
    # mcp_cls_transport: str = "streamable-http"
    # mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_cls_transport: str = "sse"
    mcp_cls_url: str = "http://localhost:3000/sse"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # Prometheus (AIOps 查询外部 Prometheus Server)
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    # 业务指标（本服务自己暴露 /metrics）
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()