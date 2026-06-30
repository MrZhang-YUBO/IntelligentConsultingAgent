# Intelligent Consulting Agent

> 企业级智能对话和运维助手，支持 RAG 知识库问答、多轮意图识别、自动分解编排、Agent 安全防护，以及大模型微调全链路。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://www.langchain.com/)
[![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-blue.svg)](https://prometheus.io/)
[![LLaMA-Factory](https://img.shields.io/badge/Fine--tune-LLaMA--Factory-purple.svg)](https://github.com/hiyouga/LLaMA-Factory)

## ✨ 核心特性

| 模块 | 功能亮点 |
|------|---------|
| 🤖 **智能对话** | LangChain 多轮对话 + 流式输出；对话记忆自动总结压缩，避免上下文爆炸 |
| 🧠 **多轮意图识别** | 每轮结构化意图识别（主/副意图、复杂意图、指代消解）；有界上下文 + 内存轨迹 |
| 🔀 **意图驱动编排** | 多意图/复杂意图问题自动分解为子任务并行检索，LLM 汇总生成最终回答 |
| 📚 **RAG 问答** | 向量相似度 + BM25 词法双通道混合检索；DashScope Rerank 二次过滤；Top-N 精确召回 |
| 🌐 **网络检索** | 知识库无答案时自动触发网络检索（Tavily）；双层安全过滤（规则引擎 + LLM 审核）；摘要压缩 |
| 📄 **多格式文件上传** | 支持 **PDF / Word (.docx) / Markdown / TXT** 四种格式；自动分块、表格识别、代码块识别 |
| 🔄 **知识库动态更新** | 事件驱动（Kafka）+ SHA-256 内容 Hash 检测；先删后增，避免旧版本 chunk 残留 |
| 🔧 **AIOps 诊断** | Plan-Execute-Replan 自动故障诊断和根因分析；Prometheus 告警查询 |
| 🔒 **五层 Agent 安全** | 输入安全 / 文档投毒防护 / 工具白名单 + HITL / 记忆投毒防护 / 输出安全 |
| 📊 **业务指标 & 可观测性** | Prometheus `counter/gauge/histogram` 6 指标集；QPS / P50/P95/P99 延迟 / 错误率 |
| 🎓 **大模型微调** | Qwen2.5-7B-Instruct 基座；SFT (LoRA) → DPO (LoRA) 两阶段训练；数据准备到推断全脚本 |

## 🛠️ 技术栈

### 应用层
- **框架**: FastAPI + LangChain + LangGraph
- **LLM**: 阿里云 DashScope (Qwen2.5 系列)
- **向量库**: Milvus
- **工具协议**: MCP (Model Context Protocol)

### 检索与数据
- **混合检索**: 向量相似度（Milvus）+ BM25 词法（rank-bm25）
- **重排**: DashScope Rerank API
- **网络检索**: Tavily API
- **文档处理**: PyMuPDF (PDF) / python-docx (Word) / LangChain (MD/TXT)

### 事件驱动与指标
- **消息队列**: Kafka (confluent-kafka + kafka-python，内存 fallback)
- **文档注册表**: JSON 文件持久化，doc_id ↔ file_path ↔ content_hash 映射
- **指标**: prometheus_client（6 指标集，`/metrics` 端点暴露）

### 安全
- **规则引擎**: 40+ 关键词黑名单、XSS/SQL 注入/prompt injection 正则检测
- **LLM 审核**: qwen-turbo 语义级柔性过滤（可选）
- **Fail-open 策略**: 任何安全检查异常默认放行，仅记录审计日志

### 大模型微调
- **基座**: Qwen2.5-7B-Instruct
- **框架**: LLaMA-Factory
- **微调方式**: SFT (LoRA) → DPO (LoRA)
- **训练设备**: 单卡 NVIDIA GPU (CUDA)

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 阿里云 DashScope API Key ([获取地址](https://dashscope.aliyun.com/))
- Docker Desktop (用于启动 Milvus)

### 安装和启动

#### Linux/macOS 环境

```bash
# 1. 克隆项目
git clone <repository_url>
cd IntelligentConsultingAgent

# 2. 安装依赖（推荐使用 uv）
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. 编辑配置文件
# 首次使用需要编辑 .env 文件，填入你的 DASHSCOPE_API_KEY
vim .env

# 4. 启动 Milvus 向量数据库
docker compose -f vector-database.yml up -d

# 5. 启动主服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 6. 上传文档（可选，示例：上传 aiops-docs 目录下文档）
# 上传接口会自动触发异步索引（事件驱动）
curl -X POST -F "file=@aiops-docs/some_doc.md" http://localhost:9900/api/upload
```

#### Windows 环境（PowerShell/CMD）

```powershell
# 1. 克隆项目
git clone <repository_url>
cd IntelligentConsultingAgent

# 2. 创建虚拟环境并安装依赖
# 方式 1: 使用 uv（推荐，更快）
pip install uv
uv venv
.venv\Scripts\activate
uv pip install -e .

# 方式 2: 使用 pip
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 编辑配置文件
# 使用记事本或其他编辑器打开 .env 文件，填入你的 DASHSCOPE_API_KEY
notepad .env

# 4. 启动 Docker Desktop
# 确保 Docker Desktop 已安装并正在运行

# 5. 启动 Milvus 向量数据库（Docker Compose）
docker compose -f vector-database.yml up -d

# 6. 启动主服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 7. 上传文档（可选）
curl -X POST -F "file=@aiops-docs\some_doc.md" http://localhost:9900/api/upload
```

### 访问服务
- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs
- **监控指标**: http://localhost:9900/metrics (Prometheus 抓取端点)

---

## ⚙️ 配置说明

通过 `.env` 文件配置（全部为可选，有合理默认值）：

### LLM 基础配置
```bash
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxx        # 必填，阿里云 API Key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max                     # 主对话模型
```

### Milvus 配置
```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 混合检索与重排配置
```bash
RAG_VECTOR_TOP_K=10          # 向量检索召回数量
RAG_BM25_TOP_K=10            # BM25 检索召回数量
RAG_HYBRID_TOP_K=20          # 混合融合后候选数量
RAG_VECTOR_WEIGHT=0.6        # 向量权重
RAG_BM25_WEIGHT=0.4          # BM25 权重
RAG_BM25_CORPUS_SIZE=200     # BM25 语料池大小
RAG_ENABLE_RERANK=true       # 是否启用重排
RAG_RERANK_TOP_K=10          # 重排后最终返回数量
RAG_RERANK_MODEL=rerank-v1   # 重排模型
```

### 网络检索配置
```bash
WEB_SEARCH_ENABLED=false     # 网络检索总开关
TAVILY_API_KEY=tvly-xxxxxxxx # Tavily API Key（若启用）
WEB_SEARCH_MAX_RESULTS=5     # 每次搜索最大结果数
WEB_SEARCH_SEARCH_DEPTH=basic  # basic / advanced
WEB_SEARCH_AUTO_TRIGGER_ENABLED=true    # 是否自动触发（知识库无答案时）
WEB_SEARCH_AUTO_TRIGGER_THRESHOLD=0.3   # 自动触发阈值（最高分低于此值）
WEB_SEARCH_SUMMARIZATION_ENABLED=true    # 是否压缩网络结果
WEB_SEARCH_SUMMARIZATION_MODEL=qwen-turbo
WEB_SEARCH_MAX_CONTENT_LENGTH=2000
WEB_SEARCH_SAFETY_ENABLED=true            # 是否启用安全过滤
WEB_SEARCH_SAFETY_KEYWORD_BLACKLIST=      # 自定义关键词黑名单（逗号分隔）
WEB_SEARCH_SAFETY_BLOCKED_DOMAINS=        # 屏蔽域名（逗号分隔）
WEB_SEARCH_LLM_REVIEW_ENABLED=true        # 是否启用 LLM 语义级审核
WEB_SEARCH_LLM_REVIEW_MODEL=qwen-turbo
```

### 对话记忆压缩配置
```bash
SUMMARY_TRIGGER_ROUNDS=5      # 每累计 N 轮对话触发一次总结
SUMMARY_MODEL=qwen-turbo       # 总结用轻量模型
```

### 多轮意图识别配置
```bash
INTENT_RECOGNITION_ENABLED=true   # 意图识别总开关
INTENT_MODEL=qwen-turbo            # 识别用轻量模型
INTENT_RECENT_MESSAGE_WINDOW=6     # 传入识别器的最近消息条数
INTENT_HISTORY_SIZE=10             # 每会话保留意图轨迹条数
INTENT_CONFIDENCE_THRESHOLD=0.5    # 置信度阈值
```

### Agent 安全配置（五层防护）
```bash
AGENT_SAFETY_ENABLED=true           # 安全总开关
AGENT_SAFETY_LLM_CHECK=true         # 是否启用 LLM 语义级审核（规则引擎始终运行）
AGENT_SAFETY_LLM_MODEL=qwen-turbo   # 安全审核用模型
AGENT_SAFETY_KEYWORD_BLACKLIST=     # 自定义关键词黑名单（逗号分隔；空用内置 40+ 默认）
AGENT_SAFETY_BLOCKED_URLS=          # 屏蔽域名（逗号分隔）
AGENT_SAFETY_TOOL_CHECK=true        # 是否启用工具调用安全检查
AGENT_SAFETY_TOOL_WHITELIST=retrieve_knowledge,web_search,get_current_time,query_prometheus_alerts
```

### 指标配置
```bash
METRICS_ENABLED=true                # 是否启用指标中间件
METRICS_PATH=/metrics               # 指标暴露路径
```

### Kafka 事件驱动配置（知识库动态更新）
```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_DOCUMENT_CHANGES=document_changes
KAFKA_CONSUMER_GROUP_ID=document-indexer-group
KAFKA_AUTO_OFFSET_RESET=latest
KAFKA_MAX_RETRIES=3
KAFKA_RETRY_BACKOFF_MS=1000
KAFKA_SESSION_TIMEOUT_MS=30000
KAFKA_HEARTBEAT_INTERVAL_MS=10000

DOCUMENT_REGISTRY_PATH=./data/document_registry.json
DOCUMENT_ID_PREFIX=doc_
DOCUMENT_ID_LENGTH=8
```

### 意图编排配置
```bash
INTENT_ORCHESTRATION_ENABLED=true    # 编排总开关
INTENT_ORCHESTRATION_MODEL=qwen-turbo # 汇总用轻量模型
INTENT_ORCHESTRATION_MIN_SUB_INTENTS=2  # 至少 N 个子意图才触发编排
```

---

## 📡 API 接口

### 核心接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 普通对话 | POST | `/api/chat` | 一次性返回；响应含 `intent` 字段（本轮意图识别结果） |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出；事件类型：`intent` / `content` / `orchestration_start` / `orchestration_step` / `orchestration_summary` / `safety_blocked` / `complete` / `error` |
| AIOps 诊断 | POST | `/api/aiops` | 自动故障诊断（流式） |
| 文件上传 | POST | `/api/upload` | 上传并异步索引文档（事件驱动，支持 PDF/Word/MD/TXT） |
| 目录索引 | POST | `/api/index_directory` | 批量索引目录下所有支持文件 |
| 文档列表 | GET | `/api/documents` | 列出所有已索引文档（从注册表读取） |
| 删除文档 | DELETE | `/api/document/{doc_id}` | 按 doc_id 删除（删 Milvus chunk + 注册表标记） |
| 会话历史 | GET | `/api/chat/session/{session_id}` | 读取会话历史 + 意图轨迹 |
| 清空会话 | DELETE | `/api/chat/session/{session_id}` | 清空会话历史 + 意图轨迹 |
| 健康检查 | GET | `/api/health` | 服务状态检查 |
| 监控指标 | GET | `/metrics` | Prometheus 指标端点（纯文本格式） |

### SSE 事件协议（`/api/chat_stream`）

前端订阅 SSE 流时，会按以下顺序收到事件：

| 事件类型 | 时机 | 说明 |
|---------|------|------|
| `intent` | 对话开始时 | 本轮意图识别结果（主/副意图、实体、置信度等） |
| `safety_blocked` | 输入不安全时 | 用户输入被安全系统拦截（替代后续所有事件） |
| `orchestration_start` | 触发编排时 | 通知"正在并行检索 N 个子问题" |
| `orchestration_step` | 子任务状态变化 | 子任务进度（v2 精简版：仅状态更新；v1/v2 兼容） |
| `orchestration_summary` | 编排完成 | 汇总模式 + 成功数/总数 + 总耗时 |
| `content` | LLM 生成中 | 流式 token；**不带** `subtask_index` 表示主回答 |
| `complete` | 对话完成 | 对话结束标记 |
| `error` | 发生错误 | 错误信息 |

### 使用示例

```bash
# 普通对话
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"介绍一下产品A的核心功能，并对比B和C的价格差异"}' \
  --no-buffer

# 上传 PDF 文档（自动异步索引）
curl -X POST -F "file=@legal_document.pdf" http://localhost:9900/api/upload

# 上传 Word 文档
curl -X POST -F "file=@ops_manual.docx" http://localhost:9900/api/upload

# 启用网络检索（前端开关）
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"2026年最新的AI立法动向是什么","EnableWebSearch":true}'

# 查看 Prometheus 指标
curl http://localhost:9900/metrics

# 查看会话历史 + 意图轨迹
curl http://localhost:9900/api/chat/session/session-123
```

---

## 📁 项目结构

```
IntelligentConsultingAgent/
├── app/                                    # 应用核心
│   ├── __init__.py                         # 包初始化（自动加载日志配置）
│   ├── main.py                             # FastAPI 应用入口（lifespan 启停 Kafka Consumer + Metrics）
│   ├── config.py                           # 配置管理（LLM/Milvus/RAG/网络检索/安全/意图/指标/Kafka）
│   ├── api/                                # API 路由层
│   │   ├── __init__.py
│   │   ├── chat.py                         # 对话接口（RAG 聊天 + 流式 + 会话历史）
│   │   ├── aiops.py                        # AIOps 接口（故障诊断）
│   │   ├── file.py                         # 文件管理（多格式上传/索引/删除）
│   │   └── health.py                       # 健康检查
│   ├── services/                           # 业务服务层
│   │   ├── __init__.py
│   │   ├── rag_agent_service.py            # RAG Agent（意图识别+安全+编排+对话）核心
│   │   ├── aiops_service.py                # AIOps 服务（计划-执行-重规划）
│   │   ├── vector_store_manager.py         # 向量存储管理器（doc_id 关联 + 批量删除）
│   │   ├── vector_embedding_service.py     # 向量 embedding 服务
│   │   ├── vector_index_service.py         # 向量索引服务（事件驱动 + Hash 检测 + 先删后增）
│   │   ├── document_splitter_service.py    # 文档分割服务（调用处理器工厂）
│   │   ├── hybrid_search_service.py        # 混合检索服务（向量 + BM25 加权融合）
│   │   ├── rerank_service.py               # 重排服务（DashScope Rerank）
│   │   ├── bm25_service.py                 # BM25 词法检索服务
│   │   ├── web_search_service.py           # 网络检索服务（Tavily + 安全过滤 + 摘要）
│   │   └── content_safety_service.py       # 内容安全服务（五层安全防护）
│   ├── processors/                         # 文档格式处理器模块（工厂模式）
│   │   ├── __init__.py                     # 工厂函数（PROCESSOR_MAP 映射 + get_supported_extensions）
│   │   ├── base_processor.py               # 抽象基类（通用工具方法）
│   │   ├── text_processor.py               # TXT 处理器
│   │   ├── markdown_processor.py           # Markdown 处理器（按标题分割）
│   │   ├── pdf_processor.py                # PDF 处理器（PyMuPDF，表格/图片/代码块识别）
│   │   └── word_processor.py               # Word 处理器（python-docx，标题层级/表格）
│   ├── agent/                              # Agent 模块
│   │   ├── __init__.py
│   │   ├── mcp_client.py                   # MCP 客户端（工具调用）
│   │   ├── orchestrator.py                 # 意图驱动编排器（v2：子任务仅工具调用 + LLM 汇总流式）
│   │   ├── intent_agent.py                 # 多轮意图识别（结构化输出 + 意图轨迹追踪）
│   │   ├── summary_agent.py                # 对话记忆总结（LLM 总结压缩 + 有界上下文）
│   │   └── aiops/                          # AIOps 核心逻辑
│   │       ├── __init__.py
│   │       ├── planner.py                  # 计划制定器
│   │       ├── executor.py                 # 步骤执行器
│   │       ├── replanner.py                # 重规划器
│   │       ├── state.py                    # 状态定义
│   │       └── utils.py                    # 工具函数
│   ├── core/                               # 核心组件
│   │   ├── __init__.py
│   │   ├── llm_factory.py                  # LLM 工厂（模型管理）
│   │   ├── milvus_client.py                # Milvus 客户端（schema 含 document_id 字段）
│   │   ├── kafka_client.py                 # Kafka 客户端（生产者/消费者 + 内存 fallback）
│   │   ├── document_registry.py            # 文档注册表（doc_id ↔ file_path ↔ content_hash 映射）
│   │   └── metrics.py                      # Prometheus 指标定义 + 中间件 + @timed_metric
│   ├── utils/                              # 工具类
│   │   ├── __init__.py
│   │   ├── logger.py                       # 日志配置（Loguru）
│   │   └── hash_utils.py                   # SHA-256 文件内容哈希（用于变更检测）
│   ├── models/                             # 数据模型层
│   │   ├── __init__.py
│   │   ├── aiops.py                        # AIOps 模型
│   │   ├── document.py                     # 文档模型（DocumentRecord/Event/DocumentStatus）
│   │   ├── request.py                      # 请求模型（含 enable_web_search）
│   │   └── response.py                     # 响应模型（SessionInfoResponse 含 intents）
│   ├── tools/                              # Agent 工具集
│   │   ├── __init__.py                     # 工具注册（含 web_search）
│   │   ├── knowledge_tool.py               # 知识库查询工具（混合检索 + 重排 + 自动网络检索）
│   │   └── time_tool.py                    # 时间工具
│   └── events/                             # 事件处理模块
│       ├── __init__.py
│       └── document_event_handler.py       # Kafka Consumer 生命周期管理
├── static/                                 # Web 前端（纯静态）
│   ├── index.html                          # 主页面（工具菜单：网络搜索开关）
│   ├── app.js                              # 前端逻辑（意图卡片 + 编排进度 + 安全拦截提示）
│   └── styles.css                          # 样式表（意图卡片/编排面板样式）
├── mcp_servers/                            # MCP 服务器
│   ├── cls_server.py                       # CLS 日志查询服务
│   ├── monitor_server.py                   # 监控数据服务
│   └── README.md                           # MCP 服务说明
├── aiops-docs/                             # 运维知识库（Markdown 文档）
├── logs/                                   # 日志目录（Loguru 自动创建）
├── uploads/                                # 上传文件临时目录
├── data/                                   # 运行时数据（document_registry.json 自动创建）
├── volumes/                                # Milvus 数据持久化目录
├── LLM_DataSet_Train/                      # 大模型微调数据集与脚本
│   ├── llm/                                # 微调 Python 脚本包
│   │   ├── __init__.py
│   │   ├── prepare_lawzhidao_sft.py        # 数据准备：lawzhidao CSV → SFT JSON
│   │   ├── generate_self_instruct_qwen_api.py  # Self-Instruct 数据扩充
│   │   ├── build_sft_dataset.py            # 三源融合 → train/val 切分
│   │   ├── generate_dpo_candidates_qwen_api.py  # DPO A/B 候选生成
│   │   ├── judge_dpo_pairs_qwen_api.py     # DPO 裁判标注
│   │   ├── infer_lora.py                   # SFT LoRA 推断
│   │   └── infer_dpo_lora.py               # DPO LoRA 推断
│   └── dataset/                            # 训练数据集目录
│       ├── sft/                            # SFT 数据与训练参数
│       │   └── 训练参数.txt                # SFT LLaMA-Factory 训练命令
│       └── dpo/                            # DPO 数据与训练参数
│           └── dpo参数.txt                 # DPO LLaMA-Factory 训练命令
├── 更新日志-*.md                           # 各功能模块更新日志（共 10 份）
├── .env                                    # 环境变量配置（需手动创建）
├── vector-database.yml                     # Milvus Docker Compose 配置
├── pyproject.toml                          # 项目配置（依赖、元数据）
├── uv.lock                                 # uv 依赖锁定文件
└── README.md                               # 项目说明（本文档）
```

---

## 🔒 五层 Agent 安全防护

覆盖从用户提问到最终回答的全链路：

| 层级 | 防护点 | 触发时机 | 策略 |
|------|--------|----------|------|
| **层 1** | 输入安全（直接 Prompt / 间接 Prompt） | 意图识别 / Agent 执行前 | 规则引擎（关键词黑名单 + XSS/注入/prompt injection 检测）+ 可选 LLM 语义级审核 |
| **层 2** | 文档投毒防护（RAG 知识库写入 / 网络检索结果） | 文档入库前、工具输出返回 Agent 前 | 规则引擎 + 可选 LLM 语义级审核，检测疑似被投毒文档 |
| **层 3** | 工具调用安全（工具白名单 + 参数清洗 + HITL） | 每个工具真正执行之前 | 工具白名单拦截 + 参数规则检查 + 高风险工具 HITL 标记 |
| **层 4** | 记忆投毒防护（对话历史） | 从 checkpointer 读取历史消息后、总结压缩前 | 规则引擎逐条检测，疑似被投毒消息内容被清空 |
| **层 5** | 输出安全（最终回答） | 最终回答发送给前端前 | 规则引擎 + 可选 LLM 语义级审核，不安全回答替换为兜底文本 |

**核心设计原则**:
- 双层防御：规则引擎（同步、快速、硬安全底线）+ LLM 审核（异步、可选、语义级柔性过滤）
- Fail-open（默认放行）：任何检查环节异常默认放行，不影响正常对话体验，仅记录审计日志
- 全局单例 `content_safety_service`：避免重复初始化，所有安全行为可通过 `config.py` 灵活开关
- 审计日志：所有安全事件通过 `[安全-*]` 前缀 `logger.warning` 记录，便于事后审计

---

## 🎯 对话架构概览（一轮对话的完整数据流）

```
用户提问 (query + session_id)
    │
    ├─ ▶ 【层 1 输入安全】 content_safety_service.check_user_input()
    │        → 不安全：返回 safety_blocked 事件 / 兜底文本，结束
    │
    ├─ ▶ 【对话记忆总结】 await _summarize_and_update(session_id)
    │        → 非系统消息 ≥ 10 条时，LLM 压缩为总结消息 + 最近 N 条保留
    │        → 层 4 记忆投毒防护：读取历史时清理可疑消息
    │
    ├─ ▶ 【多轮意图识别】 await intent_agent.recognize()
    │        → 输入：最近 N 条消息 + 已有总结 + 最近 2 条意图（有界上下文）
    │        → 输出：结构化 IntentRecognitionResult（主/副意图/实体/置信度）
    │        → 记录轨迹到 intent_tracker
    │        → 流式：先 yield intent 事件给前端
    │
    ├─ ▶ 【系统提示词重建】 注入本轮意图 context（临时，不污染历史）
    │
    └─ ▶ 判断是否触发意图编排：
         ├─ 是（多意图/复杂意图）：进入【编排路径】
         │    │
         │    ├─▶ 为每个子意图构造子问题
         │    ├─▶ 子任务并行执行（仅调用工具：知识库检索/网络检索/时间/告警查询）
         │    │       ├─ 层 3 工具白名单检查（orchestrator.check_tool_call）
         │    │       └─ 层 2 文档投毒检查（工具输出清洗）
         │    ├─▶ LLM (qwen-turbo) 汇总并流式生成最终回答
         │    └─▶ 层 5 输出安全检查（check_output → 用 sanitized_answer）
         │
         └─ 否（单意图简单问题）：进入【原 Agent 路径】
              ├─▶ Agent 自主调用工具（retrieve_knowledge 含混合检索+重排+自动网络检索）
              └─▶ 层 5 输出安全检查（check_output → 用 sanitized_answer）

    ↓ 最终：SSE 流式输出给前端（前端显示意图卡片 + 答案）
```

---

## 📊 业务指标 & 可观测性

通过 Prometheus `counter/gauge/histogram` 收集 6 个核心指标：

| 指标名 | 类型 | Labels | 说明 |
|--------|------|--------|------|
| `http_request_count` | Counter | `method, path, status_code` | 累计请求数（按路径+状态码拆分） |
| `http_request_latency_seconds` | Histogram | `method, path` | 请求耗时（秒，默认分桶 0.005~10s） |
| `http_request_in_progress` | Gauge | `method, path` | 当前正在处理的请求数（并发数） |
| `http_error_count` | Counter | `method, path, status_code` | 4xx/5xx 错误数 |
| `service_request_latency_seconds` | Histogram | `service, method` | 业务层函数耗时（如 `rag.query`） |
| `service_error_count` | Counter | `service, method` | 业务层函数错误数 |

### PromQL 查询示例

| 需求 | PromQL |
|------|--------|
| QPS（近 5 分钟速率） | `rate(http_request_count[5m])` |
| 按路径分的 QPS | `sum by (path) (rate(http_request_count[5m]))` |
| P95 延迟 | `histogram_quantile(0.95, rate(http_request_latency_seconds[5m]))` |
| P99 延迟 | `histogram_quantile(0.99, rate(http_request_latency_seconds[5m]))` |
| 错误率 | `sum(rate(http_error_count[5m])) / sum(rate(http_request_count[5m]))` |
| 并发请求数 | `http_request_in_progress` |
| RAG query 平均耗时 | `rate(service_request_latency_seconds_sum{service="rag",method="query"}[5m]) / rate(service_request_latency_seconds_count{service="rag",method="query"}[5m])` |

### Prometheus 抓取配置

```yaml
scrape_configs:
  - job_name: 'intelligent-consulting-agent'
    static_configs:
      - targets: ['localhost:9900']
    scrape_interval: 15s
    metrics_path: /metrics
```

---

## 🔄 向量知识库动态更新

事件驱动 + 内容 Hash 检测 + 先删后增 的方案：

```
HTTP /api/upload
    │
    ▼
保存到本地 → compute_file_hash() (SHA-256)
    │
    ▼
document_registry.is_content_changed(path, hash)
    ├─ 新文件 (record=None)   → DOCUMENT_CREATED 事件
    ├─ hash 相同（内容未变）   → 跳过，不重复索引
    └─ hash 不同（内容变更）   → DOCUMENT_UPDATED 事件
    │
    ▼
Kafka Topic: document_changes  (Kafka 不可用时自动降级为内存队列)
    │
    ▼
Kafka Consumer（后台线程，FastAPI lifespan 中启停）
    ├─ DOCUMENT_CREATED: 切分 → 入库（带 document_id）
    └─ DOCUMENT_UPDATED: 先删旧 chunk（按 doc_id 批量删除）→ 重新切分 → 重新入库
    │
    ▼
Milvus collection: biz （schema 含 document_id 独立字段，便于按文档批量删除）
    │
    ▼
document_registry.update_hash(new_hash, chunk_count) + update_status(ACTIVE)
    │
    ▼
持久化到 data/document_registry.json（原子写入：temp file + replace）
```

**关键优势**:
- **精准更新**: 文档修改后，旧版本 chunk 被完整清理，只保留最新版本
- **事件驱动**: HTTP 层快速返回，切分/入库由后台异步执行，不阻塞用户
- **Hash 检测**: 内容未变则跳过，避免重复计算浪费算力
- **可追踪**: 每个 chunk 与 document_id 关联，删除/更新有据可查

---

## 🎓 大模型微调（法律领域）

### 基座与框架
- **基座模型**: Qwen2.5-7B-Instruct
- **微调框架**: LLaMA-Factory
- **训练方式**: SFT (LoRA) → DPO (LoRA) 两阶段
- **训练设备**: 单卡 NVIDIA GPU (CUDA)

### 流水线总图

```
原始数据 ──► SFT 数据构建 ──► SFT LoRA 训练 ──► DPO 数据构建 ──► DPO LoRA 训练 ──► 模型推断/评估
     │            │                 │               │                 │                 │
     ▼            ▼                 ▼               ▼                 ▼                 ▼
lawzhidao    prepare + self-instruct  qwen2.5-7b      A/B 候选        qwen2.5-7b        infer_*.py
 .csv        build_sft_dataset.py     _sft_lora      judge 模型        _dpo_lora
```

### SFT 训练要点（节选）

| 参数 | 值 | 说明 |
|-----|----|------|
| `cutoff_len` | 768 | 99% 数据 < 721 tokens，768 足够，大幅省显存/提速 |
| `num_train_epochs` | 1.0 | 单 epoch，避免过拟合 |
| `per_device_train_batch_size` | 4 | + `gradient_accumulation_steps=4` → **有效 batch=16** |
| `learning_rate` | 1e-5 | LoRA 常用学习率 |
| `lora_target` | all | 对全部线性层做 LoRA 微调 |
| `lora_rank / alpha` | 8 / 16 | alpha = 2×rank，经典配置 |
| `bf16` | True | bf16 训练，精度/显存/速度平衡 |
| `flash_attn` | auto | Flash Attention 加速 + 显存友好 |

### DPO 训练要点（节选）

| 参数 | 值 | 说明 |
|-----|----|------|
| `stage` | dpo | Direct Preference Optimization，偏好优化 |
| `cutoff_len` | 512 | 比 SFT 更短（DPO 输入=prompt+chosen+rejected，省显存） |
| `per_device_train_batch_size` | 2 | + `gradient_accumulation_steps=4` → **有效 batch=8** |
| `learning_rate` | 5e-6 | DPO 通常用比 SFT 更小的 LR（约 1/2） |
| `beta` | 0.1 | DPO loss 的 KL 惩罚系数 |

### 脚本一览（`LLM_DataSet_Train/llm/`）

| 脚本 | 阶段 | 功能 |
|------|------|------|
| `prepare_lawzhidao_sft.py` | 数据准备 | CSV → seed(200) + SFT 全量 JSON |
| `generate_self_instruct_qwen_api.py` | 数据准备 | Self-Instruct 扩充（Qwen2.5-32B API） |
| `build_sft_dataset.py` | 数据准备 | 三源融合（alpaca+lawzhidao+self-instruct）→ 9:1 切分 train/val |
| `generate_dpo_candidates_qwen_api.py` | DPO-1 | 生成 A/B 候选（temp=0.2 vs 0.9） |
| `judge_dpo_pairs_qwen_api.py` | DPO-2 | Qwen API 裁判标注 chosen/rejected |
| `infer_lora.py` | 推断 | SFT LoRA 推断（base + LoRA） |
| `infer_dpo_lora.py` | 推断 | DPO LoRA 推断（MODE=dpo / MODE=sft_dpo） |

> 💡 详细的训练命令、数据流、产物路径和问题，请参考 `更新日志-大模型微调.md`。

---

## 🐛 常见问题

### 文件上传问题

**Q: 上传 PDF 后提示 "PyMuPDF 未安装"**
A: 执行 `pip install pymupdf` 或 `uv add pymupdf` 安装依赖。

**Q: 上传 .doc（旧版 Word）失败？**
A: `.doc` 是 Word 97-2003 二进制格式，python-docx 不支持。请先用 Word / LibreOffice 另存为 `.docx`。

**Q: 前端文件选择对话框里看不到 PDF？**
A: 这是典型的前端白名单遗漏问题，请确认：
1. `static/index.html` 的 `<input accept="...">` 中包含 `.pdf,.docx`
2. `static/app.js` 的 `allowedExtensions` 数组中包含相应扩展名
3. 刷新浏览器缓存（Ctrl+F5）

### 向量库与索引问题

**Q: Milvus 连接失败？**
A: 确保本机 Docker 服务已启动（可用 Docker Desktop）。执行 `docker ps | grep milvus` 确认 Milvus 容器运行中；如需重启：`docker compose -f vector-database.yml restart`。

**Q: 修改了一个文档后重新上传，向量库里同时存在新旧内容？**
A: v1.4+ 已通过事件驱动 + doc_id 关联 + 先删后增修复此问题。请确认使用的是最新代码（上传接口走 `vector_index_service.index_single_file`）。

**Q: Consumer 日志在哪里看？**
A: Consumer 事件处理的日志通过 `app.main` 的 logger 输出，查看 `logs/app_YYYY-MM-DD.log` 中 `[Consumer]` 前缀的日志。

### 对话与 Agent 问题

**Q: 连续对话多轮后上下文变乱？**
A: v1.0.1+ 的对话记忆总结机制会在累计 5 轮后自动压缩历史。若想更频繁总结，调低 `SUMMARY_TRIGGER_ROUNDS=3`；若想保留更多原始对话，调高此值。

**Q: 对话答案里有"ignore previous instructions"之类的垃圾？**
A: 请确认 `AGENT_SAFETY_ENABLED=true`（默认已开启），Agent 五层安全防护中的层 4（记忆投毒防护）会自动检测并清理此类内容。

**Q: 多意图问题回答不完整？**
A: 检查 `INTENT_ORCHESTRATION_ENABLED=true`；如已开启，可在日志中搜索 `[编排]` 前缀查看子任务执行情况和汇总模式（`LLM 综合汇总` / `取唯一成功` / `拼接所有` / `fallback`）。

### 指标与监控

**Q: `/metrics` 端点返回空？**
A: 确认 `.env` 中 `METRICS_ENABLED=true`（或 config.py 中对应字段）。启动日志中应可见 `[Metrics] prometheus_client 已加载，指标注册完成`。

### 网络检索

**Q: 网络检索一直不触发？**
A: 自动触发需要两个条件：① `WEB_SEARCH_ENABLED=true` 且配置了 `TAVILY_API_KEY`；② 混合检索的最高分 < `WEB_SEARCH_AUTO_TRIGGER_THRESHOLD`（默认 0.3）。你也可以通过前端 `网络搜索` 开关手动启用（请求体加 `EnableWebSearch: true`）。

### GPU / 微调问题

**Q: 训练时 GPU 显存不足？**
A: 可尝试：① 减小 `cutoff_len`（如从 768 → 512）；② 减小 `per_device_train_batch_size`；③ 增大 `gradient_accumulation_steps` 保持有效 batch；④ 开启 `gradient_checkpointing`（若 LLaMA-Factory 支持）。

---

## 📚 参考资源

### 应用层
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph Plan-Execute](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [阿里云 DashScope](https://dashscope.aliyun.com/)
- [MCP 协议](https://modelcontextprotocol.io/)

### 向量与检索
- [Milvus 文档](https://milvus.io/docs)
- [rank-bm25 (BM25 算法)](https://github.com/dorianbrown/rank_bm25)

### 网络检索
- [Tavily API](https://tavily.com/)

### 指标与监控
- [Prometheus 文档](https://prometheus.io/docs/)
- [prometheus_client (Python)](https://github.com/prometheus/client_python)
- [Grafana 文档](https://grafana.com/docs/)

### 大模型微调
- [LLaMA-Factory (GitHub)](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5 (HuggingFace)](https://huggingface.co/Qwen)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [DPO 论文](https://arxiv.org/abs/2305.18290)

### 消息队列
- [Apache Kafka](https://kafka.apache.org/)
- [confluent-kafka Python](https://github.com/confluentinc/confluent-kafka-python)

---

## 📄 许可证

**Author**: chief

MIT License

---

## 📋 更新日志索引

各功能模块的详细设计与实现说明请参考独立更新日志：

- `更新日志-网络检索.md` — Tavily 网络检索 + 双层安全过滤 + 摘要压缩
- `更新日志-知识库多类型文件支持.md` — PDF/Word/MD/TXT 四格式处理器工厂
- `更新日志-混合检索与重排.md` — 向量 + BM25 双通道 + Rerank 二次过滤
- `更新日志-意图驱动的自动分解编排.md` — 多意图/复杂意图分解 + 并行工具调用 + LLM 汇总
- `更新日志-对话记忆总结机制.md` — LLM 总结压缩 + 有界上下文 + 增量总结
- `更新日志-多轮意图识别.md` — 结构化意图（Pydantic）+ 意图轨迹追踪 + 前端意图卡片
- `更新日志-向量知识库动态更新.md` — Kafka 事件驱动 + SHA-256 变更检测 + 先删后增
- `更新日志-业务指标.md` — Prometheus 6 指标集 + FastAPI Middleware + `/metrics` 端点
- `更新日志-Agent安全.md` — 五层安全防护（输入/文档/工具/记忆/输出）
- `更新日志-大模型微调.md` — Qwen2.5-7B SFT+DPO 全链路训练流水线