# 📚 Personal Knowledge Assistant

基于 RAG 架构的个人知识库助手，支持混合检索（向量 + BM25 + RRF 融合）、CrossEncoder 重排序、流式生成，并内置 LLM-as-Judge 评估和消融实验框架。

## ✨ 特性

- 🔍 **混合检索** — 向量检索 + BM25 关键词检索，RRF 融合排序
- 🎯 **二阶段重排序** — CrossEncoder Reranker 精排
- 🤖 **流式生成** — 基于 DeepSeek LLM，逐 token 实时输出
- 📊 **可观测性仪表盘** — 各阶段延迟分解、趋势图、请求记录
- 🧪 **消融实验** — 对比 5 种检索策略，LLM-as-Judge 三维度打分
- ⚡ **增量索引** — 基于文件 hash，跳过未变更文件
- 💾 **本地向量库** — ChromaDB 持久化，零运维
- 🧠 **知识图谱增强** — LLM 抽取三元组，NetworkX 图谱检索补召回

## 🏗️ 架构

```
Notes → Data Pipeline → ChromaDB ─┐
                             ├→ Retriever (Vector/BM25/KG) → Reranker → Generator → Gradio
KG Store ─────────────────────────┘
Intent Router (Global/Local) ─────► Retriever
```

## 📁 项目结构

```
personal-knowledge-assistant/
├── chroma_db/             # ChromaDB 持久化目录
├── eval_reports/          # 消融实验报告
├── notes/                  # Markdown 笔记（38 篇示例）
├── tests/                  # 单元测试 (pytest)
│   ├── test_data_pipeline.py
│   ├── test_retriever.py
│   ├── test_metrics.py
│   └── test_embedder.py
├── config.py               # 集中配置管理（从 .env 加载）
├── logger.py               # 统一日志配置
├── metrics.py              # 可观测性（计时 + 指标收集）
├── embedder.py             # Embedding 模块（智谱 API + 重试）
├── data_pipeline.py        # 数据处理 + 增量索引
├── retriever.py            # 混合检索（Vector + BM25 + RRF + Rerank）
├── intent_router.py        # 意图路由（原型向量分类）
├── intent_prototypes.json  # 路由原型示例
├── generator.py            # LLM 生成（阻塞 + 流式）
├── kg_extractor.py         # LLM 三元组抽取
├── kg_store.py             # 本地 JSON 三元组存储
├── kg_retriever.py         # KG 检索（NetworkX）
├── evaluator.py            # 评估（Embedding + LLM-as-Judge）
├── ablation.py             # 消融实验脚本
├── app.py                  # Gradio 应用（聊天 + 仪表盘）
├── Dockerfile              # Docker 一键部署
├── .env.example            # 环境变量模板
├── kg_triples.json         # 知识图谱三元组（运行后生成）
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/hqc135/personal-knowledge-assistant1
cd personal-knowledge-assistant1
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 ZHIPUAI_API_KEY 和 DEEPSEEK_API_KEY
```

### 3. 构建索引 & 启动

```bash
python data_pipeline.py   # 构建/增量更新索引
python app.py              # 启动应用 → http://localhost:7860
```

## 🧪 测试

```bash
pytest -v
```

## 🐳 Docker 部署

```bash
docker build -t knowledge-assistant .
docker run -p 7860:7860 --env-file .env knowledge-assistant
```

## 📊 消融实验

对齐主链路（含路由/KG/邻居扩展）的消融对比，使用 Embedding 相似度 + LLM-as-Judge 三维度评估：

```bash
python ablation.py
```

输出示例：

```
📊 消融实验结果对比
══════════════════════════════════════════════════
实验                    Emb Rel  Emb Faith  LLM Rel LLM Faith  LLM Comp
──────────────────────────────────────────────────
Auto (Main Pipeline)     0.6500     0.8200      4.2       4.4       4.1 ★
Auto (No Intent Router)  0.6200     0.7800      3.4       3.8       3.4
Vector Only              0.6234     0.7102      3.0       3.7       3.3
Vector + Reranker        0.6891     0.7503      3.7       4.0       3.7
BM25 Only                0.5102     0.6201      2.3       3.3       2.7
Hybrid (RRF)             0.7012     0.7601      4.0       4.0       4.0
Hybrid + Reranker        0.7234     0.7890      4.3       4.3       4.3
══════════════════════════════════════════════════
```

详细 JSON 报告保存在 `eval_reports/` 目录。

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Embedding | 智谱 `embedding-3` API | 高质量中文向量化 (1024 维) |
| BM25 | `rank_bm25` + `jieba` | 关键词检索，jieba 中文分词 |
| 融合排序 | RRF (Reciprocal Rank Fusion) | 多路检索结果融合 |
| Reranker | `BAAI/bge-reranker-v2-m3` | 交叉编码器精排 |
| Vector DB | ChromaDB | 本地持久化，零运维 |
| Knowledge Graph | NetworkX + JSON | 图谱检索，补充结构化召回 |
| LLM | deepseek-v4-flash | OpenAI 兼容接口 |
| 评估 | LLM-as-Judge + Embedding | 三维度自动化评估 |
| Frontend | Gradio (Blocks) | 聊天 + 可观测性仪表盘 |

## 📝 设计决策

- **混合检索 (Hybrid Search)**: 纯语义检索对精确关键词弱，BM25 补充关键词匹配，RRF 融合两路结果
- **两阶段排序**: 先多路召回 top-5，再 Rerank 取 top-3，平衡召回率和精准度
- **LLM-as-Judge**: 从检索相关性、回答忠实度、回答完整度三个维度自动化评估
- **消融实验**: 对比 5 种策略组合，用数据支撑技术选型决策
- **可观测性**: 全链路计时，各阶段延迟可视化，便于性能调优
- **增量索引**: 文件 MD5 hash 追踪，避免重复 API 调用
- **KG 增强**: 抽取三元组并存储为 JSON，检索时从图谱召回关联 chunk
- **意图路由**: 基于原型向量分类，低置信度回退到本地检索链路

## ⚙️ KG 配置

KG 相关参数已集中在 `.env` / `config.py`。需要时只修改 `USE_KG_EXTRACTION`、`USE_KG_RETRIEVAL` 以及 `KG_*` 参数即可。

## 🧭 意图路由配置

意图路由默认开启，参数集中在 `.env` / `config.py`。通常只需调整 `USE_INTENT_ROUTER`、`INTENT_ROUTER_*` 的阈值与全局模式。

## 📄 License

MIT
