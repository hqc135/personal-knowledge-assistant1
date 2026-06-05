# 📚 Personal Knowledge Assistant

基于高阶 RAG 架构的个人知识库助手，内置了行业前沿的多种检索优化策略：包括层级切分、代码块隔离屏蔽、意图路由、查询对齐、混合检索（向量 + BM25 + KG）、CrossEncoder 重排序及流式生成。

系统还自带了一套强大的 LLM-as-Judge 消融实验评估框架，并通过渐进式全链路的可观测仪表盘（Gradio UI）帮助开发者直观调优。

## ✨ 核心特性

- 🔍 **混合架构检索 (Hybrid Search)** — 融合向量相似度、BM25 词频匹配与知识图谱关系检索，通过 RRF（Reciprocal Rank Fusion）统一排序融合。
- 🎯 **二阶段精排 (Reranking)** — 借助 CrossEncoder 模型对粗召回结果进行高精度重排打分。
- 🧱 **父子分层 RAG (Parent-Child Hierarchical RAG)** — 离线细粒度切块（Child），命中后动态召回大文本块（Parent），解决片段化语义丢失问题。
- 🛡️ **结构感知代码 RAG (Structure-Aware Code RAG)** — 物理隔离 Markdown 代码块，避免纯自然语言正则误切；引入“断路器”机制智能跳过代码块的无效大模型图谱抽取。
- 🧠 **知识图谱增强 (Knowledge Graph)** — 离线自动化抽取三元组建立图谱（NetworkX），在线通过实体提取进行两跳关系检索，极大提升推理型问题的召回能力。
- 🧭 **意图路由 (Intent Router)** — 根据 Query 的原型向量聚类得分，智能切换“局部精确问答”与“全局总结”双重检索模式。
- 🌐 **查询扩展管线 (Query Enhancements)** — 内置 Query Aligner（多语言术语对齐）与 Query Rewriter（上下文感知重写），保证召回词汇的结构精准度。
- 📊 **可观测仪表盘 (Observability Dashboard)** — 内置 SVG 渲染引擎，实时拆解各阶段延迟（Routing/Retrieval/Rerank/Generation），方便性能调优。
- 🧪 **自动化评估与消融 (Ablation Framework)** — 自带一键多线程跑批脚本，利用 LLM-as-Judge 从“检索相关度”、“回答忠实度”及“完整性”三个维度全方位输出数据报告。

## 🏗️ 架构图解

```text
Offline Data Pipeline:
Notes ──> Semantic Chunker (w/ Code Shielding) ──> Parent/Child Split ──> ChromaDB
                                              └──> LLM Extractor ───────> KG Store (NetworkX)

Online Retrieval Flow:
Query ──> Intent Router ──> Query Rewriter & Aligner ──> Multi-Way Retrieval ──> Reranker ──> Generator
                                 ├── Vector Store (Child-to-Parent)
                                 ├── BM25 Index
                                 └── KG Graph Sub-network
```

## 📁 项目结构

```
personal-knowledge-assistant/
├── chroma_db/             # ChromaDB 向量持久化目录
├── eval_reports/          # JSON 格式的离线消融实验报告输出目录
├── notes/                 # 你的个人 Markdown 笔记知识源
├── tests/                 # Pytest 单元测试组件
├── core/                  # 核心数据契约（Schema）与管线编排（Pipeline）
├── retrieval/             # 细分检索策略（BM25, Reranker, 扩展器等）
├── ui/                    # Gradio 视图层（聊天视图、主题、统计仪表盘）
├── config.py              # `.env` 配置集中入口
├── data_pipeline.py       # 核心离线处理器（含增量哈希、Chunking）
├── parent_child_chunker.py# 专职处理父子块的层级切分器
├── retriever.py           # 运行时混合召回总控入口
├── intent_router.py       # Query 意图分类与多路分发
├── query_aligner.py       # 术语级对齐模块
├── query_rewriter.py      # LLM 意图改写模块
├── kg_*.py                # 知识图谱三元组抽取、存储与检索系列
├── generator.py           # LLM 流式问答生成器
├── evaluator.py           # LLM-as-Judge 与 Embedding 打分器
├── ablation.py            # 多维消融实验并发脚本
├── app.py                 # Gradio Web 界面主入口
├── Dockerfile             # 容器化部署脚本
└── .env.example           # 环境变量参考模板
```

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/hqc135/personal-knowledge-assistant1
cd personal-knowledge-assistant1
python -m venv .venv
# 激活虚拟环境 (Windows)
.\.venv\Scripts\activate  
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制并编辑 `.env` 文件：
```bash
cp .env.example .env
```
填入你的 `ZHIPUAI_API_KEY`（用于 Embeddings）和 `DEEPSEEK_API_KEY`（用于检索重写与回答生成）。

### 3. 构建索引并启动

当你首次运行或放入了新的笔记时，需要运行管线构建索引与图谱：
```bash
python data_pipeline.py
```
> **提示**：如果调整了底层切分策略（如代码盾牌或父子分块），建议先删除 `./chroma_db` 与 `./kg_triples.json` 后再运行上条命令强制重构。

启动基于 Gradio 的对话与监控面板：
```bash
python app.py
```
打开浏览器访问 [http://localhost:7860](http://localhost:7860) 即可开始使用。

## 🧪 测试与评估

**运行单元测试**：
```bash
pytest -v
```

**运行消融实验**：
系统自带的 `ablation.py` 会对比多种架构（从 Base Vector Only 到 Full Pipeline），并测试父子检索和图谱增强的效能差异：
```bash
python ablation.py
```
*实验结果会自动打印在控制台，并将带有详细分数的 JSON 报告留存在 `eval_reports/` 目录下。*

## 🐳 Docker 部署

```bash
docker build -t knowledge-assistant .
docker run -p 7860:7860 --env-file .env knowledge-assistant
```

## 📝 高阶调优说明

所有的阈值与开关都在 `.env` (并由 `config.py` 解析) 中：
- **层级架构**：调整 `USE_PARENT_CHILD` 和 Chunk 大小限制。
- **知识图谱**：调整 `USE_KG_EXTRACTION` / `USE_KG_RETRIEVAL`，并可限制最大跳数 `KG_MAX_HOPS`。
- **查询处理**：通过 `USE_INTENT_ROUTER`、`USE_QUERY_ALIGNER` 改变前置的请求干预力度。

## 📄 License
MIT
