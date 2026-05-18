# 📚 Personal Knowledge Assistant

基于 RAG（Retrieval-Augmented Generation）架构的个人知识库助手，能够从你的 Markdown 笔记中检索相关内容，并通过大语言模型生成智能回答。

## ✨ 特性

- 🔍 **语义检索** — 使用智谱 embedding-3 API 进行向量化检索
- 🎯 **二阶段重排序** — 使用 CrossEncoder Reranker 提升检索精度
- 🤖 **流式生成** — 基于 DeepSeek LLM，逐 token 实时输出
- 💾 **本地向量库** — ChromaDB 持久化存储，无需外部数据库服务
- ⚡ **增量索引** — 基于文件 hash，只重新处理变更/新增的笔记
- 📊 **评估框架** — 内置检索相关性和回答忠实度评估
- 🖥️ **Gradio 界面** — 开箱即用的聊天界面

## 🏗️ 架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Markdown   │────▶│   Data       │────▶│   ChromaDB   │
│   Notes      │     │   Pipeline   │     │   (Vector DB)│
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│   Gradio     │◀────│   Generator  │◀────│   Retriever  │
│   Frontend   │     │   (DeepSeek) │     │   + Reranker │
└──────────────┘     └──────────────┘     └──────────────┘
```

## 📁 项目结构

```
personal-knowledge-assistant/
├── notes/                  # 你的 Markdown 笔记（数据源）
│   ├── rag-notes.md
│   ├── system-design.md
│   └── ...
├── config.py               # 集中配置管理（从 .env 加载）
├── logger.py               # 统一日志配置
├── embedder.py             # 统一 Embedding 模块（智谱 API）
├── data_pipeline.py        # 数据处理 + 增量索引
├── retriever.py            # 向量检索 + Rerank 二阶段排序
├── generator.py            # LLM 回答生成（支持流式输出）
├── evaluator.py            # 评估框架（相关性 + 忠实度）
├── app.py                  # Gradio 聊天界面入口
├── eval_cases.json         # 评估测试用例
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── .gitignore              # Git 忽略规则
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd personal-knowledge-assistant

# 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入必需的 API Key：

```dotenv
# 必填
ZHIPUAI_API_KEY=your-zhipuai-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here

# 可选（有默认值，按需修改）
# EMBEDDING_MODEL=embedding-3
# LLM_MODEL=deepseek-chat
# CHUNK_SIZE=512
```

> ⚠️ `.env` 文件已被 `.gitignore` 排除，不会提交到 Git，你的 Key 是安全的。
>
> 💡 所有可配置项见 `.env.example`，运行时由 `config.py` 统一加载。

### 3. 添加笔记

将你的 Markdown 笔记文件放入 `notes/` 目录下。

### 4. 构建知识库索引

```bash
python data_pipeline.py
```

首次运行会索引所有文件；后续运行自动增量索引（只处理新增/变更的文件）。

### 5. 启动应用

```bash
python app.py
```

访问 `http://localhost:7860` 即可开始使用。

## 📊 评估

项目内置了简单的评估框架，用于衡量检索质量和回答忠实度：

```python
from evaluator import SimpleEvaluator
from retriever import Retriever
from generator import Generator
import json

evaluator = SimpleEvaluator()
retriever = Retriever()
generator = Generator()

with open("eval_cases.json") as f:
    cases = json.load(f)

results = evaluator.run_eval(cases, retriever, generator)
print(f"平均检索相关性: {results['avg_relevance']:.3f}")
print(f"回答忠实度: {results['grounded_rate']:.1%}")
```

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Embedding | 智谱 `embedding-3` API | 高质量中文向量化，支持自定义维度 |
| Reranker | `BAAI/bge-reranker-v2-m3` | 交叉编码器精排，提升检索质量 |
| Vector DB | ChromaDB | 本地持久化，零运维 |
| LLM | DeepSeek Chat | OpenAI 兼容接口，性价比高 |
| Text Splitting | LangChain | 递归字符分割，支持中文 |
| Frontend | Gradio | 快速构建聊天界面 |

## 📝 设计决策

- **Chunk Size = 512, Overlap = 64**: 平衡上下文完整性和检索精度
- **两阶段检索**: 先向量召回 top-5，再 Rerank 取 top-3，提升精准度
- **API Embedding**: 使用智谱 embedding-3 API（1024 维），避免本地 GPU 依赖
- **增量索引**: 基于文件 MD5 hash，避免重复处理未变更文件，节省 API 额度
- **流式输出**: 逐 token 返回，减少用户等待感
- **幂等写入**: 使用 ChromaDB 的 `upsert` 操作，重复索引不会产生重复数据

## 📄 License

MIT
