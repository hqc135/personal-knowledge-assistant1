# RAG 学习笔记

## 什么是 RAG

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术框架，通过从外部知识库中检索相关文档，再将其作为上下文输入到大语言模型中进行回答。

## 核心组件

1. **文档处理 (Data Pipeline)**: 将原始文档分割成小块 (chunks)，并生成向量嵌入 (embeddings)
2. **向量检索 (Retriever)**: 根据用户查询，从向量数据库中检索最相关的文档片段
3. **重排序 (Reranker)**: 使用交叉编码器对初步检索结果进行精排
4. **生成 (Generator)**: 将检索到的上下文与用户问题一起发送给 LLM 生成回答

## 关键设计决策

- Chunk size 设为 512，overlap 64，平衡上下文完整性和检索精度
- 使用 BGE-small-zh 作为 embedding 模型，轻量且支持中文
- 使用 BGE-reranker-v2-m3 做二阶段重排序，提升检索质量
- ChromaDB 作为向量数据库，本地持久化，无需外部服务
