
# 文本嵌入模型：BGE、E5、OpenAI Embedding 原理与选型

## 嵌入模型在 RAG 中的位置

嵌入模型是 RAG 的"地基"——它决定了语义空间的质量。如果 embedding 不好，后面的检索、重排序都是在垃圾上做优化。

核心任务：把文本映射到一个稠密向量空间，使得语义相似的文本在空间中距离近。

## 训练范式演进

```
Word2Vec → Sentence-BERT → 对比学习 (SimCSE) → 指令微调 (E5/BGE)
```

现代嵌入模型基本都是 **Bi-encoder 架构**：

```
text_a → Encoder → pooling → vec_a
text_b → Encoder → pooling → vec_b
similarity = cosine(vec_a, vec_b)
```

> 和重排序的 Cross-encoder 对比记忆：Bi-encoder 可以离线算好所有文档向量存起来，查询时只需要算 query 的向量然后做 ANN 搜索。Cross-encoder 必须把 query 和每个 doc 拼在一起过模型，所以只能用在重排序阶段。

## 主流模型详解

### BGE（BAAI General Embedding）

- 智源出品，中英文都很强
- 基于 RetroMAE 预训练 + 对比学习微调
- 有个关键 trick：**query 前面加 "Represent this sentence: " 前缀**
- 系列：bge-small (384d) / bge-base (768d) / bge-large (1024d)
- bge-m3：多语言、多粒度、多功能（dense + sparse + colbert）

```python
# BGE 使用示例
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
# 注意：query 要加前缀，document 不加
query_emb = model.encode("为这个句子生成表示：什么是向量数据库")
doc_emb = model.encode("向量数据库是专门存储和检索向量的数据库系统")
```

> 踩过的坑：忘了加 query 前缀，recall 直接掉了好几个点。这个设计是为了区分"查询意图"和"文档内容"在语义空间中的分布。

### E5（EmbEddings from bidirEctional Encoder rEpresentations）

- 微软出品
- 核心创新：**指令微调**（instruction-tuned embedding）
- E5-mistral：用 Mistral-7B 做 backbone，效果炸裂但推理慢
- 前缀格式：`query: ` 和 `passage: `

训练数据构造很有意思：
1. 先用 GPT 生成大量 (query, positive, negative) 三元组
2. 用对比学习 loss 训练

```
# InfoNCE Loss（对比学习核心 loss）
L = -log( exp(sim(q, p+)/τ) / Σ exp(sim(q, pi)/τ) )
```

> 面试高频考点：温度参数 τ 的作用——τ 越小，分布越尖锐，模型越关注 hard negatives。这和 Transformer 中 attention score 除以 √d_k 的 scaling 思路类似，都是在控制 softmax 的锐度。

### OpenAI Embedding

- text-embedding-3-small (1536d) / text-embedding-3-large (3072d)
- 支持 **Matryoshka Representation Learning**：可以截断到更短维度而不严重损失性能
- 闭源 API，不知道具体架构
- 优点：效果好、使用简单；缺点：有 API 调用成本、数据隐私问题

### 其他值得关注的

- **GTE**（阿里）：和 BGE 类似定位，中文场景表现好
- **Jina Embeddings**：支持 8192 token 长文本
- **Cohere Embed v3**：商用 API，支持多种 input_type

## 选型考量

| 维度 | 考虑因素 |
|------|----------|
| 语言 | 中文为主选 BGE/GTE，英文为主选 E5 |
| 部署 | 能自部署选开源，不能就 OpenAI/Cohere |
| 维度 | 维度越高精度越好但存储/计算成本越大 |
| 长度 | 大部分模型 512 token 上限，长文本需要分块或选 Jina |
| 延迟 | 小模型（bge-small）适合实时场景 |

## Pooling 策略

把 token-level 表示聚合成 sentence-level 的方式：

- **[CLS] token**：BERT 原始方式，但直接用效果差
- **Mean pooling**：所有 token 取平均，最常用
- **Last token**：decoder-only 模型（如 E5-mistral）用最后一个 token
- **Weighted mean**：按 attention weight 加权平均

> 这里我一开始理解错了：以为 [CLS] 一定比 mean pooling 好（毕竟是专门设计的聚合 token），但实际上未经微调的 BERT [CLS] 表示质量很差，SimCSE 论文里有详细分析。

## 评估指标

- **MTEB Benchmark**：最权威的嵌入模型评测，涵盖检索、分类、聚类等任务
- 关注 Retrieval 子任务的 nDCG@10
- 注意：MTEB 上的排名不一定适用于你的特定领域，最好在自己的数据上评测

## 实践建议

1. 先用 bge-large 或 E5-large 跑个 baseline
2. 如果延迟敏感，换 small/base 版本看精度掉多少
3. 如果领域很垂直（医疗、法律），考虑在领域数据上 finetune
4. embedding 模型一旦选定，换模型意味着全量重新编码——这和向量数据库的索引重建是同一个问题

> 和文档分块策略的关系：embedding 模型的 max_length 直接约束了分块大小。如果模型只支持 512 token，你的 chunk 超过这个长度就会被截断，信息丢失。