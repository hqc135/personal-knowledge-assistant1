

# 重排序模型：Cross-encoder vs Bi-encoder、ColBERT

## 为什么需要重排序

RAG 的检索是两阶段的：

```
全量文档 → [召回阶段: Bi-encoder + ANN] → top-100 → [精排阶段: Reranker] → top-5 → LLM
```

召回阶段追求**速度和召回率**（别漏掉相关文档），精排阶段追求**精确度**（把最相关的排到最前面）。

> 面试高频考点：这个两阶段架构和搜索引擎、推荐系统的"召回-排序"是完全一样的范式。

## Bi-encoder vs Cross-encoder

这是理解重排序的核心对比：

### Bi-encoder（双塔模型）

```
query  → Encoder → q_vec ─┐
                           ├→ cosine(q_vec, d_vec) → score
doc    → Encoder → d_vec ─┘
```

- query 和 doc **独立编码**，互不影响
- doc 向量可以离线预计算
- 查询时只需算 query 向量 + ANN 搜索
- 速度快，但精度有限（因为 query 和 doc 之间没有交互）

### Cross-encoder（交叉编码器）

```
[CLS] query [SEP] doc [SEP] → Encoder → [CLS] hidden → Linear → score
```

- query 和 doc **拼接后一起编码**
- 每一层 attention 都能看到 query 和 doc 的所有 token
- 精度高（token 级别的交互），但速度慢（不能预计算）

```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 必须成对打分，不能预计算
scores = model.predict([
    ("什么是向量数据库", "向量数据库是专门存储向量的系统"),
    ("什么是向量数据库", "今天天气很好"),
])
# scores: [0.95, 0.02]
```

> 这里我一开始理解错了：以为 Cross-encoder 只是"更大的模型所以更准"。实际上关键区别是**交互方式**——Cross-encoder 的 self-attention 让 query 的每个 token 都能 attend 到 doc 的每个 token，这种细粒度交互是 Bi-encoder 做不到的。这和 Transformer 中 self-attention vs cross-attention 的区别是一回事。

### 对比总结

| 维度 | Bi-encoder | Cross-encoder |
|------|-----------|---------------|
| 编码方式 | 独立编码 | 联合编码 |
| 预计算 | ✅ 可以 | ❌ 不行 |
| 速度 | 快（ms 级） | 慢（100ms+ per pair） |
| 精度 | 较好 | 最好 |
| 适用阶段 | 召回 | 精排 |
| 候选规模 | 百万级 | 几十到几百 |

## ColBERT：Late Interaction 的折中方案

ColBERT 是 Bi-encoder 和 Cross-encoder 之间的折中：

```
query tokens → Encoder → [q1, q2, ..., qm]  (保留所有 token 向量)
doc tokens   → Encoder → [d1, d2, ..., dn]  (保留所有 token 向量)

# MaxSim 操作
score = Σ_i max_j cosine(qi, dj)
```

核心思想：**Late Interaction**
- 编码阶段：query 和 doc 独立编码（像 Bi-encoder）
- 打分阶段：token 级别交互（像 Cross-encoder，但更轻量）

```python
# ColBERT MaxSim 伪代码
def colbert_score(query_embeddings, doc_embeddings):
    """
    query_embeddings: [m, d]  # m 个 query token
    doc_embeddings: [n, d]    # n 个 doc token
    """
    # 计算所有 token pair 的相似度
    sim_matrix = query_embeddings @ doc_embeddings.T  # [m, n]
    # 每个 query token 取与 doc 中最相似的 token
    max_sim = sim_matrix.max(dim=1).values  # [m]
    return max_sim.sum()
```

> MaxSim 的直觉：对于 query 中的每个词，找到文档中最匹配的词，然后把所有匹配分数加起来。这比单一向量的 cosine 保留了更多细粒度信息。

### ColBERT 的优势

1. doc 向量可以离线预计算（和 Bi-encoder 一样）
2. 精度接近 Cross-encoder
3. 速度比 Cross-encoder 快很多

### ColBERT 的代价

- 存储量大：每个 doc 要存所有 token 的向量（不是一个向量，是 n 个）
- 对于 100 token 的文档，存储量是 Bi-encoder 的 100 倍

> 和向量数据库的关系：ColBERT 的存储需求对向量数据库提出了更高要求。BGE-M3 模型同时输出 dense、sparse、colbert 三种表示，可以配合 Milvus 的多向量检索功能使用。

## 常用重排序模型

| 模型 | 类型 | 特点 |
|------|------|------|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder | 轻量，英文 |
| BAAI/bge-reranker-v2-m3 | Cross-encoder | 多语言，中文好 |
| Cohere Rerank | API | 商用，效果好 |
| ColBERTv2 | Late Interaction | 速度精度平衡 |
| bge-reranker-v2-gemma | Cross-encoder | 基于 LLM，效果强 |

## 实践中的重排序 Pipeline

```python
# 典型的两阶段检索
def retrieve_and_rerank(query, top_k=5):
    # Stage 1: 召回（Bi-encoder + ANN）
    query_emb = bi_encoder.encode(query)
    candidates = vector_db.search(query_emb, top_n=100)
    
    # Stage 2: 精排（Cross-encoder）
    pairs = [(query, doc.text) for doc in candidates]
    scores = cross_encoder.predict(pairs)
    
    # 按重排序分数排序
    reranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return reranked[:top_k]
```

## 踩过的坑

1. **Cross-encoder 的输入长度限制**：大部分模型 max_length=512，query+doc 超过就截断。长文档要么先截断要么用支持长文本的模型。

2. **重排序不能弥补召回的缺陷**：如果召回阶段就没把相关文档捞回来，重排序再好也没用。所以召回阶段要保证高 recall，宁可多召回一些噪声让 reranker 去过滤。

3. **batch size 对延迟的影响**：Cross-encoder 对 100 个候选打分，如果逐个推理会很慢。要用 batch inference，GPU 利用率才能上去。

4. **分数校准问题**：Cross-encoder 输出的 score 不是概率，不同 query 之间的分数不可比。不要用绝对阈值过滤，要用相对排序。

## 什么时候不需要重排序

- 数据量小（< 1000 条），Bi-encoder 精度已经够用
- 延迟要求极高（< 50ms），加不起 reranker 的开销
- embedding 模型本身质量很高，召回 top-10 已经很准

## LLM as Reranker

最近的趋势：直接用 LLM 做重排序。

```python
prompt = """
Given the query: "{query}"
Rank the following documents by relevance:
[1] {doc1}
[2] {doc2}
[3] {doc3}
Output the ranking as a list of numbers.
"""
```

优点：利用 LLM 的语义理解能力，zero-shot 就能用
缺点：成本高、延迟大、输出不稳定

> 和混合检索的关系：重排序阶段可以融合多路召回的结果。比如 BM25 召回 50 个 + 向量召回 50 个，去重后一起送给 reranker 打分，这比单独对每路结果重排序效果更好。

## 面试常见问题

- Q：为什么不直接用 Cross-encoder 做全量检索？A：计算量太大。100 万文档 × 100ms/pair = 不可接受的延迟。
- Q：ColBERT 和 Cross-encoder 的本质区别？A：ColBERT 的交互发生在编码之后（late interaction），不影响编码过程；Cross-encoder 的交互贯穿整个编码过程。
- Q：Reranker 需要和 Retriever 用同一个模型吗？A：不需要，而且通常不应该。Retriever 用轻量 Bi-encoder，Reranker 用重量级 Cross-encoder，各司其职。



