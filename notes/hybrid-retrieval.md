
# 混合检索：BM25 + 向量检索 + RRF 融合

## 为什么需要混合检索

单一检索方式各有盲区：

- **向量检索**擅长语义匹配（"汽车" → "轿车"），但对精确关键词匹配弱（搜 "Error Code 4012" 可能匹配不上）
- **BM25**擅长精确关键词匹配，但不理解语义（"如何减肥" 搜不到 "控制体重的方法"）

混合检索 = 取长补短，两路召回 + 分数融合。

> 面试高频考点：这是 RAG 系统中提升检索质量最有效的手段之一，实现简单但效果显著。

## BM25 回顾

BM25 是经典的稀疏检索算法，基于词频统计：

```
BM25(q, d) = Σ IDF(qi) · (f(qi, d) · (k1 + 1)) / (f(qi, d) + k1 · (1 - b + b · |d|/avgdl))
```

各项含义：
- `IDF(qi)`：逆文档频率，罕见词权重高
- `f(qi, d)`：词 qi 在文档 d 中的词频
- `k1`：词频饱和参数（通常 1.2-2.0）
- `b`：文档长度归一化参数（通常 0.75）
- `|d|/avgdl`：文档长度 / 平均文档长度

```python
# 使用 rank_bm25 库
from rank_bm25 import BM25Okapi

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(query.split())
```

> 这里我一开始理解错了：以为 BM25 就是简单的 TF-IDF。实际上 BM25 在 TF-IDF 基础上加了词频饱和（一个词出现 10 次不会比出现 5 次好很多）和文档长度归一化（长文档不会因为词多就占便宜）。

### BM25 的优势

1. 不需要训练，不需要 GPU
2. 对精确匹配（专有名词、编号、代码）效果好
3. 可解释性强
4. 索引构建快，增量更新方便

### BM25 的局限

1. 无法处理同义词（"ML" vs "机器学习"）
2. 对短 query 效果差（信息量不够）
3. 不理解词序和上下文

## 向量检索回顾

```python
query_emb = embedding_model.encode(query)
results = vector_db.search(query_emb, top_k=50)
```

优势：语义理解、跨语言、对 query 改写鲁棒
局限：精确匹配弱、需要 GPU 推理、需要预计算索引

> 和 embedding 模型的关系：向量检索的质量完全取决于 embedding 模型。如果用的是通用模型但检索领域很垂直，可能还不如 BM25。这时候要么 finetune embedding 模型，要么靠混合检索兜底。

## 分数融合策略

两路检索各返回一个排序列表，怎么合并？

### 方法一：线性加权

```python
final_score = α * normalize(bm25_score) + (1-α) * normalize(vector_score)
```

问题：
- BM25 分数和向量相似度的量纲不同，需要归一化
- 归一化方式（min-max、z-score）会影响结果
- α 是超参，需要调优

### 方法二：RRF（Reciprocal Rank Fusion）⭐

RRF 是最常用的融合方法，简单有效：

```python
def rrf_score(doc, rankings, k=60):
    """
    rankings: 多路检索的排序结果
    k: 平滑参数（通常 60）
    """
    score = 0
    for ranking in rankings:
        if doc in ranking:
            rank = ranking.index(doc) + 1  # 排名从1开始
            score += 1.0 / (k + rank)
    return score
```

**核心公式**：

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

其中 `rank_i(d)` 是文档 d 在第 i 路检索中的排名。

> RRF 的精妙之处：它只用排名，不用原始分数。这完美避开了不同检索系统分数不可比的问题。k=60 是论文推荐值，实际中 40-80 都行。

```python
# 完整的混合检索实现
def hybrid_search(query, corpus, embedding_model, bm25_index, vector_db, top_k=10):
    # 路径1：BM25 检索
    bm25_scores = bm25_index.get_scores(tokenize(query))
    bm25_top = sorted(range(len(bm25_scores)), 
                      key=lambda i: -bm25_scores[i])[:100]
    
    # 路径2：向量检索
    query_emb = embedding_model.encode(query)
    vector_top = vector_db.search(query_emb, top_n=100)  # 返回 doc_ids
    
    # RRF 融合
    all_docs = set(bm25_top) | set(vector_top)
    rrf_scores = {}
    for doc_id in all_docs:
        score = 0
        if doc_id in bm25_top:
            score += 1.0 / (60 + bm25_top.index(doc_id) + 1)
        if doc_id in vector_top:
            score += 1.0 / (60 + vector_top.index(doc_id) + 1)
        rrf_scores[doc_id] = score
    
    # 按 RRF 分数排序
    final_ranking = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return final_ranking[:top_k]
```

### 方法三：学习排序（Learning to Rank）

用一个小模型学习如何融合多路信号：

```
features = [bm25_score, vector_score, bm25_rank, vector_rank, ...]
final_score = learned_model(features)
```

效果最好但需要标注数据，适合有足够训练数据的场景。

## 实际系统中的实现

### Elasticsearch 8.x

```json
{
  "query": {
    "bool": {
      "should": [
        { "match": { "content": "向量数据库" } },
        { "knn": { "field": "embedding", "query_vector": [...], "k": 50 } }
      ]
    }
  }
}
```

### Milvus + BM25

Milvus 2.4+ 原生支持 sparse vector（BM25 的稀疏表示），可以在一个系统内做混合检索。

### LangChain EnsembleRetriever

```python
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.vectorstores import FAISS

bm25_retriever = BM25Retriever.from_documents(docs)
faiss_retriever = FAISS.from_documents(docs, embeddings).as_retriever()

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]  # BM25 权重 0.4，向量权重 0.6
)
```

## 什么时候混合检索特别有用

1. **专业领域**：有大量专有名词、缩写、编号（医疗、法律、代码）
2. **中文场景**：中文分词质量影响 BM25，向量检索可以补位
3. **query 多样性大**：有些 query 是关键词型，有些是自然语言型
4. **不确定 embedding 质量**：BM25 作为兜底

## 踩过的坑

1. **中文 BM25 的分词很关键**：用 jieba 分词 vs 字符级分词效果差很多。停用词也要处理。

2. **RRF 的 k 值不是越小越好**：k 太小会让排名靠前的文档权重过大，k 太大则各排名差异被抹平。

3. **两路召回数量要平衡**：如果 BM25 召回 100 个、向量只召回 10 个，RRF 会偏向 BM25（因为更多文档有 BM25 排名）。

4. **去重逻辑**：同一个文档可能在两路中都出现，融合时要用 doc_id 去重，不能重复计算。

5. **BM25 索引更新**：新增文档后 IDF 会变化，理论上需要重建索引。实际中如果文档量大，增量更新对 IDF 影响很小，可以定期重建。

## 进阶：多路融合

不止两路，可以融合更多信号：

```
路径1：BM25（关键词匹配）
路径2：Dense vector（语义匹配）
路径3：Sparse vector / SPLADE（学习的稀疏表示）
路径4：ColBERT（token 级交互）
路径5：知识图谱检索（结构化关系）
```

BGE-M3 模型同时输出 dense + sparse + colbert 三种表示，天然适合多路融合。

> 和重排序的关系：混合检索解决的是"召回"阶段的问题（尽可能把相关文档捞回来），之后还可以接一个 Cross-encoder reranker 做精排。完整 pipeline：多路召回 → RRF 融合 → Reranker 精排 → top-k 送给 LLM。

## 面试常见问题

- Q：RRF 和线性加权哪个好？A：RRF 更鲁棒（不需要归一化、不需要调权重），线性加权在有标注数据调优后可能更好。
- Q：混合检索一定比纯向量检索好吗？A：不一定。如果 query 都是自然语言且 embedding 质量高，纯向量可能就够了。混合检索的优势在关键词匹配场景。
- Q：BM25 的 k1 和 b 怎么调？A：大部分场景默认值就行（k1=1.5, b=0.75）。如果文档长度差异大，可以适当调小 b。
