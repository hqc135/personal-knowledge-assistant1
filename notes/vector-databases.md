
# 向量数据库对比：Chroma、Milvus、Pinecone、FAISS

## 为什么需要向量数据库

传统数据库做精确匹配（WHERE name = 'xxx'），但在 RAG 场景下我们需要的是**语义相似度搜索**——给一个 query embedding，找到最近的 k 个文档向量。这就是 ANN（Approximate Nearest Neighbor）问题。

> 面试高频考点：为什么用"近似"而不是精确最近邻？因为精确 KNN 在高维空间是 O(n) 的，百万级向量根本扛不住。

## 核心索引算法

在对比具体产品之前，先搞清楚底层索引类型：

| 算法 | 原理 | 特点 |
|------|------|------|
| IVF (Inverted File) | 先用 K-means 聚类，查询时只搜最近的几个簇 | 需要训练，适合静态数据 |
| HNSW (Hierarchical NSW) | 多层跳表 + 贪心搜索 | 内存占用大，但查询快且无需训练 |
| PQ (Product Quantization) | 把向量切分子空间，每个子空间独立量化 | 压缩比高，精度有损 |
| ScaNN | Google 的各向异性量化 | 在精度-速度 tradeoff 上很强 |

```
# HNSW 查询伪代码
def search(query, entry_point, max_layer):
    current = entry_point
    for layer in range(max_layer, 0, -1):
        # 贪心搜索，每层只保留最近的1个
        current = greedy_search(query, current, layer, ef=1)
    # 最底层扩大搜索范围
    return greedy_search(query, current, layer=0, ef=ef_search)
```

> 这里我一开始理解错了：HNSW 不是树结构，是图结构。每一层是一个稀疏图，越往上节点越少（类似跳表的思想）。和 Transformer 的多头注意力有点像——不同层关注不同粒度的邻居关系。

## 四个产品对比

### FAISS（Meta）

- **定位**：库（library），不是服务
- 纯 C++ 实现，Python binding，GPU 加速支持好
- 适合离线实验、单机场景
- 没有持久化、没有 API 服务、没有元数据过滤（需要自己包一层）

```python
import faiss
index = faiss.IndexHNSWFlat(768, 32)  # dim=768, M=32
index.add(vectors)
D, I = index.search(query_vector, k=10)
```

### Chroma

- **定位**：轻量级嵌入式向量数据库，开发者友好
- Python-native，开箱即用，适合原型开发
- 底层用的 HNSW（hnswlib）
- 支持元数据过滤，但大规模性能一般
- **适合**：本地开发、小项目、快速 POC

### Milvus（Zilliz）

- **定位**：分布式、生产级向量数据库
- 支持多种索引（IVF_FLAT, IVF_PQ, HNSW, DiskANN）
- 存算分离架构，可以水平扩展
- 支持标量过滤 + 向量搜索的混合查询
- **适合**：大规模生产环境、需要高可用

> 和传统分布式数据库对比记忆：Milvus 的架构有点像 TiDB 的存算分离思路，query node / data node / index node 各司其职。

### Pinecone

- **定位**：全托管 SaaS 向量数据库
- 不开源，按用量付费
- 开发体验好，不用运维
- 支持 namespace 隔离、metadata filtering
- **适合**：不想运维、快速上线的团队

## 选型决策树

```
需要生产部署吗？
├── 否 → 只是实验/POC
│   ├── 数据量 < 100k → Chroma
│   └── 需要 GPU 加速 → FAISS
└── 是
    ├── 想要全托管 → Pinecone
    └── 需要自部署/定制 → Milvus
```

## 踩过的坑

1. **FAISS 的 IndexFlatL2 不做任何近似**，就是暴力搜索。别以为用了 FAISS 就自动快了。
2. Chroma 默认用 cosine similarity，但 FAISS 默认是 L2 distance。**归一化后 L2 和 cosine 等价**（这个面试常问）。
3. Milvus 的 collection 需要先 `load()` 到内存才能查询，忘了这步会报错。
4. 向量维度一旦建好索引就不能改，换 embedding 模型意味着全量重建索引。

## 性能关键参数

- **HNSW 的 M**：每个节点的最大连接数，越大精度越高但内存越大（一般 16-64）
- **HNSW 的 ef_construction**：建图时的搜索宽度（一般 200-500）
- **HNSW 的 ef_search**：查询时的搜索宽度，直接影响 recall vs latency
- **IVF 的 nprobe**：查询时搜索的簇数量

> 这些参数的调优本质上都是 **recall vs latency 的 tradeoff**，和混合检索中 BM25 + 向量的权重调优是类似的思路。

## 面试常见问题

- Q：向量数据库和传统数据库的本质区别？A：索引结构不同（B+树 vs ANN 索引），查询语义不同（精确匹配 vs 相似度搜索）
- Q：为什么不直接用 PostgreSQL + pgvector？A：小规模可以，但缺少专用 ANN 索引优化，百万级以上性能差距明显
- Q：HNSW 和 IVF 怎么选？A：数据频繁更新选 HNSW（增量友好），静态大数据集选 IVF（内存省）


