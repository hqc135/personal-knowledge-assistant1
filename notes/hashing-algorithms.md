<file_write file="hashing-algorithms.md">
# 哈希算法：一致性哈希、SimHash、局部敏感哈希 LSH

> 哈希不只是 HashMap 那么简单。这三个哈希算法分别解决分布式系统、文本去重、高维近似搜索三个不同领域的问题，但底层思想有很多共通之处。

## 一致性哈希（Consistent Hashing）

### 解决什么问题？

普通哈希分片：`node = hash(key) % N`。当 N 变化时（加减节点），几乎所有 key 都要重新映射。在分布式缓存场景下，这意味着缓存全部失效 → 雪崩。

一致性哈希的目标：节点变化时，只有 K/N 的 key 需要迁移（K 是总 key 数，N 是节点数）。

### 核心思想：哈希环

```
        0
       / \
     N3    N1
      |    |
     N2 ── ·
     
把节点和 key 都映射到 [0, 2^32) 的环上
key 顺时针找到的第一个节点就是它的归属
```

### 虚拟节点

问题：节点少的时候分布不均匀。

解决：每个物理节点映射多个虚拟节点（比如 150-200 个）到环上。

```python
class ConsistentHash:
    def __init__(self, nodes, replicas=150):
        self.ring = SortedDict()  # 有序字典
        for node in nodes:
            for i in range(replicas):
                virtual_key = hash(f"{node}#{i}")
                self.ring[virtual_key] = node
    
    def get_node(self, key):
        h = hash(key)
        # 找到第一个 >= h 的位置
        idx = self.ring.bisect_left(h)
        if idx == len(self.ring):
            idx = 0  # 绕回环的起点
        return self.ring.values()[idx]
```

### 实际应用

- **Memcached/Redis 集群**：客户端用一致性哈希决定 key 存哪个节点
- **CDN**：根据内容 URL 哈希决定缓存在哪个边缘节点
- **DynamoDB / Cassandra**：分区策略的基础

> 面试高频考点：一致性哈希加减节点时的数据迁移过程。加节点时只需要从后继节点迁移一部分数据；删节点时数据自然落到后继节点。

### 带权重的一致性哈希

不同机器性能不同怎么办？给性能强的机器分配更多虚拟节点。虚拟节点数量和权重成正比。

> 这里我一开始理解错了：以为一致性哈希能保证完美均匀。实际上即使有虚拟节点，负载也只是"大致均匀"。Google 的 Jump Consistent Hash 和 Maglev Hash 是更现代的方案，能做到更均匀的分布且内存开销更小。

## SimHash

### 解决什么问题？

判断两篇文档是否"近似相同"（near-duplicate detection）。Google 用它做网页去重。

核心性质：**相似的文档产生相似的哈希值**（汉明距离小）。这和传统哈希"一点变化输出完全不同"的雪崩效应恰好相反。

### 算法步骤

```
输入：文档 → 分词得到特征集合，每个特征有权重

1. 初始化 V = [0] * f  （f 是哈希位数，比如 64）
2. 对每个特征 feature：
   a. 计算 hash(feature) → f 位的二进制串
   b. 对 V 的每一位：
      - 如果 hash 的第 i 位是 1：V[i] += weight
      - 如果 hash 的第 i 位是 0：V[i] -= weight
3. 最终签名：V[i] > 0 → 1，否则 → 0
```

```python
def simhash(features_with_weights, f=64):
    v = [0] * f
    for feature, weight in features_with_weights:
        h = hash_to_bits(feature, f)  # 返回 f 位哈希
        for i in range(f):
            if h & (1 << i):
                v[i] += weight
            else:
                v[i] -= weight
    # 降维：向量 → 二进制指纹
    fingerprint = 0
    for i in range(f):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint
```

### 相似度判断

两个 SimHash 值的**汉明距离**（不同位的个数）反映文档差异：
- 汉明距离 ≤ 3（64位情况下）→ 认为是近似重复

汉明距离计算：`bin(hash1 ^ hash2).count('1')`

### 大规模去重的工程优化

64 位 SimHash，阈值 3 位。暴力比较 O(N²) 不可接受。

**分桶策略**：把 64 位分成若干段（比如 4 段各 16 位），如果两个文档汉明距离 ≤ 3，那么至少有一段是完全相同的（鸽巢原理）。

所以建 4 个索引表，每个表以 16 位为 key。查询时在 4 个表中分别查找，取并集作为候选集，再精确比较。

> 这个分桶思想和 LSH 的核心思想完全一致——通过多次哈希把相似的项映射到同一个桶。

## 局部敏感哈希 LSH（Locality-Sensitive Hashing）

### 解决什么问题？

高维空间中的近似最近邻搜索（ANN）。精确最近邻在高维下是 O(N) 或者受维度灾难影响，LSH 提供亚线性的近似解。

### 核心性质

一个哈希函数族 H 是 (d₁, d₂, p₁, p₂)-sensitive 的，如果对任意 x, y：
- 如果 dist(x, y) ≤ d₁，则 Pr[h(x) = h(y)] ≥ p₁
- 如果 dist(x, y) ≥ d₂，则 Pr[h(x) = h(y)] ≤ p₂

其中 d₁ < d₂，p₁ > p₂。

直觉：**近的点大概率哈希到一起，远的点大概率哈希到不同桶。**

### 不同距离度量对应的 LSH 族

| 距离度量 | LSH 方法 | 典型应用 |
|---------|---------|---------|
| Jaccard 距离 | MinHash | 文档相似度、推荐 |
| 余弦距离 | Random Hyperplane | 文本/图像嵌入 |
| 欧氏距离 | Random Projection (E2LSH) | 向量数据库 |
| 汉明距离 | Bit Sampling | SimHash 去重 |

### MinHash 详解（Jaccard 距离的 LSH）

两个集合 A, B 的 Jaccard 相似度：J(A,B) = |A∩B| / |A∪B|

MinHash 的神奇性质：

$$Pr[MinHash(A) = MinHash(B)] = J(A, B)$$

```python
def minhash_signature(document_set, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        h = hash_function_i  # 第 i 个哈希函数
        min_val = min(h(element) for element in document_set)
        signature.append(min_val)
    return signature
```

用 k 个哈希函数生成长度为 k 的签名，两个签名中相同位置相等的比例就是 Jaccard 相似度的无偏估计。

### LSH 的 AND/OR 放大

单个哈希函数的区分能力不够强。通过组合放大：

- **AND 放大**（b 个哈希都相同才算匹配）：降低假阳性，但也降低了真阳性
- **OR 放大**（任一组匹配就算）：提高召回率

实践中用 **b bands × r rows** 的结构：

```
签名分成 b 个 band，每个 band 有 r 行
两个文档在任意一个 band 中完全匹配 → 成为候选对

匹配概率：1 - (1 - s^r)^b
其中 s 是真实相似度
```

通过调整 b 和 r，可以得到一个 S 形曲线，在阈值附近急剧变化。

> 和 Transformer 中 multi-head attention 对比：LSH 的多组哈希函数类似于多头注意力——每个"头"从不同角度捕捉相似性，最后综合判断。实际上 Reformer 论文就是用 LSH 来近似 attention 的。

### 向量数据库中的 LSH

现在的向量数据库（Milvus、Pinecone、Weaviate）虽然更多用 HNSW 或 IVF-PQ，但 LSH 仍然是理解 ANN 的基础：

- **优点**：理论保证强，支持动态插入
- **缺点**：实际性能不如 HNSW（图方法），内存开销大
- **适用场景**：数据频繁更新、需要严格的近似比保证

## 三者的统一视角

这三个算法看似不同，但有一个共同的哲学：

**通过精心设计的哈希函数，把"相似性"编码到哈希值的"接近性"中。**

- 一致性哈希：key 的"位置相似性"映射到环上的"位置接近性"，保证拓扑变化时影响最小
- SimHash：文档的"内容相似性"映射到哈希值的"汉明距离接近性"
- LSH：高维向量的"距离相似性"映射到"同桶概率"

传统密码学哈希追求的是雪崩效应（一点变化，输出完全不同），而这三个算法恰恰追求**反雪崩**——相似的输入产生相似的输出。

## 工程实践中的选型

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 分布式缓存分片 | 一致性哈希 + 虚拟节点 | 节点变化时迁移量最小 |
| 网页/文档去重 | SimHash + 分桶 | 指纹紧凑（64bit），比较快 |
| 百万级向量相似搜索 | LSH 或 HNSW | LSH 理论保证好，HNSW 实际性能好 |
| 推荐系统候选召回 | MinHash + LSH | 适合集合型特征的相似度计算 |

## 面试常见追问

1. **一致性哈希的热点问题怎么解决？** 虚拟节点 + 如果单 key 热点则需要应用层缓存或拆分
2. **SimHash 对短文本效果好吗？** 不好，特征太少导致哈希值不稳定，短文本更适合用编辑距离或 embedding
3. **LSH 和 KD-Tree 的区别？** KD-Tree 在低维（<20）效果好，高维退化为线性扫描；LSH 在高维依然有效
4. **Bloom Filter 算不算 LSH？** 不算。Bloom Filter 是精确成员判断（有假阳性），不保留相似性信息

> 最后一个待深挖的问题：在 RAG（检索增强生成）系统中，embedding 检索通常用 HNSW。但如果数据量到了十亿级别，HNSW 的内存开销是否还能接受？这时候 LSH + 磁盘索引（比如 DiskANN）是不是更好的选择？和 B+ 树笔记中讨论的"内存 vs 磁盘"trade-off 是同一个问题。
