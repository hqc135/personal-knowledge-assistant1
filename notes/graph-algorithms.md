
# 图算法：最短路径、PageRank、GNN 入门

> 写这篇笔记的时候发现图算法是面试里出现频率极高的一类题，尤其是最短路径和 PageRank，GNN 则是系统设计面试里偶尔会被问到的加分项。

## 最短路径算法

### Dijkstra

核心思想：贪心。每次从未访问节点中选距离源点最近的，松弛其邻居。

```python
import heapq

def dijkstra(graph, src):
    dist = {v: float('inf') for v in graph}
    dist[src] = 0
    pq = [(0, src)]  # (距离, 节点)
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:  # 已经有更优解，跳过
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

时间复杂度：O((V+E) log V)，用优先队列的情况下。

**限制：不能处理负权边。** 这里我一开始理解错了——以为只要没有负环就行，其实 Dijkstra 的贪心假设是"已确定的最短距离不会再被更新"，负权边会破坏这个假设。

### Bellman-Ford

能处理负权边，还能检测负环。核心是对所有边松弛 V-1 次：

```
for i in range(V - 1):
    for (u, v, w) in edges:
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
```

第 V 次还能松弛 → 存在负环。时间 O(VE)，比 Dijkstra 慢但更通用。

### Floyd-Warshall

全源最短路，DP 思想：`dp[k][i][j]` 表示经过前 k 个中间节点时 i→j 的最短路。

状态转移：`dp[k][i][j] = min(dp[k-1][i][j], dp[k-1][i][k] + dp[k-1][k][j])`

时间 O(V³)，空间可以压缩到 O(V²)。适合节点少但需要所有点对距离的场景。

> 面试高频考点：三种算法的适用场景对比，尤其是"什么时候用哪个"。

## PageRank

Google 早期核心算法。把 Web 看成有向图，页面重要性由"谁链接了你"决定。

### 核心公式

$$PR(v) = \frac{1-d}{N} + d \sum_{u \in B(v)} \frac{PR(u)}{L(u)}$$

- d：阻尼因子（通常 0.85），表示用户继续点击链接的概率
- B(v)：所有指向 v 的页面集合
- L(u)：页面 u 的出链数
- N：总页面数

直觉理解：一个随机冲浪者，85% 概率点击当前页面的某个链接，15% 概率随机跳转到任意页面。PageRank 就是这个马尔可夫链的稳态分布。

### 迭代计算

```python
def pagerank(graph, d=0.85, iterations=100):
    N = len(graph)
    pr = {node: 1/N for node in graph}
    
    for _ in range(iterations):
        new_pr = {}
        for v in graph:
            rank_sum = sum(pr[u] / out_degree(u) for u in in_neighbors(v))
            new_pr[v] = (1 - d) / N + d * rank_sum
        pr = new_pr
    return pr
```

收敛条件一般看 L1 范数变化 < ε。

> 这个和 Transformer 的注意力机制有异曲同工之处——都是通过"邻居的加权投票"来决定当前节点/token 的表示。区别是 attention 的权重是学出来的，PageRank 的权重是结构决定的。

### 工程上的坑

- Dead ends（没有出链的节点）：会导致概率"泄漏"，解决方案是让 dead end 等概率链接到所有页面
- Spider traps（自环或小环）：阻尼因子 d < 1 就是为了解决这个
- 大规模计算：MapReduce 天然适合，每轮迭代就是一次 map（分发 PR 值）+ reduce（汇总）

## 图神经网络 GNN 入门

### 核心思想：消息传递（Message Passing）

GNN 的本质是让每个节点聚合邻居的信息来更新自己的表示：

```
h_v^(k) = UPDATE(h_v^(k-1), AGGREGATE({h_u^(k-1) : u ∈ N(v)}))
```

这其实就是 PageRank 思想的泛化——PageRank 是固定的聚合规则，GNN 把聚合函数参数化了。

### GCN（Graph Convolutional Network）

Kipf & Welling 2017 的经典公式：

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

- $\tilde{A} = A + I$（加自环）
- $\tilde{D}$：$\tilde{A}$ 的度矩阵
- W：可学习参数

直觉：对称归一化的邻接矩阵做特征传播，再过一个线性变换 + 非线性激活。

### GAT（Graph Attention Network）

和 GCN 的区别：聚合邻居时不是等权的，而是用 attention 学习权重。

$$\alpha_{ij} = \text{softmax}_j(e_{ij}), \quad e_{ij} = \text{LeakyReLU}(a^T [Wh_i \| Wh_j])$$

> 和 Transformer 的 self-attention 对比记忆：Transformer 是全连接图上的 attention，GAT 是稀疏图上的 attention。GAT 只在边存在的地方计算 attention，计算量和边数成正比。

### GNN 的局限

- **过平滑（Over-smoothing）**：层数太多时所有节点表示趋同，这是因为多次消息传递相当于在图上做扩散
- **表达力上界**：标准 GNN 的判别力不超过 1-WL test（Weisfeiler-Leman 图同构测试）
- 对长距离依赖建模差——和 RNN 的梯度消失问题类似

## 交叉引用与面试要点

1. PageRank 的迭代计算本质是**幂迭代法**求矩阵最大特征向量，和 PCA 中求协方差矩阵特征向量的方法一脉相承
2. GNN 的消息传递框架和**分布式系统中的 gossip 协议**思想类似——都是通过局部通信达到全局一致
3. 图的最短路径算法在**微服务路由、网络 SDN 控制面**中有直接应用

| 算法 | 适用场景 | 时间复杂度 | 能否处理负权 |
|------|---------|-----------|-------------|
| Dijkstra | 单源、非负权 | O((V+E)logV) | ❌ |
| Bellman-Ford | 单源、有负权 | O(VE) | ✅ |
| Floyd-Warshall | 全源 | O(V³) | ✅（无负环） |

> 最后一个自己的疑问：GNN 在工业界的 serving 延迟问题怎么解决？训练时可以 batch 整个子图，但推理时如果图在动态变化（比如社交网络），邻居采样策略（GraphSAGE 的做法）是不是唯一的出路？后续再深挖。

