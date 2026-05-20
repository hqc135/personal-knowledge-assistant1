
# 树结构：B+树、LSM-Tree、跳表在数据库中的应用

> 这三个数据结构是存储引擎的核心，面试数据库/存储方向几乎必问。理解它们的关键不是背结构定义，而是搞清楚"为什么这个场景选这个结构"。

## B+ 树

### 为什么不用二叉树？

磁盘 I/O 的单位是页（通常 4KB/16KB）。二叉树一个节点存一个 key，树高 log₂N，N=1亿时树高约 27 层 = 27 次磁盘 I/O。B+ 树通过增大扇出（fanout）降低树高。

一个节点存 m 个 key（m 可能是几百），树高变成 log_m(N)。m=200, N=1亿，树高 ≈ 4。**4 次 I/O vs 27 次 I/O，这就是 B+ 树存在的意义。**

### B+ 树 vs B 树

| 特性 | B 树 | B+ 树 |
|------|------|-------|
| 数据存储位置 | 所有节点 | 仅叶子节点 |
| 叶子节点链表 | 无 | 有（支持范围查询） |
| 内部节点大小 | 较大（含数据） | 较小（仅索引） |
| 范围查询效率 | 差（需要中序遍历） | 好（顺序扫描叶子链表） |

> 面试高频考点：为什么 MySQL InnoDB 用 B+ 树而不是 B 树？答案就是范围查询 + 内部节点更小意味着更大的扇出。

### 关键操作复杂度

- 查找：O(log_m N)，每层一次二分查找 O(log m)，总共 O(log_m N × log m) = O(log N)
- 插入：可能触发分裂（split），从叶子向上传播
- 删除：可能触发合并（merge）或重分配

```
插入伪代码：
1. 找到目标叶子节点
2. 如果叶子未满，直接插入
3. 如果叶子已满：
   a. 分裂为两个节点
   b. 中间 key 上提到父节点
   c. 如果父节点也满了，递归分裂
```

### InnoDB 中的 B+ 树

- 聚簇索引（主键索引）：叶子节点存完整行数据
- 二级索引：叶子节点存主键值（所以二级索引查询可能需要"回表"）
- 页大小默认 16KB，一个内部节点大约能存 1000+ 个指针

> 这里我一开始理解错了：以为二级索引的叶子存的是行数据的物理地址。实际上 InnoDB 存的是主键值，这样页分裂时不需要更新所有二级索引。这个设计和"间接层"的思想一致——计算机科学中的大多数问题都可以通过加一层间接来解决。

## LSM-Tree（Log-Structured Merge Tree）

### 核心思想：把随机写变成顺序写

磁盘（包括 SSD）的顺序写性能远优于随机写。LSM-Tree 的策略：

1. 写入先进内存中的 MemTable（通常是红黑树或跳表）
2. MemTable 满了后 flush 成磁盘上的 SSTable（Sorted String Table）
3. 后台定期做 Compaction，合并多个 SSTable

```
写入流程：
Client Write → WAL（持久化保证）→ MemTable（内存排序结构）
                                        ↓ flush
                                    SSTable (L0)
                                        ↓ compaction
                                    SSTable (L1, L2, ...)
```

### 读放大 vs 写放大 vs 空间放大

这是 LSM-Tree 的核心 trade-off（RUM 猜想的体现）：

- **写放大（Write Amplification）**：一条数据在 compaction 过程中被反复读写。Leveled compaction 写放大约 10-30x
- **读放大（Read Amplification）**：读一个 key 可能要查多层 SSTable。用 Bloom Filter 优化
- **空间放大（Space Amplification）**：同一个 key 的多个版本同时存在

> 和 B+ 树对比记忆：B+ 树是读优化结构（原地更新），LSM-Tree 是写优化结构（追加写入）。这就是为什么 MySQL 用 B+ 树（OLTP 读多写少），而 Cassandra/RocksDB/LevelDB 用 LSM-Tree（写密集场景）。

### Compaction 策略

**Size-Tiered Compaction（STCS）**：
- 同层的 SSTable 大小相近时合并
- 写放大低，但空间放大大（最坏 2x）
- Cassandra 默认策略

**Leveled Compaction（LCS）**：
- 每层容量是上一层的 10 倍
- 同层 SSTable key 范围不重叠
- 读放大低，写放大高
- RocksDB/LevelDB 默认策略

### Bloom Filter 的作用

读取时需要判断"这个 key 在不在某个 SSTable 里"。Bloom Filter 提供 O(1) 的否定判断：

- 说"不在"→ 一定不在（避免无效 I/O）
- 说"在"→ 可能在（需要实际读取确认）

> 这里和哈希算法笔记中的 LSH 有相似之处——都是用概率数据结构换取效率，接受一定的假阳性率。

## 跳表（Skip List）

### 为什么用跳表？

跳表是一种概率性数据结构，提供 O(log N) 的查找/插入/删除，和平衡二叉树性能相当，但实现简单得多。

Redis 的 Sorted Set 和 LevelDB 的 MemTable 都用了跳表。

### 结构

```
Level 3:  1 ─────────────────────── 9
Level 2:  1 ────── 4 ────────────── 9
Level 1:  1 ── 3 ─ 4 ── 6 ── 7 ── 9
Level 0:  1  2  3  4  5  6  7  8  9  (完整链表)
```

每个节点以概率 p（通常 1/2 或 1/4）决定是否提升到上一层。期望层数 O(log N)。

### 查找过程

```python
def search(skip_list, target):
    current = skip_list.header
    for level in range(skip_list.max_level, -1, -1):
        while current.forward[level] and current.forward[level].key < target:
            current = current.forward[level]
    current = current.forward[0]
    return current if current and current.key == target else None
```

从最高层开始，尽可能向右走，走不动了就下降一层。

### 为什么 Redis 选跳表而不是红黑树？

这是经典面试题。Redis 作者 antirez 的原话大意：

1. 实现简单，代码好维护
2. 范围查询天然支持（链表顺序遍历）
3. 通过调整 p 值可以灵活平衡时间和空间
4. 并发友好——局部修改不需要像红黑树那样做全局旋转

> 面试高频考点：跳表的期望空间复杂度是 O(N)（等比数列求和 N + N/2 + N/4 + ... ≈ 2N），时间复杂度 O(log N)。

## 三者在存储引擎中的协作

一个典型的 LSM-Tree 存储引擎（如 RocksDB）中：

```
MemTable (跳表) → flush → SSTable (排序文件，内部用 B+ 树索引块)
                              ↓
                    Compaction 合并多个 SSTable
```

- **跳表**：MemTable 的实现，支持高效的内存有序插入和范围扫描
- **B+ 树**：SSTable 内部的索引结构，或者直接作为存储引擎（InnoDB）
- **LSM-Tree**：整体架构，协调内存和磁盘之间的数据流动

## 面试常见问题清单

1. B+ 树的扇出怎么计算？（页大小 / (key大小 + 指针大小)）
2. LSM-Tree 的 compaction 会不会影响前台读写？（会，所以有 rate limiter）
3. 跳表如何保证平衡？（概率保证，不需要显式平衡操作）
4. 为什么 SSD 上 LSM-Tree 依然有优势？（SSD 随机写有写放大，且需要 GC）

> 最后的思考：这三个结构的选择本质上是在 read/write/space 三个维度做 trade-off。没有银弹，只有适合场景的选择。这个思想和分布式系统中的 CAP 定理异曲同工——你不可能同时优化所有维度。