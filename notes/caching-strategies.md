
# 缓存策略：Redis 数据结构、淘汰策略、缓存穿透/雪崩

## 为什么 ML 系统特别需要缓存

1. **模型推理贵**：一次 LLM 推理可能要几百毫秒 + GPU 资源，相同 query 缓存结果直接返回
2. **特征查询频繁**：推荐系统每次请求要查几百个特征，不缓存数据库扛不住
3. **Embedding 复用**：相同文本的 embedding 不需要重复计算

## Redis 核心数据结构及 ML 场景应用

### String

最基础，用于缓存推理结果：

```redis
# 缓存模型推理结果，TTL 1小时
SET inference:model_v2:hash(input) '{"output": "...", "score": 0.95}' EX 3600

# 模型版本标记
SET current_model_version "v2.3.1"
```

### Hash

适合存储结构化特征：

```redis
# 用户特征向量
HSET user:12345 age 28 gender "M" click_rate 0.15 last_active 1684000000

# 一次取多个特征
HMGET user:12345 age click_rate last_active
```

> 面试高频考点：Hash vs 多个 String 的选择。Hash 省内存（共享 key 的开销），适合字段数 < 128 且值较小的场景（ziplist 编码）。但 Hash 不能给单个 field 设 TTL。

### Sorted Set (ZSet)

适合排行榜、Top-K 推荐结果缓存：

```redis
# 缓存推荐结果，score 是推荐分数
ZADD recommend:user:12345 0.95 "item_A" 0.87 "item_B" 0.82 "item_C"

# 取 Top 10
ZREVRANGE recommend:user:12345 0 9 WITHSCORES

# 带过期的实现：用 score 存时间戳，定期清理
ZREMRANGEBYSCORE recommend:user:12345 -inf (now-3600)
```

### Bitmap / HyperLogLog

用于特征工程中的统计：

```redis
# 用户今天是否活跃（bitmap，省内存）
SETBIT user_active:2024-01-15 12345 1

# 统计日活
BITCOUNT user_active:2024-01-15

# 近似去重计数（HyperLogLog，12KB 固定内存）
PFADD unique_queries:2024-01-15 "query_hash_1" "query_hash_2"
PFCOUNT unique_queries:2024-01-15
```

## 淘汰策略

当 Redis 内存达到 `maxmemory` 时，需要淘汰策略：

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| noeviction | 不淘汰，写入报错 | 不能丢数据的场景 |
| allkeys-lru | 所有 key 中淘汰最近最少使用 | **通用推荐** |
| allkeys-lfu | 所有 key 中淘汰最不经常使用 | 热点数据明显的场景 |
| volatile-lru | 只淘汰设了 TTL 的 key（LRU） | 混合持久+缓存数据 |
| volatile-ttl | 淘汰 TTL 最短的 | 希望快过期的先走 |
| allkeys-random | 随机淘汰 | 访问模式均匀 |

> 这里我一开始理解错了：Redis 的 LRU 不是精确 LRU，而是近似 LRU（随机采样 N 个 key，淘汰其中最旧的）。`maxmemory-samples` 默认 5，增大可以更精确但更慢。LFU 也是近似的，用 Morris counter 做频率估计。

### ML 场景的选择建议

- **推理结果缓存**：`allkeys-lfu`，热门 query 被反复问到，LFU 能保留高频结果
- **特征缓存**：`volatile-lru`，特征有时效性，设 TTL + LRU 淘汰冷特征
- **Embedding 缓存**：`allkeys-lru`，最近用过的 embedding 更可能再次被用到

## 缓存三大经典问题

### 缓存穿透 (Cache Penetration)

**问题**：查询一个根本不存在的 key，缓存永远 miss，每次都打到数据库/模型。

恶意攻击场景：大量请求不存在的 user_id 的推荐结果。

**解决方案**：

```python
# 方案 1：缓存空值
def get_recommendation(user_id):
    result = redis.get(f"rec:{user_id}")
    if result == "NULL_PLACEHOLDER":
        return None  # 已知不存在，直接返回
    if result is not None:
        return json.loads(result)
    
    # 缓存 miss，查数据库
    db_result = db.query_user(user_id)
    if db_result is None:
        # 缓存空值，短 TTL 防止占用太多内存
        redis.set(f"rec:{user_id}", "NULL_PLACEHOLDER", ex=300)
        return None
    
    redis.set(f"rec:{user_id}", json.dumps(db_result), ex=3600)
    return db_result
```

```python
# 方案 2：布隆过滤器（Bloom Filter）
# 预先把所有合法 user_id 加入布隆过滤器
bloom = BloomFilter(capacity=10_000_000, error_rate=0.001)
for uid in all_valid_user_ids:
    bloom.add(uid)

def get_recommendation(user_id):
    if user_id not in bloom:  # O(1)，可能有假阳性但无假阴性
        return None  # 一定不存在，直接拦截
    # 正常查缓存 → 数据库
    ...
```

> 布隆过滤器的原理和 LSH (Locality Sensitive Hashing) 有相似之处——都是用多个 hash 函数做近似判断。但布隆过滤器判断存在性，LSH 判断相似性。

### 缓存雪崩 (Cache Avalanche)

**问题**：大量 key 同时过期，瞬间所有请求打到后端。

典型场景：凌晨批量刷新推荐结果缓存，所有 key 设了相同 TTL，同时过期。

**解决方案**：

```python
import random

# 方案 1：TTL 加随机抖动
base_ttl = 3600
jitter = random.randint(0, 600)  # 0-10 分钟随机
redis.set(key, value, ex=base_ttl + jitter)

# 方案 2：永不过期 + 异步刷新
# 缓存不设 TTL，后台线程定期刷新
# 适合特征缓存这种可以容忍短暂过期的场景

# 方案 3：多级缓存
# L1: 本地缓存 (进程内, 如 Python dict / LRU cache)
# L2: Redis
# L3: 数据库/模型推理
```

> 面试追问：如果 Redis 整个挂了怎么办？这就不是雪崩而是缓存击穿的极端情况了。需要：1) Redis 高可用（Sentinel/Cluster）；2) 本地缓存兜底；3) 限流降级。

### 缓存击穿 (Cache Breakdown / Hotspot Invalid)

**问题**：某个热点 key 过期的瞬间，大量并发请求同时穿透到后端。

和雪崩的区别：雪崩是大量 key 同时过期，击穿是单个热点 key 过期。

```python
# 方案：互斥锁（分布式锁）
def get_hot_data(key):
    result = redis.get(key)
    if result is not None:
        return result
    
    # 尝试获取锁
    lock_key = f"lock:{key}"
    if redis.set(lock_key, "1", nx=True, ex=10):  # SETNX
        try:
            # 只有一个请求去重建缓存
            result = expensive_compute(key)
            redis.set(key, result, ex=3600)
            return result
        finally:
            redis.delete(lock_key)
    else:
        # 没拿到锁，等一下重试
        time.sleep(0.1)
        return get_hot_data(key)  # 递归重试
```

> 踩过的坑：上面的分布式锁有个问题——如果持锁的进程挂了，锁会在 10s 后自动释放，但这 10s 内其他请求都在 sleep 重试。生产环境建议用 Redlock 或者"逻辑过期"方案（缓存不设物理 TTL，value 里带逻辑过期时间，过期后异步刷新，旧值继续用）。

## ML 系统中的缓存设计模式

### 模式 1：Embedding Cache

```python
# 文本 embedding 缓存，避免重复调用 embedding 模型
def get_embedding(text):
    cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
    cached = redis.get(cache_key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)
    
    embedding = embedding_model.encode(text)  # 耗时操作
    redis.set(cache_key, embedding.tobytes(), ex=86400)
    return embedding
```

### 模式 2：特征存储 (Feature Store) 缓存层

```
离线特征计算 (Spark) → 写入 Redis Hash
在线推理时 → 从 Redis 批量读取特征 → 拼接 → 送入模型

# 批量读取优化：Pipeline
pipe = redis.pipeline()
for uid in user_ids:
    pipe.hmget(f"user:{uid}", "feat1", "feat2", "feat3")
results = pipe.execute()  # 一次网络往返拿所有结果
```

> 这里和消息队列的关联：特征更新通常通过 Kafka 消费实时事件，计算后写入 Redis。Kafka 保证数据不丢，Redis 保证读取快。两者配合构成实时特征 pipeline。

### 模式 3：模型推理结果缓存 + 版本管理

```python
# key 中包含模型版本，模型更新时自动失效
model_version = redis.get("current_model_version")  # "v2.3"
cache_key = f"pred:{model_version}:{hash(input)}"

# 模型更新时只需要更新版本号，旧缓存自然过期
# 不需要手动清理所有旧缓存
```

## 面试常见追问

1. **Redis 单线程为什么还这么快？**
   - 内存操作、IO 多路复用（epoll）、避免上下文切换、高效数据结构

2. **Redis Cluster 如何分片？**
   - 16384 个 hash slot，CRC16(key) % 16384 决定 slot，每个节点负责一部分 slot

3. **缓存和数据库一致性怎么保证？**
   - Cache Aside Pattern：先更新 DB，再删缓存（不是更新缓存）
   - 延迟双删：删缓存 → 更新 DB → 延迟再删一次缓存
   - 最终一致性：通过消息队列异步同步

4. **本地缓存 vs Redis？**
   - 本地缓存：无网络开销，但多实例间不一致
   - Redis：一致性好，但有网络 RTT（通常 0.5-1ms）
   - 最佳实践：L1 本地 + L2 Redis，热点数据两层都有
