
# 消息队列在 ML Pipeline 中的应用：Kafka、Redis Stream

## 为什么 ML Pipeline 需要消息队列

ML 系统不只是一个模型，而是一整条流水线：

```
数据采集 → 特征工程 → 模型推理 → 后处理 → 结果存储 → 监控反馈
```

各环节速度不同、可能失败、需要解耦。消息队列解决的核心问题：
1. **异步解耦**：生产者和消费者独立扩缩容
2. **削峰填谷**：突发流量不会打垮下游推理服务
3. **可靠性**：消息持久化，失败可重试
4. **顺序保证**：某些场景需要保序（如时序特征）

> 面试高频考点：为什么不直接用 HTTP 同步调用？答：推理服务有 GPU 资源限制，并发能力有限，需要队列做缓冲和流控。这和模型服务部署中的 batching 策略是配合使用的。

## Kafka 基础架构

```
Producer → [Topic: ml-requests]
              ├── Partition 0 → Consumer Group A (inference-worker-0)
              ├── Partition 1 → Consumer Group A (inference-worker-1)
              └── Partition 2 → Consumer Group A (inference-worker-2)
```

### 核心概念

- **Topic**：逻辑上的消息分类
- **Partition**：Topic 的物理分片，是并行度的基本单位
- **Consumer Group**：同组内每个 consumer 消费不同 partition，实现负载均衡
- **Offset**：每条消息在 partition 内的位置，consumer 自己维护消费进度

### Kafka 在 ML Pipeline 中的典型用法

```python
# 推理请求入队
producer.send('inference-requests', {
    'request_id': 'uuid-xxx',
    'model_name': 'llm-v2',
    'input': 'user query...',
    'timestamp': time.time(),
    'priority': 'high'
})

# 推理 worker 消费
consumer = KafkaConsumer(
    'inference-requests',
    group_id='inference-workers',
    auto_offset_reset='latest',  # 推理场景通常不需要历史消息
    enable_auto_commit=False      # 手动 commit，确保处理完才确认
)

for msg in consumer:
    result = model.predict(msg.value['input'])
    # 结果写入另一个 topic
    producer.send('inference-results', {
        'request_id': msg.value['request_id'],
        'output': result
    })
    consumer.commit()  # 处理完再 commit
```

> 这里我一开始理解错了：`enable_auto_commit=False` 不是说不 commit，而是手动控制 commit 时机。如果 auto commit，消息拉下来还没处理完就 commit 了，worker 挂了消息就丢了。

### Kafka 的优势场景

1. **训练数据流**：实时数据采集 → Kafka → 特征计算 → 训练数据集
2. **A/B 测试日志**：推理结果 + 用户反馈 → Kafka → 离线分析
3. **模型监控**：预测结果 → Kafka → 数据漂移检测
4. **高吞吐场景**：百万级 QPS 的特征日志

## Redis Stream

Redis 5.0 引入的数据结构，轻量级消息队列。

### 基本操作

```redis
# 生产消息
XADD ml-requests * model "gpt" input "hello" priority "high"
# 返回: "1684000000000-0" (时间戳-序号)

# 创建消费者组
XGROUP CREATE ml-requests inference-group $ MKSTREAM

# 消费（阻塞读取）
XREADGROUP GROUP inference-group worker-1 COUNT 10 BLOCK 5000 STREAMS ml-requests >

# 确认处理完成
XACK ml-requests inference-group "1684000000000-0"

# 查看未确认的消息（pending）
XPENDING ml-requests inference-group
```

### Redis Stream vs Kafka

| 维度 | Kafka | Redis Stream |
|------|-------|-------------|
| 吞吐量 | 极高（百万/s） | 中等（十万/s） |
| 持久化 | 磁盘，可保留很久 | 内存为主，可 AOF |
| 消息保留 | 按时间/大小策略 | 需手动 XTRIM |
| 运维复杂度 | 高（ZooKeeper/KRaft） | 低 |
| 消息大小 | 适合大消息 | 适合小消息 |
| 适用场景 | 大规模数据流 | 轻量级任务队列 |

> 个人理解：如果你的 ML pipeline 已经用了 Redis 做缓存（比如特征缓存），加个 Stream 做轻量队列很自然，不用额外引入 Kafka。但如果是大规模训练数据流或需要长期保留的日志，Kafka 更合适。

## ML Pipeline 中的具体模式

### 模式 1：推理请求缓冲 + 动态 Batching

```
API Server → Redis Stream → Batch Collector → GPU Inference
                                    ↓
                            每 50ms 或凑够 32 条
                            打包成一个 batch 送推理
```

```python
# Batch Collector 伪代码
batch = []
last_flush = time.time()

while True:
    msgs = redis.xreadgroup(..., COUNT=32, BLOCK=50)
    batch.extend(msgs)
    
    if len(batch) >= 32 or time.time() - last_flush > 0.05:
        results = model.batch_predict([m['input'] for m in batch])
        for msg, result in zip(batch, results):
            redis.set(f"result:{msg['request_id']}", result)
            redis.xack(...)
        batch = []
        last_flush = time.time()
```

> 这个模式和 Triton 的 Dynamic Batching 思路一样，只是把调度逻辑放到了应用层。适合不用 Triton 但又想做 batching 的场景。

### 模式 2：特征计算 Pipeline（Kafka Streams）

```
用户行为事件 → Kafka Topic A
                    ↓ (Kafka Streams / Flink)
              实时特征计算（滑动窗口聚合）
                    ↓
              Kafka Topic B → 写入特征存储 (Redis/DynamoDB)
                    ↓
              推理服务读取最新特征
```

### 模式 3：Dead Letter Queue (DLQ)

推理失败的请求不能丢，放入 DLQ 后续重试或人工排查：

```python
try:
    result = model.predict(input)
except Exception as e:
    # 重试 3 次后放入 DLQ
    if retry_count >= 3:
        producer.send('inference-dlq', {
            'original_msg': msg,
            'error': str(e),
            'retry_count': retry_count
        })
```

## 踩坑记录

1. **Consumer Rebalance 风暴**：Kafka consumer group 频繁加入/退出会触发 rebalance，期间所有 consumer 暂停消费。ML worker 如果推理时间长（比如 LLM 生成），容易超时被踢出 → 恶性循环。解决：增大 `max.poll.interval.ms`，或用 cooperative rebalance。

2. **消息积压监控**：consumer lag 是最重要的监控指标。如果 lag 持续增长，说明推理速度跟不上请求速度，需要扩容 worker 或限流。

3. **消息顺序 vs 并行**：同一用户的请求可能需要保序（多轮对话），用 user_id 作为 partition key 保证同一用户的消息落在同一 partition。

> 和缓存策略的关联：消息队列 + 缓存经常配合使用。比如推理结果先写缓存（Redis），再异步通过消息队列通知下游。这样 API 可以先返回"处理中"，客户端轮询缓存拿结果。