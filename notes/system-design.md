# 系统设计笔记

## 关键原则

1. **可扩展性 (Scalability)**: 系统应能随负载增长而水平扩展
2. **高可用 (High Availability)**: 通过冗余和故障转移保证服务可用性
3. **一致性 (Consistency)**: 根据场景选择强一致性或最终一致性
4. **低延迟 (Low Latency)**: 通过缓存、CDN、数据分区等手段降低响应时间

## 常见模式

- **负载均衡**: Round Robin, 加权轮询, 一致性哈希
- **缓存策略**: Cache-aside, Write-through, Write-back
- **数据库分区**: 水平分片 (Sharding), 垂直拆分
- **消息队列**: 解耦服务间通信，实现异步处理
