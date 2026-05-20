
# 微服务架构：服务发现、负载均衡、熔断降级

## ML 系统为什么要用微服务

一个完整的 ML 服务通常包含多个独立组件：

```
API Gateway → 特征服务 → 推理服务 (GPU)
                ↓              ↓
           特征存储(Redis)   模型存储(S3)
                ↓
           日志服务 → 监控/告警
```

每个组件的扩缩容需求不同：推理服务需要 GPU、特征服务需要高内存、API Gateway 需要高并发。单体架构无法独立扩展，微服务是自然选择。

> 但也别过度拆分。我见过把 tokenizer 和 model inference 拆成两个服务的，中间多了一次网络调用，延迟反而更高。拆分粒度要看实际瓶颈。

## 服务发现 (Service Discovery)

### 问题

微服务实例动态变化（扩缩容、故障重启），调用方不能硬编码 IP 地址。

### 两种模式

**客户端发现 (Client-side Discovery)**：
```
调用方 → 注册中心查询可用实例列表 → 自己选一个调用

优点：少一跳网络，客户端可以做智能路由
缺点：每种语言都要实现发现逻辑
代表：Eureka、Nacos
```

**服务端发现 (Server-side Discovery)**：
```
调用方 → 负载均衡器/代理 → 后端实例
                ↑
           注册中心

优点：客户端简单，只需知道代理地址
缺点：多一跳，代理可能成为瓶颈
代表：Kubernetes Service、AWS ALB、Consul + Envoy
```

### Kubernetes 中的服务发现

```yaml
# K8s Service 自动做服务发现 + 负载均衡
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  selector:
    app: llm-inference  # 匹配所有带这个 label 的 Pod
  ports:
    - port: 8080
      targetPort: 8080
  type: ClusterIP  # 集群内部访问
```

```python
# 其他服务只需要用 DNS 名称调用
import requests
# K8s 自动解析 inference-service 到可用 Pod
response = requests.post("http://inference-service:8080/predict", json=payload)
```

> 面试高频考点：K8s Service 的几种类型——ClusterIP（内部）、NodePort（节点端口暴露）、LoadBalancer（云厂商 LB）、Headless（不分配 ClusterIP，直接返回 Pod IP 列表，适合有状态服务）。

### 健康检查

服务发现的前提是知道哪些实例是健康的：

```yaml
# K8s 探针配置
livenessProbe:    # 不健康就重启
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30  # GPU 模型加载慢，给足启动时间
  periodSeconds: 10

readinessProbe:   # 不就绪就从 Service 摘除，不接流量
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 60  # 模型加载完才算 ready
  periodSeconds: 5
```

> 踩过的坑：LLM 推理服务启动很慢（加载模型到 GPU 可能要 2-3 分钟），`initialDelaySeconds` 设太短会导致 Pod 反复被杀重启。一定要根据实际模型加载时间设置。

## 负载均衡 (Load Balancing)

### 常见算法

```python
# 1. Round Robin（轮询）
# 简单，但不考虑实例负载差异
next_instance = instances[counter % len(instances)]
counter += 1

# 2. Weighted Round Robin（加权轮询）
# GPU 实例性能不同时有用（A100 vs V100）
# A100 权重 3，V100 权重 1

# 3. Least Connections（最少连接）
# 选当前活跃连接最少的实例
next_instance = min(instances, key=lambda i: i.active_connections)

# 4. Consistent Hashing（一致性哈希）
# 相同请求总是路由到同一实例，利于缓存
# 适合有本地缓存的推理服务
```

### ML 推理场景的特殊考虑

传统负载均衡假设每个请求处理时间差不多，但 LLM 推理不是这样：
- 短 query 可能 100ms，长生成可能 10s
- 不同模型占用不同 GPU 显存

**更好的策略：基于队列深度/GPU 利用率的负载均衡**

```python
# 自定义负载均衡：选 GPU 利用率最低的实例
def select_instance(instances):
    # 每个实例定期上报 GPU 利用率和队列深度
    metrics = [(i, get_metrics(i)) for i in instances]
    # 综合考虑 GPU 利用率和排队请求数
    return min(metrics, key=lambda x: x[1]['gpu_util'] * 0.7 + x[1]['queue_depth'] * 0.3)
```

> 这和 vLLM/TGI 的 continuous batching 有关联：即使实例内部做了动态 batching，外部负载均衡仍然需要感知每个实例的实际负载，否则可能把请求都打到一个已经满载的实例上。

## 熔断降级 (Circuit Breaker & Degradation)

### 熔断器模式

灵感来自电路断路器：当下游服务故障率过高时，直接"断开"，不再发送请求，避免级联故障。

```
三种状态：
Closed (正常) → 失败率超阈值 → Open (熔断)
Open (熔断) → 超时后 → Half-Open (试探)
Half-Open → 试探成功 → Closed
Half-Open → 试探失败 → Open
```

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.state = "CLOSED"
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                return self.fallback()  # 直接走降级逻辑
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            return self.fallback()
    
    def fallback(self):
        # 降级逻辑
        return {"result": "default_recommendation", "degraded": True}
```

### ML 系统中的降级策略

| 场景 | 正常逻辑 | 降级逻辑 |
|------|---------|---------|
| 推荐系统 | 个性化模型推理 | 返回热门榜单（预计算） |
| LLM 对话 | 大模型生成 | 切换到小模型 / 返回模板回复 |
| 特征服务挂了 | 实时特征 | 用离线特征兜底 |
| Embedding 服务 | 实时计算 | 返回缓存的旧 embedding |

> 个人理解：降级不是"出错了随便返回个东西"，而是预先设计好的 Plan B。好的降级策略应该让用户几乎感知不到服务异常。

### 限流 (Rate Limiting)

和熔断配合使用，保护下游服务：

```python
# 令牌桶算法
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate          # 每秒生成的令牌数
        self.capacity = capacity  # 桶容量
        self.tokens = capacity
        self.last_time = time.time()
    
    def allow(self):
        now = time.time()
        # 补充令牌
        self.tokens = min(
            self.capacity,
            self.tokens + (now - self.last_time) * self.rate
        )
        self.last_time = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

# GPU 推理服务限流：每秒最多处理 100 个请求
limiter = TokenBucket(rate=100, capacity=200)  # 允许短暂突发到 200
```

> 面试追问：令牌桶 vs 漏桶的区别？令牌桶允许突发流量（桶里有积攒的令牌），漏桶严格匀速输出。ML 推理场景用令牌桶更合适，因为 GPU batching 天然适合处理突发（一个 batch 处理多个请求）。

## 可观测性 (Observability)

微服务架构下，问题定位变得困难。三大支柱：

### 1. Metrics（指标）

```python
# Prometheus 指标示例
from prometheus_client import Histogram, Counter, Gauge

inference_latency = Histogram(
    'inference_latency_seconds',
    'Model inference latency',
    labelnames=['model_name', 'model_version'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization')
request_count = Counter('inference_requests_total', 'Total requests',
                       labelnames=['status'])  # success/failure/degraded
```

### 2. Tracing（链路追踪）

```
用户请求 → API Gateway (trace_id: abc123)
    → 特征服务 (span_id: 001, parent: root)
    → 推理服务 (span_id: 002, parent: root)
        → 模型加载 (span_id: 003, parent: 002)
        → 推理计算 (span_id: 004, parent: 002)
    → 后处理 (span_id: 005, parent: root)
```

### 3. Logging（日志）

结构化日志 + 关联 trace_id，方便排查问题。

## 面试常见问题总结

1. **服务间通信选 gRPC 还是 REST？**
   - gRPC：高性能（protobuf 序列化 + HTTP/2 多路复用），适合内部服务间通信
   - REST：简单通用，适合对外 API
   - ML 推理服务内部通常用 gRPC（Triton 就是 gRPC 接口）

2. **如何处理分布式事务？**
   - ML 系统通常不需要强一致性事务
   - 用 Saga 模式或最终一致性（通过消息队列）

3. **服务网格 (Service Mesh) 是什么？**
   - Sidecar 代理（如 Envoy）处理所有网络通信
   - 应用代码不需要关心服务发现、负载均衡、熔断等
   - Istio 是最流行的实现

> 和消息队列的关联：微服务间的异步通信通常通过消息队列实现。同步调用用 gRPC/REST + 服务发现 + 负载均衡，异步调用用 Kafka/Redis Stream。两种模式各有适用场景，很多系统是混合使用的。