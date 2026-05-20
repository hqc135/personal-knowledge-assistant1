
# 模型服务部署：vLLM、TGI、Triton Inference Server

## 核心挑战

LLM 推理和传统模型推理完全不同：
1. **自回归生成**：每个 token 依赖前面所有 token，无法一次性并行出结果
2. **KV Cache 显存爆炸**：长序列的 KV Cache 占用巨大且动态变化
3. **吞吐 vs 延迟的权衡**：batch 越大吞吐越高，但单请求延迟也会增加
4. **请求长度不一**：不同请求的输入/输出长度差异很大，静态 batching 浪费严重

## vLLM：PagedAttention 的革命

vLLM 的核心创新是 **PagedAttention**，灵感来自操作系统的虚拟内存分页。

### 传统 KV Cache 的问题

```
传统方式：为每个请求预分配最大长度的连续显存
请求 A (实际 100 tokens, 预分配 2048): [used: 100][wasted: 1948]
请求 B (实际 500 tokens, 预分配 2048): [used: 500][wasted: 1548]

显存利用率极低，大量内部碎片
```

### PagedAttention 的解决方案

```
将 KV Cache 分成固定大小的 block（类似内存页，通常 16 tokens）
逻辑上连续的 KV Cache 物理上可以不连续

Block Table (类似页表):
请求 A: [Block 7] → [Block 3] → [Block 12] → ...
请求 B: [Block 1] → [Block 9] → [Block 5] → ...

好处：
1. 按需分配，无内部碎片
2. 不同请求可以共享 block（比如相同的 system prompt）
3. 显存利用率从 ~60% 提升到 >95%
```

> 这里和操作系统的虚拟内存管理几乎一模一样：逻辑地址连续、物理地址不连续、通过页表映射。面试时可以用这个类比来解释。

### Continuous Batching

传统 static batching：一个 batch 里所有请求必须等最长的那个结束才能处理下一批。

vLLM 的 continuous batching（也叫 iteration-level scheduling）：
- 每个 decode step 都可以加入新请求或移除已完成的请求
- 大幅提升 GPU 利用率

```python
# 伪代码
while requests_pending:
    # 每一步都重新调度
    active_batch = scheduler.select_requests()
    
    for req in finished_requests:
        active_batch.remove(req)
        yield req.output
    
    for req in waiting_queue:
        if has_memory():
            active_batch.add(req)
    
    # 一次 forward 处理所有 active requests
    next_tokens = model.forward(active_batch)
```

### vLLM 其他特性

- **Speculative Decoding**：用小模型 draft 多个 token，大模型一次验证
- **Prefix Caching**：相同前缀的请求共享 KV Cache blocks
- **Tensor Parallelism**：推理时跨多卡切分模型

> 踩过的坑：vLLM 的 `gpu_memory_utilization` 参数默认 0.9，在多模型共存的机器上容易 OOM。实际部署建议设 0.8 左右并监控。

## TGI (Text Generation Inference)

HuggingFace 出品，Rust + Python 混合架构。

**核心特性**：
- **Flash Attention**：融合 kernel，减少 HBM 访问（和 Transformer 的注意力计算优化直接相关）
- **Continuous Batching**：和 vLLM 类似的动态 batching
- **Quantization**：支持 GPTQ、AWQ、bitsandbytes
- **Watermarking**：内置文本水印
- **Token Streaming**：SSE 流式返回

**架构**：
```
Client → Router (Rust, 负责调度和排队)
              → Model Server (Python, 实际推理)
```

> TGI vs vLLM 对比：vLLM 在高并发场景下吞吐通常更高（PagedAttention 的显存效率优势），TGI 在 HuggingFace 生态集成更好，部署更简单。实际选型看场景。

## Triton Inference Server

NVIDIA 出品，定位不同：它是一个**通用推理服务框架**，不限于 LLM。

### 核心概念

```
Model Repository 结构：
model_repository/
├── model_a/
│   ├── config.pbtxt          # 模型配置
│   ├── 1/                    # 版本 1
│   │   └── model.onnx
│   └── 2/                    # 版本 2
│       └── model.onnx
├── model_b/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan        # TensorRT engine
```

### 关键特性

1. **多后端支持**：TensorRT、ONNX Runtime、PyTorch、TensorFlow、Python 自定义
2. **Dynamic Batching**：自动将多个请求合并成 batch
3. **Model Ensemble**：多模型串联成 pipeline（比如 tokenizer → model → postprocess）
4. **Concurrent Model Execution**：同一 GPU 上跑多个模型实例

```protobuf
# config.pbtxt 示例
name: "llama"
backend: "python"
max_batch_size: 64

dynamic_batching {
    preferred_batch_size: [8, 16, 32]
    max_queue_delay_microseconds: 100000
}

instance_group [
    { count: 2, kind: KIND_GPU, gpus: [0, 1] }
]
```

> 面试高频考点：Triton 的 Dynamic Batching 配置。`max_queue_delay_microseconds` 是关键参数——等太久延迟高，等太短 batch 太小吞吐低。

### Triton + vLLM 组合

实际生产中常见的模式：用 Triton 做统一的服务网关和调度，后端接 vLLM 做实际的 LLM 推理。

## 性能优化技术总结

| 技术 | 优化目标 | 原理 |
|------|---------|------|
| KV Cache | 避免重复计算 | 缓存历史 token 的 K、V |
| PagedAttention | 显存利用率 | 非连续存储 + 按需分配 |
| Flash Attention | 计算速度 | 减少 HBM 读写，tiling |
| Continuous Batching | 吞吐量 | 动态调度，减少 idle |
| Quantization (W4A16) | 显存 + 速度 | 权重压缩，计算时反量化 |
| Speculative Decoding | 延迟 | 小模型猜测 + 大模型验证 |
| Tensor Parallelism | 单请求延迟 | 跨卡切分计算 |

> 这些优化和分布式训练中的技术有很多交叉：TP 在训练和推理中都用，Flash Attention 训练推理都受益，量化则是训练后压缩用于推理。

## 部署选型建议

- **纯 LLM 服务，追求吞吐**：vLLM
- **HuggingFace 生态，快速上线**：TGI
- **多模型混合部署，企业级**：Triton
- **超大规模，需要精细控制**：Triton + vLLM backend 或自研
