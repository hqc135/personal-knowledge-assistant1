
# 大模型推理优化：KV Cache、量化（GPTQ/AWQ）、投机解码

> 这块是工程面试的重点。模型训练完了不代表能用，推理效率直接决定成本和用户体验。

## 为什么推理是瓶颈？

LLM 推理是**自回归**的：每生成一个 token 都要跑一次前向传播。生成 1000 个 token 就要跑 1000 次。而且大模型参数量巨大（7B 参数 FP16 就要 14GB 显存），显存带宽成为主要瓶颈。

关键指标：
- **Prefill 阶段**：处理输入 prompt，compute-bound（计算密集）
- **Decode 阶段**：逐 token 生成，memory-bound（带宽密集）

## KV Cache

### 原理

在自回归生成时，每个新 token 的注意力计算需要用到所有之前 token 的 K 和 V。如果每次都重新算，复杂度是 O(n²)。

**KV Cache 的做法**：把之前算过的 K、V 缓存起来，新 token 只需要算自己的 Q、K、V，然后和缓存的 K、V 做注意力。

```python
# 没有 KV Cache（每次重算）
for t in range(seq_len):
    Q, K, V = compute_qkv(all_tokens[:t+1])  # O(t) 计算
    output = attention(Q, K, V)

# 有 KV Cache
kv_cache = []
for t in range(seq_len):
    q, k, v = compute_qkv(new_token)  # O(1) 计算
    kv_cache.append((k, v))
    K_all = concat([cached_k for cached_k, _ in kv_cache])
    V_all = concat([_, cached_v for _, cached_v in kv_cache])
    output = attention(q, K_all, V_all)
```

### KV Cache 的显存开销

对于一个 L 层、d_model 维度、h 头的模型，缓存 n 个 token 的 KV：

```
显存 = 2 × L × n × d_model × sizeof(dtype)
```

以 LLaMA-70B 为例（80层，8192维，FP16）：缓存 4096 个 token 约需 **5GB**。这就是为什么长上下文是个难题。

### 优化方向

- **MQA（Multi-Query Attention）**：所有头共享一组 KV → 缓存减少 h 倍
- **GQA（Grouped-Query Attention）**：折中方案，几个头共享一组 KV（LLaMA-2 70B 用的）
- **PagedAttention**（vLLM）：像操作系统的虚拟内存一样管理 KV Cache，避免碎片化

这里和 Transformer 架构笔记中多头注意力的设计直接相关——MQA/GQA 本质上是在注意力头的设计上做 trade-off。

## 量化（Quantization）

把模型权重从 FP16（16bit）压缩到 INT8 甚至 INT4，减少显存占用和带宽需求。

### GPTQ

- **Post-training quantization**（训练后量化）
- 基于 OBQ（Optimal Brain Quantization）的思想：逐层量化，用 Hessian 信息来最小化量化误差
- 核心：量化一个权重时，用剩余未量化的权重来补偿误差

```
# GPTQ 核心思想（简化）
for each column j:
    quantize w_j to w_q_j
    error = w_j - w_q_j
    # 用 Hessian 逆来把误差分摊到后续列
    remaining_weights += error * H_inv[j, j+1:]
```

**踩过的坑**：GPTQ 量化到 4bit 后，某些任务（特别是数学推理）性能下降明显。不是所有场景都能无脑量化。

### AWQ（Activation-Aware Weight Quantization）

核心观察：**不是所有权重同等重要**。那些对应激活值大的通道（salient channels）更重要，量化时应该保护它们。

做法：对重要通道乘一个 scale 因子放大，量化后再缩回来。这样重要通道的量化精度更高。

```
# AWQ 思路
scale = compute_importance(activations)
w_scaled = w * scale
w_quantized = quantize(w_scaled)
# 推理时: output = input / scale @ w_quantized（等价变换）
```

### GPTQ vs AWQ 对比

| | GPTQ | AWQ |
|---|---|---|
| 方法 | 逐层最优量化 | 保护重要通道 |
| 速度 | 量化过程较慢 | 量化过程快 |
| 效果 | 4bit 下略好 | 4bit 下接近 |
| 硬件友好 | 一般 | 更好（对齐硬件） |

## 投机解码（Speculative Decoding）

这是我觉得最巧妙的优化之一。

### 核心思想

用一个小模型（draft model）快速生成 k 个候选 token，然后用大模型**一次性**验证这 k 个 token。如果小模型猜对了，就省了 k-1 次大模型推理。

```python
# 投机解码流程
draft_tokens = small_model.generate(k_tokens)  # 快，但不太准
# 大模型一次前向传播验证 k 个 token
logits = large_model.forward(prompt + draft_tokens)
# 从左到右检查每个 draft token
for i in range(k):
    if accept(draft_tokens[i], logits[i]):
        accept_token(draft_tokens[i])
    else:
        resample_from(logits[i])
        break
```

### 关键性质

- **无损**：通过精心设计的接受/拒绝策略，最终输出的分布和直接用大模型生成**完全一致**
- 加速比取决于小模型和大模型的"一致率"（acceptance rate）
- 典型加速：2-3x（取决于任务和模型对）

**面试高频考点**：为什么是无损的？因为用了类似 rejection sampling 的策略，被拒绝时从修正分布中重新采样。

### 和并行解码的关系

投机解码本质上是把串行的 decode 变成了"猜测+验证"的模式。这和 CPU 的分支预测（branch prediction）思想异曲同工——猜对了加速，猜错了回滚。

## 其他优化手段（简要）

- **Flash Attention**：优化注意力计算的 IO，减少 HBM 访问次数，不改变数学结果
- **Continuous Batching**：动态 batch，不同请求生成完就释放，不用等最长的
- **Tensor Parallelism**：把模型切分到多卡上并行计算

## 面试总结

推理优化的核心矛盾：**模型越大效果越好，但推理越慢越贵**。所有优化都是在这个 trade-off 上做文章：
1. 减少计算量（KV Cache、投机解码）
2. 减少显存/带宽需求（量化、MQA/GQA）
3. 提高硬件利用率（Flash Attention、Continuous Batching）