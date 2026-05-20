
# 分布式训练：数据并行、模型并行、ZeRO、FSDP

## 为什么需要分布式训练

一句话：模型太大，单卡放不下；数据太多，单卡训练太慢。

GPT-3 有 175B 参数，fp16 下光参数就要 350GB 显存，而 A100 才 80GB。所以分布式不是"优化"，是"必须"。

## 数据并行 (Data Parallelism, DP)

最简单的思路：每张卡都放一份完整模型，各自处理不同的 mini-batch，然后同步梯度。

```python
# 伪代码
for each GPU i:
    loss_i = model(data_chunk_i)
    grad_i = backward(loss_i)

# All-Reduce: 所有卡的梯度求平均
avg_grad = all_reduce(grad_0, grad_1, ..., grad_n) / n

for each GPU i:
    model_i.params -= lr * avg_grad
```

**关键通信操作**：All-Reduce，复杂度 O(2 * model_size * (n-1)/n)，和 GPU 数量近似无关（Ring All-Reduce）。

> 面试高频考点：PyTorch 的 `DistributedDataParallel (DDP)` 和旧版 `DataParallel` 的区别。DP 用单进程多线程 + GIL 瓶颈，DDP 用多进程 + NCCL 通信，性能差很多。

**局限**：每张卡都要放完整模型，当模型本身超过单卡显存时就不行了。

## 模型并行 (Model Parallelism)

### 张量并行 (Tensor Parallelism, TP)

把单个层的权重矩阵切开，分到不同卡上。比如一个线性层 Y = XA，把 A 按列切成 [A1, A2]，两张卡各算一半，最后拼起来。

Megatron-LM 的做法：MLP 层先列切再行切，中间不需要通信；Attention 层按 head 切分。

> 这里我一开始理解错了：TP 不是把不同层放不同卡（那是流水线并行），而是把同一层的矩阵运算拆开。

### 流水线并行 (Pipeline Parallelism, PP)

把模型按层分段，每段放一张卡。问题是朴素实现会导致大量 GPU 空闲（bubble）。

GPipe 的解决方案：把 mini-batch 再切成 micro-batch，形成流水线。

```
时间 →
GPU 0: [F1][F2][F3][F4][  ][  ][  ][B4][B3][B2][B1]
GPU 1: [  ][F1][F2][F3][F4][  ][B4][B3][B2][B1]
GPU 2: [  ][  ][F1][F2][F3][F4][B4][B3][B2][B1]
```

Bubble ratio ≈ (p-1) / (m + p - 1)，其中 p 是 pipeline stages，m 是 micro-batches。m 越大 bubble 越小。

## ZeRO (Zero Redundancy Optimizer)

DeepSpeed 提出的核心优化。核心观察：数据并行中，每张卡都存了完整的 optimizer states、gradients、parameters，太浪费了。

**三个阶段**：
- ZeRO-1：切分 optimizer states（Adam 的 m 和 v），显存省 4x
- ZeRO-2：再切分 gradients，省 8x
- ZeRO-3：连 parameters 也切分，省的和模型并行一样多（N 倍，N 是 GPU 数）

```
显存占用（以 fp16 训练 + Adam 为例，模型参数量 Φ）：
- 参数: 2Φ bytes (fp16)
- 梯度: 2Φ bytes (fp16)  
- Optimizer states: 12Φ bytes (fp32 参数副本 4Φ + m 4Φ + v 4Φ)
总计: 16Φ bytes per GPU (数据并行，无 ZeRO)

ZeRO-3: 16Φ / N bytes per GPU
```

> 和模型并行对比记忆：ZeRO-3 在通信模式上仍然是数据并行的逻辑（每张卡处理不同数据），但存储上达到了模型并行的效果。代价是需要额外的 All-Gather 通信来临时拼出完整参数。

## FSDP (Fully Sharded Data Parallel)

PyTorch 原生的 ZeRO-3 实现。核心思想一样：参数、梯度、优化器状态都分片存储。

```python
# FSDP 的工作流程
# Forward:
#   1. All-Gather 拿到当前层的完整参数
#   2. 计算前向
#   3. 丢弃非本分片的参数（省显存）
# Backward:
#   1. All-Gather 拿到当前层的完整参数
#   2. 计算梯度
#   3. Reduce-Scatter 把梯度分片回各卡
#   4. 丢弃非本分片的参数和梯度
```

**FSDP vs DDP**：
| | DDP | FSDP |
|---|---|---|
| 参数存储 | 每卡完整 | 分片 |
| 通信 | All-Reduce 梯度 | All-Gather 参数 + Reduce-Scatter 梯度 |
| 显存 | 高 | 低 |
| 通信量 | 2Φ | 3Φ（forward AG + backward AG + RS） |

> 踩过的坑：FSDP 的 `auto_wrap_policy` 很关键。如果 wrapping 粒度太细（每个小层都 wrap），通信次数太多；太粗（整个模型一个 unit），显存省不下来。一般按 Transformer Block 来 wrap。

## 实际训练中的混合策略

大模型训练通常是 3D 并行：TP + PP + DP（或 FSDP）。

比如训练一个 70B 模型在 64 张 A100 上：
- TP=8（一个节点内 8 卡，NVLink 高带宽）
- PP=2（跨 2 个节点）
- DP=4（4 组 pipeline 副本）

> 这里和 Transformer 架构的设计有关联：Attention 天然适合按 head 做 TP，FFN 适合按列/行切分。模型架构本身就考虑了并行友好性。

## 面试常见问题

1. All-Reduce 的通信量是多少？—— Ring All-Reduce 下是 2*(N-1)/N * model_size ≈ 2 * model_size
2. ZeRO-3 和模型并行的区别？—— 逻辑上仍是数据并行，每卡看不同数据；通信模式不同
3. 梯度累积 (Gradient Accumulation) 的作用？—— 在显存不够用大 batch 时，多个 micro-batch 累积梯度再更新，等效于大 batch
4. 通信和计算如何 overlap？—— FSDP 可以在计算当前层时预取下一层参数（prefetch）

> 这个话题和后面的模型服务部署（vLLM 的 tensor parallelism）直接相关，推理时的并行策略是训练时的子集。

