# 上下文窗口扩展：RoPE、ALiBi、Ring Attention、长文本处理

## 为什么上下文窗口是个问题

标准 Transformer 的自注意力是 O(n²) 的时间和空间复杂度。训练时用 2048 或 4096 的序列长度，推理时想处理 100K+ token 就会遇到两个问题：
1. 显存爆炸（KV cache 线性增长，attention 矩阵二次增长）
2. 位置编码外推失败（没见过的位置，模型不知道怎么处理）

这两个问题对应两条技术路线：**位置编码的外推能力** 和 **注意力计算的工程优化**。

## RoPE（Rotary Position Embedding）

RoPE 是目前主流开源模型（LLaMA、Qwen、Mistral）的标配位置编码。

核心思想：把位置信息编码为**旋转矩阵**，作用在 query 和 key 的向量上。两个 token 的注意力分数只取决于它们的**相对位置**。

```
q_m = R(m) · W_q · x_m
k_n = R(n) · W_k · x_n

q_m^T · k_n = (W_q x_m)^T · R(m-n) · (W_k x_n)
```

其中 R(θ) 是分块的 2D 旋转矩阵，每个维度对用不同频率的旋转：

```python
# RoPE 的核心实现
def apply_rope(x, positions, dim):
    # 频率：theta_i = 10000^(-2i/d)
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    angles = positions[:, None] * freqs[None, :]  # [seq_len, dim//2]
    cos, sin = angles.cos(), angles.sin()
    # 对相邻维度做 2D 旋转
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

我的理解/类比：可以把每个维度对想象成一个时钟的指针，不同维度对的"转速"不同（低频到高频）。两个 token 的相对位置就体现为指针之间的角度差。这和傅里叶变换用不同频率的正弦波编码信号是一个思路。

### RoPE 的外推问题与 NTK-aware Scaling

RoPE 训练时只见过 [0, L] 范围的位置，直接外推到 [0, 4L] 效果会崩。

常见扩展方法：
- **线性插值（Position Interpolation）**：把位置缩放到训练范围内，`pos' = pos × (L_train / L_target)`，需要少量微调
- **NTK-aware Scaling**：修改 base frequency（把 10000 改大），相当于"拉伸"低频分量，高频分量基本不变。不需要微调就能有一定效果
- **YaRN**：结合了 NTK scaling + attention temperature 修正，目前效果最好的方案之一

这里我一开始理解错了：线性插值不是简单地把所有频率都缩放，NTK-aware 的关键 insight 是高频维度已经在训练范围内"转了很多圈"，不需要缩放；低频维度才是外推失败的主因。

## ALiBi（Attention with Linear Biases）

ALiBi 的思路完全不同：**不用位置编码**，而是在 attention score 上直接加一个和距离成正比的负偏置。

```
attention_score(i, j) = q_i^T · k_j - m · |i - j|

其中 m 是每个 head 不同的斜率（几何级数分配）
```

优点：
- 实现极其简单，就是加个 bias 矩阵
- 天然支持外推——距离越远惩罚越大，模型自动学会"远处的信息不太重要"
- 不需要额外微调就能处理更长序列

缺点：
- 线性衰减假设太强，对需要长距离精确引用的任务（如代码补全引用远处的函数定义）不太友好
- 实际效果在超长序列上不如 RoPE + 插值方案

和 RoPE 对比记忆：RoPE 是"编码位置到向量里"，ALiBi 是"在注意力分数上加惩罚"。RoPE 更灵活但需要处理外推，ALiBi 更简单但表达力受限。

## Ring Attention

这是解决**显存瓶颈**的工程方案，核心思想是把长序列的注意力计算分布到多个设备上。

原理：
1. 把序列分成 N 块，分配到 N 个设备
2. 每个设备持有自己那块的 Q，同时 K/V 在设备间**环形传递**
3. 每一步，每个设备用当前收到的 K/V 块计算部分 attention，然后把 K/V 传给下一个设备
4. N 步后，每个设备都见过了所有的 K/V，得到完整的 attention 输出

```
# Ring Attention 伪代码
for step in range(num_devices):
    local_attn += compute_attention(Q_local, KV_current)
    KV_current = send_to_next_device(KV_current)  # 环形通信
    # 通信和计算可以 overlap！
```

关键优化：通信和计算的 overlap。当设备在计算当前 KV 块的 attention 时，同时在传输下一个 KV 块。只要计算时间 ≥ 通信时间，通信开销就被完全隐藏。

和 Flash Attention 的关系：Flash Attention 解决的是单卡内的显存问题（通过 tiling 避免 materialization O(n²) 的 attention 矩阵），Ring Attention 解决的是多卡间的分布式问题。两者是互补的——每个设备内部用 Flash Attention，设备之间用 Ring Attention。

## 长文本处理的工程实践

除了上面的"硬核"方法，实际工程中还有很多 trick：

**KV Cache 优化**：
- GQA（Grouped Query Attention）：多个 query head 共享一组 KV head，直接减少 KV cache 大小
- KV cache 量化：用 FP8/INT8 存储 KV cache
- Token dropping / eviction：丢弃不重要的历史 KV（如 H2O、StreamingLLM 的 attention sink）

**分层处理**：
- Sliding window attention（Mistral 用的）：局部窗口 + 少量全局 token
- 分块摘要：先对长文档分块摘要，再对摘要做推理

面试高频考点：被问到"如何让模型处理 100K token"时，要能从位置编码（RoPE 插值）、注意力计算（Flash/Ring Attention）、KV cache 优化（GQA + 量化）三个层面回答，展示系统性思维。

## 面试要点速记

1. RoPE：旋转矩阵编码相对位置，外推靠 NTK scaling / YaRN
2. ALiBi：attention bias 线性衰减，简单但表达力有限
3. Ring Attention：多卡环形传递 KV，通信计算 overlap
4. Flash Attention：单卡 tiling，避免 O(n²) 显存，和 Ring Attention 互补
5. 实际长文本方案 = 位置编码扩展 + 稀疏/分块注意力 + KV cache 压缩
6. 和 RAG 的交叉引用：当上下文窗口不够时，RAG 是另一条路——不把所有信息塞进窗口，而是按需检索