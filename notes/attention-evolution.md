
## 注意力机制演进：Bahdanau → Luong → Self-Attention → Flash Attention

### 演进主线

注意力机制的发展可以概括为：

**"让模型学会看哪里"** → **"让模型看自己"** → **"让模型看得更快"**

```
Bahdanau (2014) → Luong (2015) → Transformer Self-Attention (2017) 
→ Multi-Head (2017) → Sparse/Linear Attention → Flash Attention (2022)
```

---

## Bahdanau Attention（加性注意力）

背景：Seq2Seq 模型把整个输入压缩成一个固定向量，长序列信息丢失严重。

核心思想：解码每个 token 时，动态地"回头看"编码器的所有隐状态。

```python
# encoder hidden states: h_1, ..., h_T
# decoder hidden state at step t: s_t

# 计算注意力分数
e_tj = v^T * tanh(W_s * s_{t-1} + W_h * h_j)   # 加性

# 归一化
α_tj = softmax(e_tj)   # 对 j 做 softmax

# 上下文向量
c_t = Σ_j α_tj * h_j
```

> 为什么叫"加性"？因为 score 函数里 s 和 h 是加在一起再过 tanh 的。

---

## Luong Attention（乘性注意力）

简化了 score 的计算方式，提出三种变体：

```
dot:      score(s_t, h_j) = s_t^T * h_j
general:  score(s_t, h_j) = s_t^T * W * h_j
concat:   score(s_t, h_j) = v^T * tanh(W * [s_t; h_j])  # 类似 Bahdanau
```

> **面试考点**：Bahdanau vs Luong 的区别？
> - Bahdanau 用 s_{t-1}（上一步的状态），Luong 用 s_t（当前步）
> - Bahdanau 是加性 score，Luong 主推乘性（dot product）
> - 乘性计算更快（矩阵乘法可以并行），效果差不多

> 这里我一开始理解错了：以为 Luong 就是简单的 dot product。实际上 Luong 论文提了三种，dot 只是其中最简单的一种。但后来大家说"乘性注意力"基本就指 dot product 了。

---

## Transformer Self-Attention

革命性的变化：**不再是 encoder-decoder 之间的注意力，而是序列对自身的注意力**。

```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

其中 Q, K, V 都是输入 X 经过不同线性变换得到的：
```python
Q = X @ W_Q   # [seq_len, d_k]
K = X @ W_K   # [seq_len, d_k]
V = X @ W_V   # [seq_len, d_v]

scores = Q @ K.T / sqrt(d_k)   # [seq_len, seq_len]
attn_weights = softmax(scores, dim=-1)
output = attn_weights @ V       # [seq_len, d_v]
```

> **为什么除以 √d_k？** 当 d_k 很大时，QK^T 的值会很大，softmax 会趋向 one-hot（梯度消失）。除以 √d_k 让方差保持在 1 附近。这个和 BatchNorm/LayerNorm 的思想类似——控制中间值的尺度。

**Multi-Head Attention**：

```python
# 不是用一个大的 attention，而是分成 h 个小的
head_i = Attention(X @ W_Q_i, X @ W_K_i, X @ W_V_i)
MultiHead = Concat(head_1, ..., head_h) @ W_O
```

> 多头的直觉：不同的头可以关注不同类型的关系。比如一个头关注语法结构，另一个关注语义相似性。实际可视化也确实发现了这种模式。

---

## Self-Attention 的复杂度问题

标准 self-attention 的时间和空间复杂度都是 O(n²)，n 是序列长度。

这在长序列上是灾难性的：
- n=512: attention matrix 约 1MB
- n=4096: 约 64MB  
- n=100K: 约 40GB 😱

各种改进方案：

| 方法 | 思路 | 复杂度 |
|------|------|--------|
| Sparse Attention (2019) | 只计算部分位置对 | O(n√n) |
| Linformer (2020) | 低秩近似 K, V | O(n) |
| Performer (2020) | 核函数近似 softmax | O(n) |
| Flash Attention (2022) | IO 感知的精确计算 | O(n²) 但实际快很多 |

---

## Flash Attention

> 这是我觉得最精彩的工作之一——不改变数学结果，只改变计算方式，就能快 2-4 倍。

核心洞察：标准 attention 的瓶颈不是计算量，而是**内存访问**（IO）。

GPU 的内存层次：
```
SRAM (片上, ~20MB, 快) ←→ HBM (显存, ~40GB, 慢)
```

标准实现的问题：
1. 计算 QK^T → 写入 HBM（n² 大小）
2. 计算 softmax → 读 HBM，写 HBM
3. 乘以 V → 读 HBM，写 HBM

每一步都要在 HBM 上读写巨大的中间矩阵。

**Flash Attention 的做法：分块（tiling）+ 在线 softmax**

```
核心思路：
1. 把 Q, K, V 分成小块，每块能放进 SRAM
2. 在 SRAM 内完成一个块的 attention 计算
3. 用 online softmax 技巧逐块累积结果
4. 永远不把 n×n 的 attention matrix 写入 HBM
```

Online softmax 的关键：

```python
# 标准 softmax 需要先算全局 max（需要看完所有数据）
# Online softmax 可以逐块更新：

# 处理第 j 块时：
m_new = max(m_old, max(block_j_scores))
# 用 m_new 修正之前的累积结果
l_new = l_old * exp(m_old - m_new) + sum(exp(block_j_scores - m_new))
O_new = O_old * (l_old * exp(m_old - m_new) / l_new) + softmax(block_j) @ V_j / l_new
```

> 和优化器笔记中的数值稳定性问题类似：softmax 的实现必须先减去 max 防止溢出，Flash Attention 把这个技巧推广到了分块场景。

Flash Attention 的效果：
- 内存从 O(n²) 降到 O(n)（不存完整 attention matrix）
- 实际速度快 2-4 倍（减少 HBM 访问）
- **数学上完全等价**，不是近似！

---

## Flash Attention 2 & 3

**Flash Attention 2 (2023)**：
- 更好的并行策略：沿 seq_len 维度并行（而不是 batch/head 维度）
- 减少非矩阵乘法运算（这些运算在 GPU 上效率低）
- 比 v1 再快 2 倍左右

**Flash Attention 3 (2024)**：
- 针对 Hopper 架构（H100）优化
- 利用 TMA（Tensor Memory Accelerator）异步加载数据
- 支持 FP8 精度
- 利用 warp specialization 让不同 warp 做不同的事（producer-consumer 模式）

---

## GQA 和 MQA（补充）

和 Flash Attention 配合使用的 KV 压缩方案：

```
MHA (Multi-Head Attention):  每个头有独立的 Q, K, V
MQA (Multi-Query Attention): 所有头共享一组 K, V，各自有 Q
GQA (Grouped-Query Attention): 折中，几个头共享一组 K, V
```

> GQA 是 LLaMA 2 开始流行的方案。好处是推理时 KV cache 小很多（和 KV cache 优化直接相关），训练质量接近 MHA。

---

## 面试常见问题

1. **Attention 的本质是什么？**
   - 加权求和。权重由 query 和 key 的相似度决定，value 是被加权的内容。
   - 可以类比为"软寻址"：query 是地址，key 是索引，value 是存储的内容。

2. **Self-Attention 和 CNN 的关系？**
   - CNN 是局部注意力（固定感受野），Self-Attention 是全局注意力
   - 1x1 卷积 ≈ 没有位置交互的 attention（只做 channel mixing）
   - ViT 证明了纯 attention 可以替代 CNN 做视觉任务

3. **为什么 Transformer 需要位置编码？**
   - Self-Attention 是置换不变的（permutation invariant），打乱输入顺序输出不变
   - 位置编码注入顺序信息。正弦编码、可学习编码、RoPE（旋转位置编码）各有优劣
   - RoPE 现在是 LLM 的主流选择（相对位置、外推性好）

4. **Flash Attention 为什么不改变结果但能加速？**
   - 因为它优化的是 IO 而不是计算。标准实现是 compute-bound 假设下的最优，但实际是 memory-bound 的。Flash Attention 用更多计算（重复计算 softmax 的部分中间结果）换更少的内存访问。

---

## 个人踩坑

- 实现 attention 时忘记 causal mask，导致 decoder 能"看到未来"，loss 降得特别快但生成时一塌糊涂
- Flash Attention 在 PyTorch 2.0+ 中通过 `torch.nn.functional.scaled_dot_product_attention` 自动启用，不需要手动装包了（但自定义 attention pattern 还是需要原版库）
- Multi-head 的 head_dim 通常是 64 或 128，不是越大越好。太大单个头的 attention matrix 放不进 SRAM，Flash Attention 效率下降
