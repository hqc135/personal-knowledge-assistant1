
# Transformer 架构详解：自注意力、多头注意力、位置编码

> 写这篇笔记的时候反复看了 "Attention Is All You Need" 原论文和 Jay Alammar 的图解博客，终于把一些细节理清了。

## 核心思想

Transformer 的本质就是**完全抛弃 RNN 的递归结构**，用注意力机制来建模序列中任意两个位置之间的依赖关系。好处是并行化训练，坏处是计算复杂度 O(n²)（n 是序列长度）。

## 自注意力（Self-Attention）

每个 token 生成三个向量：Query、Key、Value。

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**我的理解**：Q 是"我在找什么"，K 是"我能提供什么标签"，V 是"我实际的内容"。QK^T 就是在算"你要的和我有的匹配度"。

为什么要除以 √d_k？这里我一开始理解错了——不是为了归一化，而是因为当 d_k 很大时，点积的方差会变大（方差约等于 d_k），导致 softmax 进入梯度极小的饱和区。除以 √d_k 让方差回到 1 附近。**面试高频考点**。

```python
# 伪代码
def self_attention(X, W_q, W_k, W_v):
    Q = X @ W_q  # (seq_len, d_k)
    K = X @ W_k  # (seq_len, d_k)
    V = X @ W_v  # (seq_len, d_v)
    scores = Q @ K.T / sqrt(d_k)  # (seq_len, seq_len)
    weights = softmax(scores, dim=-1)
    return weights @ V
```

## 多头注意力（Multi-Head Attention）

把 Q、K、V 分成 h 个头，每个头独立做注意力，最后 concat 再投影：

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**为什么要多头？** 单头注意力只能学一种"关注模式"，多头让模型同时关注不同子空间的信息。比如一个头关注语法关系，另一个头关注语义相似性。

关键细节：每个头的维度是 d_k = d_model / h，所以多头的总计算量和单头全维度差不多。这不是增加计算量，而是增加表达多样性。

## 位置编码（Positional Encoding）

Transformer 本身是置换不变的（permutation invariant），必须显式注入位置信息。

**原始正弦位置编码：**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

设计直觉：不同维度的频率不同，低维变化快（捕捉局部位置），高维变化慢（捕捉全局位置）。而且 PE(pos+k) 可以表示为 PE(pos) 的线性变换，理论上能学到相对位置。

**和 RoPE 对比记忆**：RoPE（Rotary Position Embedding）是现在主流 LLM 用的方案（LLaMA、Qwen 等），它把位置信息编码到注意力计算中而不是加到输入上，天然支持相对位置，外推性更好。这个和后面 LLM 推理优化中的长上下文处理有关。

## Encoder-Decoder 结构要点

- Encoder：双向自注意力，每个 token 能看到所有位置
- Decoder：因果掩码（causal mask），只能看到左边的 token + 交叉注意力看 encoder 输出
- 残差连接 + LayerNorm：每个子层都有，稳定训练。Pre-Norm（先 LN 再注意力）vs Post-Norm（先注意力再 LN），现在主流用 Pre-Norm，训练更稳定

## FFN 层容易被忽略

每个 Transformer block 里除了注意力还有一个 FFN：

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

FFN 的参数量占了模型的 2/3！有研究认为 FFN 是"知识存储"的地方，而注意力层负责"信息路由"。这个观点和后面 RLHF 中讨论模型编辑/知识注入有交叉。

## 踩过的坑

1. 注意力矩阵的 mask 在 decoder 中是**加一个很大的负数**（如 -1e9）再 softmax，不是直接置零
2. d_model、d_k、d_v 的关系：d_k = d_v = d_model / h（原论文设定），但不是必须相等
3. 位置编码是**加**到 embedding 上的，不是 concat

## 面试常见追问

- Q：为什么不用相对位置编码？→ 原始 Transformer 用绝对的，但后续 T5 用了相对位置，现在 RoPE 是主流
- Q：注意力复杂度怎么降？→ 引出 Flash Attention、稀疏注意力、线性注意力等（和推理优化笔记交叉）
- Q：Transformer 能处理多长的序列？→ 取决于位置编码的外推能力和显存限制


