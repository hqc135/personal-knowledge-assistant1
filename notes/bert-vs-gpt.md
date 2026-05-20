
# BERT vs GPT：预训练范式对比，MLM vs CLM

> 这两个模型代表了 NLP 预训练的两条路线，面试必考。核心区别一句话：BERT 是完形填空，GPT 是续写作文。

## 预训练目标对比

| | BERT | GPT |
|---|---|---|
| 目标 | Masked Language Model (MLM) | Causal Language Model (CLM) |
| 方向 | 双向（bidirectional） | 单向（left-to-right） |
| 架构 | Transformer Encoder | Transformer Decoder |
| 下游使用 | 微调（fine-tune） | Prompt / In-context learning |

## MLM（Masked Language Model）

随机 mask 15% 的 token，让模型预测被 mask 的词。

```
输入: The [MASK] sat on the mat
目标: cat
```

**细节（面试高频）**：那 15% 里面，80% 替换为 [MASK]，10% 替换为随机词，10% 保持不变。为什么？因为下游任务没有 [MASK] token，如果训练时全用 [MASK]，模型会过度依赖这个特殊标记。

BERT 还有 NSP（Next Sentence Prediction）任务，但后来 RoBERTa 证明 NSP 没啥用甚至有害，去掉反而更好。

## CLM（Causal Language Model）

标准的自回归语言模型，预测下一个 token：

```
P(x_1, x_2, ..., x_n) = ∏ P(x_t | x_1, ..., x_{t-1})
```

训练时用 causal mask 保证每个位置只能看到前面的 token。这和 Transformer 笔记里 decoder 的 mask 机制完全一致。

**我的理解**：CLM 天然适合生成任务，因为推理时就是一个 token 一个 token 往后生成的。而 MLM 天然适合理解任务，因为它能看到上下文。

## 为什么 GPT 路线最终"赢了"？

这是我自己的思考总结：

1. **Scaling law 更友好**：CLM 的 loss 定义清晰，和模型大小/数据量的关系更可预测
2. **生成能力是刚需**：ChatGPT 证明了对话/生成是杀手级应用
3. **In-context learning 涌现**：大模型不需要微调就能做任务，这是 BERT 范式做不到的
4. **统一范式**：GPT 可以通过 prompt 做分类、翻译、摘要等所有任务，BERT 每个任务要单独加 head

但 BERT 类模型没有死——在 embedding、检索、分类等场景，encoder 模型（如 BGE、E5）仍然是主流。

## 关键架构差异

```python
# BERT: 双向注意力，所有位置互相可见
attention_mask = ones(seq_len, seq_len)

# GPT: 因果注意力，只能看左边
attention_mask = tril(ones(seq_len, seq_len))
```

这里我一开始理解错了：BERT 不是"两个方向的 RNN 拼起来"（那是 BiLSTM），而是每个位置同时看到所有其他位置，是真正的全局双向。

## 微调 vs Prompting

**BERT 时代**：预训练 + 微调（加一个分类头，在下游数据上训练）
- 优点：小数据也能用，效果好
- 缺点：每个任务一个模型，部署成本高

**GPT 时代**：预训练 + Prompt（不改模型参数）
- Zero-shot：直接问
- Few-shot：给几个例子
- 这和 Prompt Engineering 笔记中的技巧直接相关

## 变体和后续发展

- **RoBERTa**：去掉 NSP，更多数据，动态 mask → BERT 的最佳实践
- **ALBERT**：参数共享 + 分解 embedding → 轻量化
- **GPT-2/3/4**：不断 scale up，涌现能力越来越强
- **T5**：把所有任务统一为 text-to-text，用 encoder-decoder 架构，算是折中方案
- **PaLM/LLaMA**：decoder-only 成为绝对主流

## 面试追问

- Q：为什么 decoder-only 比 encoder-decoder 更流行？→ 简单、scaling 效率高、KV Cache 友好（和推理优化笔记交叉）
- Q：BERT 能做生成吗？→ 理论上可以（逐个位置 unmask），但效率极低且效果差
- Q：GPT 能做好 NLU 任务吗？→ 大模型可以，小模型不如 BERT。规模是关键变量

## 一个有趣的视角

BERT 和 GPT 的区别本质上是**信息流方向**的区别。BERT 允许信息双向流动，所以对每个 token 的表示更"完整"；GPT 限制信息只能从左到右流动，但这个约束恰好和生成任务的因果性一致。约束有时候是优势。