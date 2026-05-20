
# Prompt Engineering 技巧：CoT、Few-shot、Self-consistency

> 说实话一开始觉得 prompt engineering 就是"调参玄学"，但深入了解后发现背后有比较 solid 的逻辑。面试中也越来越常问了。

## 基本概念

Prompt Engineering 就是通过设计输入（prompt）来引导 LLM 产生期望的输出，**不修改模型参数**。

这和 BERT 时代的 fine-tuning 形成鲜明对比——fine-tuning 改参数适应任务，prompting 改输入适应模型。本质上是把"任务适配"的工作从模型侧转移到了输入侧。

## Few-shot Prompting

在 prompt 中给几个示例，让模型"学会"任务格式：

```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
cheese => 
```

**关键发现（GPT-3 论文）**：
- 0-shot < 1-shot < few-shot，但边际收益递减
- 示例的**格式**比内容更重要——模型主要在学输出格式
- 示例顺序会影响结果（recency bias）

我的理解：Few-shot 不是真的在"学习"，而是在激活模型预训练时见过的类似模式。这和 in-context learning 的机制研究有关——有论文认为 Transformer 在做隐式的梯度下降。

## Chain-of-Thought (CoT)

### 核心思想

让模型"展示推理过程"而不是直接给答案：

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have now?
A: Roger started with 5 balls. 2 cans of 3 = 6 balls. 5 + 6 = 11. The answer is 11.
```

### 为什么有效？

这里我一开始理解错了——不是因为模型"学会了思考"，而是：
1. 把复杂问题分解成简单步骤，每步的 token 预测难度降低
2. 中间步骤作为"工作记忆"，弥补 Transformer 没有显式记忆的缺陷
3. 减少了从问题到答案的"推理跳跃距离"

**和 Transformer 架构的联系**：Transformer 每层的计算深度是固定的。CoT 通过增加 token 数量，间接增加了"计算步数"。这本质上是用序列长度换计算深度。

### Zero-shot CoT

不需要给示例，只需要加一句 "Let's think step by step"：

```
Q: [问题]
A: Let's think step by step.
```

这个简单到离谱但确实有效。Wei et al. 2022 的论文显示在 GSM8K 上准确率从 17.7% 提升到 78.7%。

## Self-Consistency

### 动机

CoT 的一个问题：同一个问题，不同的推理路径可能得到不同答案。哪个对？

### 方法

1. 对同一个问题，用 temperature > 0 采样多条推理路径
2. 提取每条路径的最终答案
3. **多数投票**（majority voting）选最终答案

```python
answers = []
for _ in range(n_samples):
    response = llm.generate(prompt, temperature=0.7)
    answer = extract_answer(response)
    answers.append(answer)
final_answer = majority_vote(answers)
```

**我的理解**：这就是 ensemble 的思想用在了推理上。和机器学习中 bagging 的逻辑一样——多个弱分类器投票比单个强分类器更鲁棒。

### 代价

需要多次推理，成本线性增加。实际中 5-10 次采样就够了，边际收益递减很快。

## 其他重要技巧

### Structured Output

用格式约束来提高输出质量：

```
Please respond in the following JSON format:
{
  "answer": "...",
  "confidence": 0.0-1.0,
  "reasoning": "..."
}
```

### Role Prompting

给模型设定角色：

```
You are an expert mathematician. Solve the following problem...
```

有效但效果不如 CoT 显著。我觉得主要是帮助模型"选择"预训练中对应领域的知识。

### Least-to-Most Prompting

先让模型分解问题，再逐步解决子问题：

```
Step 1: What sub-problems do I need to solve?
Step 2: Solve each sub-problem...
```

和 CoT 的区别：CoT 是一步到位写出推理链，Least-to-Most 是显式地先规划再执行。对复杂问题更有效。

## 常见坑

1. **Prompt 对格式极其敏感**：多一个换行、少一个空格都可能影响结果。这不是玄学，是因为 tokenizer 对空白字符的处理方式不同
2. **Few-shot 示例的选择**：和测试样本越相似越好，但太相似会导致模型"抄答案"
3. **CoT 不是万能的**：简单任务加 CoT 反而可能降低性能（overthinking）
4. **模型大小很关键**：CoT 在小模型（<10B）上基本无效，这是涌现能力（emergent ability）

## 和 RLHF 的关系

Prompt engineering 的效果很大程度上依赖于模型的 instruction following 能力，而这个能力主要来自 RLHF/RLHF 对齐训练。一个没有经过对齐的 base model，prompt engineering 的效果会大打折扣。这和 RLHF 笔记中讨论的对齐方法直接相关。

## 面试高频问题

- Q：CoT 为什么在小模型上不 work？→ 涌现能力，小模型的中间推理步骤本身就不可靠
- Q：Self-consistency 和 beam search 的区别？→ beam search 是在 token 级别搜索，self-consistency 是在推理路径级别搜索
- Q：如何评估 prompt 的好坏？→ 没有统一标准，通常用 benchmark 准确率 + 人工评估