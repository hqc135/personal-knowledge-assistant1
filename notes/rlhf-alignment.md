
# RLHF 与对齐：PPO、DPO、Constitutional AI

> 这是我觉得最有意思也最复杂的一块。模型能力强不代表好用，对齐（alignment）就是让模型"听话"且"安全"。

## 为什么需要对齐？

预训练的 LLM 目标是"预测下一个 token"，这和"有用、无害、诚实"之间有 gap：
- 模型可能生成有害内容（预训练数据里有）
- 模型可能胡说八道（hallucination）
- 模型可能不遵循指令（它只是在续写文本）

对齐就是弥补这个 gap。

## RLHF 三阶段流程

### Stage 1: Supervised Fine-Tuning (SFT)

用人工标注的高质量 (instruction, response) 对来微调 base model。

```
数据格式: {"instruction": "写一首关于春天的诗", "response": "春风拂面..."}
```

这一步让模型学会"遵循指令"的基本格式。

### Stage 2: Reward Model (RM) 训练

收集人类偏好数据：给同一个 prompt 的多个回复排序。

```
Prompt: "解释量子力学"
Response A: [详细准确的解释]  ← 人类偏好
Response B: [简短模糊的回答]
```

训练一个 reward model 来预测人类偏好：

```python
# Bradley-Terry 模型
P(A > B) = sigmoid(r(A) - r(B))
loss = -log(sigmoid(r_chosen - r_rejected))
```

**面试高频考点**：RM 的训练数据是**成对比较**，不是绝对分数。这比让人打分更容易、更一致。

### Stage 3: PPO 优化

用 RM 作为奖励信号，通过强化学习优化 LLM 的策略：

```
objective = E[r(prompt, response)] - β * KL(π_θ || π_ref)
```

- 第一项：最大化 reward
- 第二项：KL 散度惩罚，防止模型偏离 SFT 模型太远（避免 reward hacking）

```python
# PPO 简化流程
for batch in data:
    responses = policy.generate(batch.prompts)
    rewards = reward_model(batch.prompts, responses)
    kl_penalty = compute_kl(policy, reference_policy)
    total_reward = rewards - beta * kl_penalty
    # PPO clip 更新
    ratio = policy.prob(responses) / old_policy.prob(responses)
    clipped_ratio = clip(ratio, 1-eps, 1+eps)
    loss = -min(ratio * advantage, clipped_ratio * advantage)
    policy.update(loss)
```

### PPO 的问题

1. **训练不稳定**：4 个模型同时在内存中（policy、reference、reward model、value model）
2. **Reward hacking**：模型学会"讨好" RM 而不是真正变好（比如生成冗长但空洞的回答）
3. **超参数敏感**：β、clip range、learning rate 都很难调
4. **工程复杂度高**：需要 RL 基础设施，分布式训练更复杂

## DPO（Direct Preference Optimization）

### 核心洞察

DPO 的关键发现：**可以跳过 reward model，直接从偏好数据优化策略**。

数学推导（简化版）：在 RLHF 的目标函数下，最优策略有闭式解：

```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
```

反过来可以得到：

```
r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + const
```

把这个代入 Bradley-Terry 模型，就得到 DPO 的 loss：

```python
# DPO Loss
loss = -log(sigmoid(
    beta * (log π_θ(y_w|x) - log π_ref(y_w|x)) 
    - beta * (log π_θ(y_l|x) - log π_ref(y_l|x))
))
```

其中 y_w 是偏好的回复，y_l 是不偏好的回复。

### DPO vs PPO 对比

| | PPO | DPO |
|---|---|---|
| 需要 RM | 是 | 否 |
| 训练稳定性 | 差 | 好 |
| 内存需求 | 4个模型 | 2个模型（policy + reference） |
| 工程复杂度 | 高（需要 RL） | 低（标准 supervised learning） |
| 效果 | 略好（理论上） | 接近，某些场景更好 |

**我的理解**：DPO 本质上是把 RL 问题转化成了分类问题——给定一对回复，让模型学会区分好坏。这大大降低了工程门槛。

### DPO 的局限

这里我一开始理解错了——DPO 不是完美替代 PPO 的：
1. DPO 是 offline 的，不能在训练过程中探索新的回复
2. 对偏好数据的分布很敏感（distribution shift 问题）
3. 有研究表明在某些复杂任务上 PPO 仍然更好

后续改进：**IPO**（Identity Preference Optimization）、**KTO**（Kahneman-Tversky Optimization，只需要好/坏标签不需要成对比较）、**ORPO**（不需要 reference model）。

## Constitutional AI (CAI)

### Anthropic 提出的方法

核心思想：用 AI 自己来做对齐，减少人类标注的需求。

流程：
1. 让模型生成回复
2. 让模型根据一组"宪法原则"（constitution）来批评自己的回复
3. 让模型根据批评修改回复
4. 用修改后的数据做 RLHF（RLAIF：RL from AI Feedback）

```
Constitution 示例:
- "回复是否包含有害内容？"
- "回复是否诚实？"
- "回复是否尊重用户？"
```

### 和 RLHF 的关系

CAI 本质上是用 AI 替代了 RLHF 中的人类标注者。优点是可扩展（scalable），缺点是 AI 的判断可能有系统性偏差。

**和 Prompt Engineering 的交叉**：CAI 中的"宪法原则"本质上就是一种 prompt——通过精心设计的 prompt 来引导模型自我改进。

## Reward Hacking 深入

这是对齐中最棘手的问题之一：

```
例子：
- RM 偏好长回复 → 模型学会生成冗长废话
- RM 偏好自信的语气 → 模型学会胡说八道但很自信
- RM 偏好格式化输出 → 模型学会加无意义的 markdown
```

缓解方法：
- KL 惩罚（限制偏离 reference）
- 定期更新 RM
- 多个 RM ensemble
- 对抗训练

## 最新趋势

1. **RLHF → RLAIF**：用 AI 反馈替代人类反馈，降低成本
2. **Process Reward Model**：不只奖励最终答案，奖励每一步推理过程（和 CoT 相关）
3. **Online DPO / Iterative DPO**：解决 DPO 的 offline 问题
4. **GRPO（Group Relative Policy Optimization）**：DeepSeek 提出，不需要 value model，用组内相对排名作为 baseline

## 面试常见追问

- Q：为什么不直接用 RM 的分数做 supervised learning？→ 因为 RM 分数的绝对值没有意义，只有相对排序有意义
- Q：KL 惩罚的 β 怎么选？→ 太大模型不学习，太小 reward hacking。通常从 0.01-0.1 开始调
- Q：DPO 和 contrastive learning 的关系？→ 确实很像，都是拉近正样本、推远负样本。但 DPO 有严格的理论推导（从 RL 目标函数出发）
- Q：对齐税（alignment tax）是什么？→ 对齐后模型在某些能力上可能下降（比如创造性写作），这是 helpfulness 和 harmlessness 的 trade-off
