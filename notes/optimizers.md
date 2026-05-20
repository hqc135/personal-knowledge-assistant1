
## 梯度下降变体：SGD、Adam、AdamW、学习率调度

### 核心思路

优化器的本质就是一句话：怎么更聪明地沿着梯度方向走。从最朴素的 SGD 到 Adam，核心演进路线是：

**SGD → 加动量 → 自适应学习率 → 两者结合 → 修正权重衰减**

---

## SGD 及其动量变体

最基础的更新规则：

```
θ = θ - lr * ∇L(θ)
```

问题很明显：梯度方向噪声大，容易在鞍点附近震荡。

加上动量（Momentum）后：

```
v_t = β * v_{t-1} + ∇L(θ)
θ = θ - lr * v_t
```

> 我的理解：动量就像一个球从山坡滚下来，有惯性。即使当前梯度方向变了，它还会保持之前的运动趋势。β 一般取 0.9。

Nesterov 动量是个小改进——先按动量方向"看一步"，再算梯度。实际效果比标准动量好一点，但面试问得不多。

---

## Adam：自适应矩估计

Adam = Momentum + RMSProp，同时维护一阶矩（均值）和二阶矩（方差）：

```python
m_t = β1 * m_{t-1} + (1 - β1) * g_t        # 一阶矩
v_t = β2 * v_{t-1} + (1 - β2) * g_t^2      # 二阶矩

# 偏差修正（这里我一开始理解错了，以为可以省略）
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)

θ = θ - lr * m_hat / (sqrt(v_hat) + ε)
```

默认超参：β1=0.9, β2=0.999, ε=1e-8

> **面试高频考点**：为什么需要偏差修正？因为 m 和 v 初始化为 0，前几步的估计会偏小。t 越大，修正项 `1-β^t` 越接近 1，修正效果消失。

Adam 的问题：
- 泛化性能有时不如 SGD with momentum（经典论文 "The Marginal Value of Adaptive Gradient Methods"）
- 权重衰减的实现有 bug（这就引出了 AdamW）

---

## AdamW：解耦权重衰减

传统 Adam 里加 L2 正则是这样的：

```
g_t = ∇L(θ) + λ * θ   # L2 正则梯度
```

但这样 L2 项也会被自适应学习率缩放，导致正则化效果不一致。

AdamW 的做法是把权重衰减从梯度计算中解耦出来：

```python
θ = θ - lr * (m_hat / (sqrt(v_hat) + ε)) - lr * λ * θ
```

> 和正则化方法（Dropout、L1/L2）对比记忆：L2 正则和权重衰减在 SGD 下数学等价，但在 Adam 下不等价！这是 AdamW 论文的核心洞察。

现在 Transformer 训练基本都用 AdamW，这已经是默认选择了。

---

## 学习率调度（LR Schedule）

光选对优化器不够，学习率怎么变化同样关键。常见策略：

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| Step Decay | 每 N 个 epoch 乘以 γ | 传统 CV |
| Cosine Annealing | 余弦曲线衰减 | 大模型预训练 |
| Warmup + Cosine | 先线性升再余弦降 | Transformer 标配 |
| OneCycleLR | 先升后降，一个大周期 | 快速训练 |

Warmup 的直觉：训练初期参数随机，梯度方向不靠谱，用小学习率"热身"避免一开始就跑偏。

```python
# 典型的 warmup + cosine 实现
if step < warmup_steps:
    lr = base_lr * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = base_lr * 0.5 * (1 + cos(π * progress))
```

> 这个和 Transformer 原论文的 schedule 对比：原论文用的是 `d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})`，本质也是先升后降，但衰减速度不同。现在大家更多用 cosine。

---

## 面试常见问题

1. **Adam 和 SGD 怎么选？** 
   - 追求训练速度/稳定性：Adam/AdamW
   - 追求最终泛化性能（CV 任务）：SGD + momentum + 精调 lr
   - LLM 训练：几乎都是 AdamW

2. **为什么 Adam 泛化可能差？**
   - 自适应学习率让 loss landscape 中的 sharp minima 也能收敛进去，而 SGD 的噪声会帮助逃离 sharp minima，找到 flat minima（泛化更好）

3. **梯度累积和大 batch 的关系？**
   - 梯度累积 N 步 ≈ batch size 扩大 N 倍，但省显存。注意 BatchNorm 在这种情况下统计量会不准（和 BatchNorm vs LayerNorm 的讨论相关）。

---

## 个人踩坑记录

- 用 Adam 时忘记设 weight_decay，模型过拟合严重，后来换 AdamW 加 0.01 的 decay 就好了
- Cosine schedule 的 T_max 设错，导致学习率中途就降到 0 了，loss 不再下降
- 混合精度训练时 ε 要调大（1e-4 甚至更大），不然 fp16 下分母会下溢
