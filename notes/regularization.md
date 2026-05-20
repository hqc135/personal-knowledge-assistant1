
## 正则化方法：Dropout、L1/L2、BatchNorm、LayerNorm

### 为什么需要正则化

一句话：防止模型"背答案"（过拟合）。正则化的本质是给模型加约束，让它学到更泛化的特征而不是记住训练数据的噪声。

---

## L1 和 L2 正则化

在损失函数上加惩罚项：

```
L_total = L_original + λ * R(θ)

L1: R(θ) = Σ|θ_i|          → 稀疏解（很多权重变成 0）
L2: R(θ) = Σθ_i^2          → 权重趋向小值但不为 0
```

> **面试高频考点**：为什么 L1 能产生稀疏解？
> 
> 几何直觉：L1 的约束区域是菱形，等高线和菱形的交点大概率在坐标轴上（某个维度为 0）。L2 是圆形，交点一般不在轴上。

实际使用：
- L1 用于特征选择（比如 LASSO 回归）
- L2 更常用于深度学习（就是 weight decay）
- 在 Adam 优化器下，L2 正则 ≠ weight decay（见优化器笔记中 AdamW 的讨论）

---

## Dropout

训练时随机"杀死"一部分神经元（置零），测试时全部保留但输出乘以 (1-p)。

```python
# 训练时（inverted dropout，更常用）
mask = (torch.rand(h.shape) > p).float()
h = h * mask / (1 - p)   # 除以 (1-p) 保证期望不变

# 测试时：什么都不做
```

> 我一开始搞混了两种实现：
> - 标准 Dropout：训练时直接 mask，测试时乘 (1-p)
> - Inverted Dropout：训练时 mask 后除以 (1-p)，测试时不变
> 
> PyTorch 用的是 inverted 版本，这样测试时不需要额外操作。

Dropout 为什么有效？几种理解角度：
1. **集成学习**：每次 forward 相当于一个不同的子网络，最终效果类似 ensemble
2. **打破共适应**：防止神经元之间形成过强的依赖关系
3. **噪声注入**：类似数据增强，增加训练的随机性

> 和注意力机制的联系：Transformer 里 attention weights 上也会加 dropout（attention dropout），防止模型过度依赖某几个 token 的注意力。

---

## BatchNorm（批归一化）

对每个 mini-batch 内的特征做归一化：

```python
# 训练时
μ_B = mean(x, dim=batch)
σ_B = std(x, dim=batch)
x_hat = (x - μ_B) / (σ_B + ε)
y = γ * x_hat + β   # 可学习的缩放和偏移

# 测试时：用训练过程中的 running mean/var
```

BatchNorm 的好处：
- 缓解内部协变量偏移（Internal Covariate Shift）—— 虽然后来有论文质疑这个解释
- 允许更大的学习率
- 有轻微正则化效果（因为 batch 统计量有噪声）
- 加速收敛

> **踩坑**：batch size 太小时 BatchNorm 效果很差（统计量不准）。这就是为什么 GN（Group Norm）在检测任务中更受欢迎——检测任务 batch size 通常很小。

---

## LayerNorm（层归一化）

对单个样本的所有特征做归一化（不依赖 batch）：

```python
# 对每个样本独立计算
μ = mean(x, dim=features)
σ = std(x, dim=features)
x_hat = (x - μ) / (σ + ε)
y = γ * x_hat + β
```

**BatchNorm vs LayerNorm 对比**（面试必问）：

| 维度 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化方向 | batch 维度 | feature 维度 |
| batch size 依赖 | 是（小 batch 不稳定） | 否 |
| 训练/推理一致性 | 不一致（running stats） | 一致 |
| 主要用途 | CNN | Transformer / RNN |
| 序列长度可变 | 不方便 | 方便 |

> 为什么 Transformer 用 LayerNorm 而不是 BatchNorm？
> 1. NLP 中 batch 内序列长度不同，batch 统计量意义不明确
> 2. 自回归生成时 batch size = 1，BatchNorm 直接废了
> 3. LayerNorm 对每个 token 独立归一化，和 self-attention 的 token-wise 计算天然匹配

---

## Pre-Norm vs Post-Norm

Transformer 中 LayerNorm 放在哪里也有讲究：

```
Post-Norm (原始 Transformer):  x + Sublayer(LayerNorm(x))  ← 不对
实际是：LayerNorm(x + Sublayer(x))

Pre-Norm (GPT-2 之后主流):  x + Sublayer(LayerNorm(x))
```

> 这里我之前记反了。Pre-Norm 训练更稳定（梯度流更好），但理论上 Post-Norm 收敛后效果可能更好。现在大模型基本都用 Pre-Norm，有些用 RMSNorm（去掉均值，只做方差归一化，省计算）。

---

## 其他正则化手段（简要）

- **数据增强**：最朴素但最有效的正则化
- **Early Stopping**：验证集 loss 不降就停
- **Label Smoothing**：把 one-hot 标签软化，防止模型过于自信（和交叉熵损失函数的讨论相关）
- **Mixup / CutMix**：样本级别的数据增强
- **Stochastic Depth**：随机跳过某些层（ResNet 中用）

---

## 面试总结

正则化方法的选择不是互斥的，实际训练中经常组合使用：
- CV：BatchNorm + Dropout + 数据增强 + weight decay
- NLP/LLM：LayerNorm + Dropout（小模型）+ weight decay
- 大模型趋势：Dropout 用得越来越少（数据量够大时过拟合不是主要问题），但 weight decay 和 LayerNorm 是标配