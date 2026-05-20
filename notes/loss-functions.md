
## 损失函数大全：交叉熵、对比损失、Focal Loss、InfoNCE

### 总览

损失函数定义了"模型输出和目标之间的差距"，选对损失函数比调超参重要得多。按任务类型分：

- **分类**：交叉熵、Focal Loss
- **表示学习/对比学习**：对比损失、Triplet Loss、InfoNCE
- **回归**：MSE、MAE、Huber Loss
- **生成**：重构损失 + KL 散度（VAE）、对抗损失（GAN）

---

## 交叉熵（Cross-Entropy）

最基础也最重要的分类损失。

二分类：
```
L = -[y * log(p) + (1-y) * log(1-p)]
```

多分类：
```
L = -Σ y_i * log(p_i)    # y 是 one-hot，p 是 softmax 输出
  = -log(p_correct)       # 简化：只有正确类别那一项非零
```

> **面试高频考点**：交叉熵和 KL 散度的关系？
> 
> `H(p, q) = H(p) + D_KL(p || q)`
> 
> 当 p 是固定的标签分布时，最小化交叉熵 = 最小化 KL 散度。

实际使用注意：
- PyTorch 的 `CrossEntropyLoss` = `LogSoftmax + NLLLoss`，输入是 logits 不是概率
- 数值稳定性：不要自己先算 softmax 再取 log，会有精度问题

> 和 Label Smoothing 的结合（正则化笔记中提到）：把 one-hot 的 1 变成 1-ε，0 变成 ε/(K-1)，防止模型输出过于极端的 logits。

---

## Focal Loss

解决类别不平衡问题（来自 RetinaNet 论文）：

```
FL = -α_t * (1 - p_t)^γ * log(p_t)
```

其中 p_t 是模型对正确类别的预测概率。

核心思想：给"容易分对的样本"降权。当 p_t 接近 1 时，`(1-p_t)^γ` 接近 0，这个样本的 loss 贡献很小。

```python
# γ=0 时退化为标准交叉熵
# γ=2 是论文推荐值

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    p_t = torch.exp(-ce_loss)  # 正确类别的概率
    focal_weight = (1 - p_t) ** gamma
    return (alpha * focal_weight * ce_loss).mean()
```

> 我的理解：Focal Loss 本质上是一种"课程学习"——让模型把注意力放在难样本上。但要注意 γ 太大会导致训练不稳定（太关注噪声样本了）。

适用场景：目标检测（正负样本比例极端，如 1:1000）、医学影像（阳性样本少）

---

## 对比损失（Contrastive Loss）

来自 Siamese Network，目标是让相似样本靠近、不相似样本远离：

```
L = y * d^2 + (1-y) * max(0, margin - d)^2
```

其中 d 是两个样本嵌入的距离，y=1 表示同类，y=0 表示异类。

> 直觉：同类样本距离越小越好；异类样本距离只要超过 margin 就不再惩罚（已经够远了）。

**Triplet Loss** 是对比损失的改进：

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

> 踩坑：Triplet Loss 对 triplet 的采样策略非常敏感。随机采样大部分 triplet 的 loss 为 0（太简单了），需要 hard negative mining 或 semi-hard mining 才能有效训练。

---

## InfoNCE

对比学习的核心损失（SimCLR、CLIP、MoCo 都用它）：

```
L = -log( exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
```

其中：
- z_i, z_j 是正样本对的表示
- 分母对所有负样本求和
- τ 是温度参数
- sim 通常是余弦相似度

```python
# 简化实现（batch 内负样本）
def info_nce_loss(features, temperature=0.07):
    # features: [2N, D]，N 个正样本对
    sim_matrix = F.cosine_similarity(features.unsqueeze(0), 
                                      features.unsqueeze(1), dim=-1)
    sim_matrix = sim_matrix / temperature
    
    # 正样本对的 mask
    # ... (构造 labels)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
```

> **关键洞察**：InfoNCE 本质上就是一个 (2N)-分类的交叉熵！把"找到正样本"当作分类问题。这和 Transformer 中 attention 的 softmax 归一化在形式上很像——都是在一组候选中做 soft selection。

温度 τ 的作用：
- τ 小：分布更尖锐，模型更关注 hard negatives
- τ 大：分布更平滑，所有负样本权重更均匀
- CLIP 用的是可学习的温度参数

> 和 softmax 温度的联系：LLM 推理时的 temperature 和这里的 τ 本质相同——控制概率分布的"锐度"。

---

## 其他值得了解的损失

**KL 散度**（VAE 中的正则项）：
```
D_KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ^2) - μ^2 - σ^2)
```

**Hinge Loss**（SVM）：
```
L = max(0, 1 - y * f(x))
```

**CTC Loss**（语音识别、OCR）：处理输入输出不等长的对齐问题

---

## 面试常见问题

1. **为什么不用 MSE 做分类？**
   - MSE 对分类的梯度在预测值接近 0 或 1 时很小（sigmoid 饱和区），训练慢
   - 交叉熵的梯度 = p - y，简洁且不会消失

2. **InfoNCE 的负样本数量为什么重要？**
   - 负样本越多，对比信号越强，学到的表示越好
   - 但太多负样本可能引入 false negatives（实际是正样本但被当负样本）
   - MoCo 用 momentum queue 解决这个问题

3. **温度参数怎么选？**
   - 通常 0.05-0.1 之间
   - 太小容易训练不稳定，太大学不到有区分度的表示
   - 可以当超参搜索，也可以像 CLIP 一样学习