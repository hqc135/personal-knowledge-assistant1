


### 为什么需要参数高效微调

全量微调一个 7B 模型需要 ~28GB 显存（fp32）光存参数就够呛，更别提优化器状态（Adam 要额外 2x 参数量）。PEFT 的核心思想：**冻结预训练权重，只训练少量新增参数**，达到接近全量微调的效果。

面试高频考点：能说清楚"为什么 LoRA 有效"比背公式重要。

---

### LoRA（Low-Rank Adaptation）

核心假设：微调时的权重变化矩阵 ΔW 是低秩的。

原始前向：h = Wx
LoRA 前向：h = Wx + BAx

其中 W ∈ R^{d×d} 冻结，B ∈ R^{d×r}，A ∈ R^{r×d}，r << d（通常 r=8 或 16）。

```python
# 伪代码
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16):
        self.W = original_linear.weight  # frozen
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))  # B 初始化为 0！
        self.scaling = alpha / r

    def forward(self, x):
        return F.linear(x, self.W) + (x @ self.A.T @ self.B.T) * self.scaling
```

关键细节：
- **B 初始化为 0**，保证训练开始时 ΔW=0，不破坏预训练表示
- `alpha/r` 这个 scaling 是为了让不同 r 值下学习率不用重新调（这里我一开始理解错了，以为 alpha 是正则项）
- 推理时可以把 BA 合并回 W，**零额外推理开销**

LoRA 一般加在 Attention 的 Q、V 投影上（原论文实验结论），有些实现也加 K 和 FFN。

和 Transformer 注意力机制的联系：LoRA 本质上是在 attention 的投影矩阵上做低秩分解，相当于限制了 attention 能"看到"的子空间维度。

---

### QLoRA

QLoRA = 4-bit 量化基座 + LoRA adapter（fp16/bf16）+ 分页优化器

三个关键技术：
1. **NF4（NormalFloat4）**：信息论最优的 4-bit 数据类型，假设权重服从正态分布
2. **双重量化**：对量化常数本身再做一次量化，省 ~0.4 bit/param
3. **分页优化器**：用 CPU 内存做 swap，处理显存 OOM 的 spike

实际效果：在单张 48GB GPU 上微调 65B 模型。我自己试过在 24GB 3090 上跑 QLoRA 微调 LLaMA-2-13B，batch_size=1 + gradient_checkpointing 刚好能跑。

踩坑记录：QLoRA 训练速度比纯 LoRA 慢不少（~30-40%），因为每次前向都要 dequantize。但显存省太多了，trade-off 值得。

---

### Adapter

最早的 PEFT 方法之一（Houlsby et al., 2019）。在 Transformer 每层插入小型瓶颈模块：

```
x -> LayerNorm -> Down-project (d→r) -> ReLU -> Up-project (r→d) -> + x (residual)
```

和 LoRA 对比记忆：

| | Adapter | LoRA |
|---|---|---|
| 额外推理延迟 | 有（串行模块） | 无（可合并） |
| 参数位置 | 新增模块 | 并行于现有权重 |
| 实现复杂度 | 需改模型结构 | 只需替换 Linear |

Adapter 的致命缺点是推理时有额外延迟，这在 serving 场景下很痛。LoRA 因为可以 merge 回去，部署时和原模型一样。

---

### Prefix-tuning / P-tuning v2

思路完全不同：不改权重，而是在输入前面拼接可学习的"虚拟 token"。

```
input: [P1, P2, ..., Pk, x1, x2, ..., xn]
# P1~Pk 是可学习的连续向量，不对应任何真实 token
```

Prefix-tuning 是在每一层的 KV 前面都拼 prefix（不只是 embedding 层），这样表达能力更强。

和 Prompt Engineering 的关系：Prefix-tuning 可以看作"连续空间中的 prompt 搜索"，而手写 prompt 是在离散 token 空间搜索。这也解释了为什么 prefix-tuning 效果通常优于手工 prompt——连续优化比离散搜索高效得多。

我的理解：Prefix-tuning 在 NLU 任务上效果不错，但在生成任务上通常不如 LoRA 稳定，可能因为 prefix 占用了有限的上下文窗口。

---

### 面试总结 & 选型建议

1. **首选 LoRA/QLoRA**：效果好、部署友好、生态成熟（HuggingFace PEFT 库直接用）
2. **显存极度紧张**：QLoRA，4-bit 量化 + LoRA
3. **多任务快速切换**：LoRA 天然支持（不同任务不同 adapter 权重，base model 共享）
4. **Prefix-tuning**：轻量级场景、参数量要求极小时考虑

一个容易被问到的问题："LoRA 的 rank 怎么选？" 经验值 r=8~64，任务越复杂/数据越多可以适当增大。但 r 太大就失去了 PEFT 的意义，不如全量微调。实际中 r=16, alpha=32 是个不错的起点。

和知识蒸馏的交叉点：PEFT 微调后的小模型 + 蒸馏是目前工业界部署大模型能力的主流 pipeline。先用 PEFT 适配任务，再蒸馏到更小的模型做 serving。