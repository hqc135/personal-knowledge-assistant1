
# 多模态模型：CLIP、LLaVA、视觉-语言对齐

> 写于面试准备期间，主要参考了 OpenAI CLIP 论文、LLaVA 系列论文和李沐的精读视频

## 核心思想：为什么要做视觉-语言对齐

传统 CV 模型（ResNet、ViT）学到的是"图像→标签"的映射，但这个标签空间是封闭的。多模态的核心目标是让视觉和语言共享一个语义空间，这样就能做 zero-shot 分类、图文检索、视觉问答等开放任务。

**我的理解**：可以类比数据库里的 JOIN 操作——两张表（图像表和文本表）通过一个共同的 key（对齐后的 embedding）关联起来。

## CLIP：对比学习做对齐

### 架构

```
Image Encoder (ViT/ResNet) → image embedding (d维)
Text Encoder (Transformer)  → text embedding (d维)

对比损失：最大化配对的 (image, text) 余弦相似度，最小化非配对的
```

### 关键公式

给定 batch 中 N 个 (image, text) 对：

```
sim(i, j) = cos(image_embed_i, text_embed_j) / τ

L_i2t = -log( exp(sim(i,i)) / Σ_j exp(sim(i,j)) )  # 对每个图像
L_t2i = -log( exp(sim(i,i)) / Σ_j exp(sim(j,i)) )  # 对每个文本
L = (L_i2t + L_t2i) / 2
```

这就是 InfoNCE loss，和自监督学习里 SimCLR 的损失函数本质一样。**面试高频考点：CLIP 的损失函数怎么写、为什么用对比学习而不是生成式。**

### 我踩过的坑

一开始以为 CLIP 的 text encoder 是 GPT 系列的 decoder，其实不是——它用的是类似 BERT 的 Transformer encoder，取 [EOS] token 的输出作为句子表示。这个细节面试被问过。

### 为什么 CLIP 这么强

- 4 亿图文对的 web 数据（WIT-400M），数据规模碾压
- 对比学习天然适合大 batch（CLIP 用了 32768 的 batch size）
- 文本作为监督信号比离散标签丰富得多

## LLaVA：把视觉接入 LLM

CLIP 解决了对齐问题，但它不能"对话"。LLaVA 的思路很直接：

```
图像 → CLIP ViT → visual tokens → 投影层(MLP) → 和 text tokens 拼接 → 送入 LLM (Vicuna/LLaMA)
```

### 训练分两阶段

1. **预训练阶段**：冻结 ViT 和 LLM，只训练投影层（MLP），用 image-caption 数据
2. **指令微调阶段**：冻结 ViT，微调投影层 + LLM，用 GPT-4 生成的视觉指令数据

**这里我一开始理解错了**：以为两阶段都要全量微调 LLM，实际上第一阶段 LLM 完全冻结，只学一个线性映射。这样设计是因为第一阶段的目标只是让 visual tokens "说 LLM 能听懂的语言"。

### 和 Transformer 注意力机制的关系

LLaVA 里 visual tokens 和 text tokens 的交互完全依赖 LLM 内部的 self-attention。视觉信息没有特殊通道，就是作为普通 token 参与注意力计算。这和 ViT 把图像 patch 当 token 的思路一脉相承——**统一架构，让 attention 自己学交互模式**。

## 视觉-语言对齐的几种范式对比

| 方法 | 对齐方式 | 代表模型 | 能力 |
|------|----------|----------|------|
| 双塔对比 | 共享 embedding 空间 | CLIP, ALIGN | 检索、zero-shot 分类 |
| 融合编码 | cross-attention | ALBEF, CoCa | 更细粒度的理解 |
| LLM 接入 | 投影到 LLM 输入空间 | LLaVA, MiniGPT-4 | 视觉对话、推理 |

## 和其他主题的交叉

1. **和 RAG 的关系**：多模态检索可以看作视觉版的 RAG——用 CLIP 检索相关图像，再送入 LLaVA 做问答。这在文档理解场景很实用。
2. **和 Agent 框架的关系**：多模态 Agent（如 GPT-4V + tool use）需要视觉理解能力作为"感知模块"，LLaVA 这类模型就是 Agent 的"眼睛"。
3. **和知识蒸馏的关系**：LLaVA 用 GPT-4 生成训练数据，本质上是一种知识蒸馏——把大模型的推理能力蒸馏到多模态模型中。

## 面试常见问题

- CLIP 为什么不用生成式损失？→ 对比学习计算效率高，且不需要逐 token 生成
- LLaVA 的投影层为什么用 MLP 而不是更复杂的结构？→ 实验表明简单 MLP 就够了（LLaVA-1.5 用两层 MLP），复杂结构收益不大
- 如何评估多模态模型的幻觉问题？→ POPE benchmark、CHAIR 指标

## 遗留疑问

- 视觉 token 数量（如 576 个）对 LLM 的上下文窗口压力很大，动态分辨率方案（如 LLaVA-NeXT）是怎么平衡效率和效果的？
- CLIP 的对齐是粗粒度的（整图-整句），细粒度对齐（区域-短语）有没有更优雅的方案？
