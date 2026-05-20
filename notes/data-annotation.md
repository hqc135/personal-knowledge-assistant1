
# 数据标注与主动学习：标注策略、一致性检验

## 背景

"Garbage in, garbage out" 这句话在 ML 里再真实不过了。模型再花哨，标注质量不行就白搭。尤其是做 NLP（NER、情感分析）和 CV（目标检测）的时候，标注成本高、质量难控制是实际工程中最头疼的问题之一。

## 标注策略设计

### 标注方案的核心要素

1. **标注指南（Annotation Guideline）**：必须写清楚，包含正例、反例、边界 case
2. **标注粒度**：token 级 vs 句子级 vs 文档级
3. **标注人员**：众包 vs 专家标注，trade-off 是成本 vs 质量
4. **质量控制**：金标数据（gold standard）穿插检测

> 这里我一开始理解错了：以为标注指南写一次就够了。实际上是迭代的——先小批量试标，发现歧义 case，更新指南，再标。这个过程可能要 3-4 轮。

### 常见标注模式

```
单人标注 → 便宜但质量不稳定
双人标注 + 仲裁 → 质量好但成本翻倍
多人标注 + 投票 → 适合众包场景

实际工程中常用：
- 核心数据：双人标注 + 专家仲裁
- 大规模数据：单人标注 + 抽样质检（10-20%）
```

### 弱监督和半监督标注

当标注预算有限时：
- **Snorkel 范式**：写 labeling function（规则/启发式），自动生成噪声标签，再用生成模型去噪
- **Self-training**：用已有模型预测未标注数据，取高置信度样本加入训练集
- **数据增强**：对已标注数据做变换（回译、同义词替换等）扩充数据

> 和特征工程的关系：弱监督生成的标签质量直接影响下游特征的有效性。如果标签本身有系统性偏差，学出来的 embedding 特征也会有偏。

## 一致性检验

标注质量的量化指标，面试常问。

### Cohen's Kappa

衡量两个标注者之间的一致性，排除了随机一致的影响：

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- $p_o$：实际一致率（observed agreement）
- $p_e$：随机一致率（expected agreement）

解读：
- κ < 0.2：几乎没有一致性
- 0.2-0.4：一般
- 0.4-0.6：中等
- 0.6-0.8：较好
- 0.8+：几乎完全一致

```python
from sklearn.metrics import cohen_kappa_score

annotator1 = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
annotator2 = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]

kappa = cohen_kappa_score(annotator1, annotator2)
print(f"Cohen's Kappa: {kappa:.3f}")  # 约 0.6
```

### Fleiss' Kappa

多个标注者（>2）的一致性度量。和 Cohen's Kappa 的区别是它不要求每个样本都由相同的标注者标注。

> **面试高频考点**：Cohen's Kappa vs Fleiss' Kappa 的适用场景。前者是两人，后者是多人。

### Krippendorff's Alpha

最通用的一致性指标：
- 支持任意数量标注者
- 支持缺失数据
- 支持不同数据类型（名义、有序、区间、比率）

实际项目中如果只能选一个指标，我倾向于用 Krippendorff's Alpha。

## 主动学习（Active Learning）

核心思想：不是随机选数据标注，而是让模型"主动"选择最有价值的样本去标注，用最少的标注量达到最好的效果。

### 基本流程

```
初始：少量标注数据训练初始模型
循环：
    1. 用当前模型对未标注池打分
    2. 按照查询策略选择最有价值的 batch
    3. 人工标注这批数据
    4. 加入训练集，重新训练模型
    直到达到性能目标或预算用完
```

### 查询策略（Query Strategy）

**不确定性采样（Uncertainty Sampling）**：
- Least Confidence：选模型最不确定的样本
- Margin Sampling：选 top-2 预测概率差最小的
- Entropy Sampling：选预测熵最大的

```python
import numpy as np

def uncertainty_sampling(model, X_pool, n_samples=100):
    probs = model.predict_proba(X_pool)
    
    # Entropy-based
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # 选 entropy 最大的 n_samples 个
    query_indices = np.argsort(entropy)[-n_samples:]
    return query_indices
```

**多样性采样（Diversity Sampling）**：
- 只选不确定的可能导致选出的样本都很相似（都在决策边界附近）
- 结合聚类，保证选出的样本覆盖不同区域

**混合策略**：不确定性 + 多样性，实际效果最好。

> 这里和 Transformer 里的 attention 有个有趣的类比：attention 是让模型"关注"最重要的 token，主动学习是让标注者"关注"最有价值的样本。都是一种资源分配的优化。

### 实际踩坑

1. **冷启动**：初始模型太差，不确定性采样退化为随机采样。解决：先随机标一批 bootstrap
2. **标注延迟**：标注不是实时的，模型可能已经更新了。解决：batch mode active learning
3. **类别不平衡**：模型对少数类永远不确定，导致一直选少数类。解决：加入类别平衡约束
4. **标注者疲劳**：连续标注困难样本会降低标注质量。解决：混入简单样本

## 工程实践

### 标注平台选型

- Label Studio：开源，支持多种任务类型
- Prodigy：spaCy 团队出品，NLP 友好
- CVAT：CV 标注专用
- 自建：大厂通常自建，和内部系统集成

### 标注质量监控 Pipeline

```
每日监控：
├── 标注速度异常检测（太快 = 可能在乱标）
├── 一致性指标追踪（Kappa 趋势）
├── 金标数据准确率
└── 标注者间差异分析
```

> 和 ETL Pipeline 的关系：标注质量监控本质上也是一个数据质量监控问题，可以集成到 Airflow DAG 里，每天自动跑一致性检验报告。

## 面试准备

常见问题：
- "标注预算有限怎么办？" → 主动学习 + 弱监督 + 数据增强
- "怎么评估标注质量？" → Kappa 系数 + 金标抽检
- "主动学习在实际中效果如何？" → 通常能减少 30-50% 标注量达到同等效果，但有冷启动和工程复杂度的代价