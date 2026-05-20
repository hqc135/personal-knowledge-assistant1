
# 特征工程：特征选择、特征交叉、Embedding 特征

## 为什么特征工程重要

Andrew Ng 那句老话："Applied ML is basically feature engineering." 虽然深度学习时代大家觉得端到端学习能自动提特征，但在推荐系统、广告、风控这些工业场景里，特征工程依然是模型效果的胜负手。面试里问到"你怎么提升模型效果"，第一反应不应该是换模型，而是看特征。

## 特征选择

核心问题：特征太多 → 维度灾难、过拟合、训练慢。需要筛掉噪声特征。

三大类方法：

| 类别 | 方法 | 特点 |
|------|------|------|
| Filter | 方差阈值、互信息、卡方检验 | 快，但不考虑特征间交互 |
| Wrapper | 前向选择、后向消除、RFE | 效果好但计算量大 |
| Embedded | L1 正则、树模型 feature_importance | 训练过程中自动选择 |

**面试高频考点**：L1 为什么能做特征选择而 L2 不行？

直觉理解：L1 的等高线是菱形，和损失函数的等高线更容易在坐标轴上相交（某些权重恰好为 0）。L2 是圆形，交点一般不在轴上，所以权重趋近于 0 但不等于 0。

```python
# sklearn 里用 L1 做特征选择
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.01)
selector = SelectFromModel(lasso)
X_selected = selector.fit_transform(X, y)
```

> 这里我一开始理解错了：以为互信息只能处理离散特征，其实 sklearn 的 `mutual_info_regression` 用 KNN 估计可以处理连续特征。

## 特征交叉

单个特征的表达能力有限，特征交叉能捕获非线性关系。

**手动交叉**：比如推荐系统里 `user_age × item_category`，表达"年轻人喜欢某类商品"这种模式。

**自动交叉的演进**：
- FM (Factorization Machine)：二阶交叉，每个特征学一个 embedding 向量，交叉 = 内积
- DeepFM：FM 处理显式交叉 + DNN 处理隐式交叉
- DCN (Deep & Cross Network)：Cross Network 自动学习有界阶的特征交叉

FM 的核心公式：

$$\hat{y} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j$$

计算技巧（面试必问）：朴素实现是 O(n²k)，但可以化简为 O(nk)：

$$\sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left(\sum_{i=1}^n v_{if} x_i\right)^2 - \sum_{i=1}^n v_{if}^2 x_i^2 \right]$$

> 和 Transformer 的注意力机制对比：Attention 本质上也是在做特征间的交互（Q·K），只不过是 token 级别的。FM 的交叉和 self-attention 的 scaled dot-product 在数学形式上很像。

## Embedding 特征

高基数类别特征（比如 user_id、item_id）没法 one-hot，必须用 embedding。

**核心思想**：把离散 ID 映射到低维稠密向量空间，语义相近的实体在空间中距离近。

几种获取 embedding 的方式：
1. **端到端训练**：和模型一起训练（推荐系统里最常见）
2. **预训练**：Word2Vec / Item2Vec 思路，先在行为序列上训练
3. **图 Embedding**：Node2Vec、GraphSAGE，利用图结构信息

```python
# PyTorch embedding 层
import torch.nn as nn

# 假设有 100 万用户，映射到 64 维
user_embedding = nn.Embedding(num_embeddings=1000000, embedding_dim=64)

# 查表
user_ids = torch.LongTensor([42, 1337, 9999])
vectors = user_embedding(user_ids)  # shape: (3, 64)
```

**踩过的坑**：
- 冷启动问题：新用户/新物品没有 embedding。解决方案：用 side information（年龄、类目等）过一个小网络生成初始 embedding
- Embedding 维度选择：经验公式 `dim ≈ min(50, n_categories^0.25)`，但实际还是要调
- 训练时要注意 embedding 的学习率通常要比上层网络小，否则容易震荡

> 这个和 NLP 里 Word2Vec 的思路完全一致——都是用上下文（共现关系）来学习分布式表示。推荐系统里用户的行为序列就相当于 NLP 里的句子。

## 工程实践中的注意事项

1. **特征穿越（data leakage）**：用了未来信息。比如用"用户是否最终购买"作为中间特征
2. **特征一致性**：线上线下特征计算逻辑必须一致，否则 offline 效果好 online 效果差
3. **特征存储**：高频更新的特征用 Redis/Feature Store，低频的可以批量计算

> 和 ETL Pipeline 的关系：特征计算本质上就是 ETL 的一部分，Airflow 里经常有专门的 feature pipeline DAG 来保证特征的时效性和质量。

## 面试常见问题

- "你项目里做过哪些特征工程？效果提升多少？" → 准备 2-3 个具体例子
- "特征选择和 PCA 降维有什么区别？" → 特征选择保留原始特征的可解释性，PCA 是线性变换生成新特征
- "Embedding 维度怎么选？" → 经验公式 + 实验验证
