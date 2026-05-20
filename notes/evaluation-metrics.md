# 评估指标：BLEU、ROUGE、BERTScore、RAGAS 评估框架

## 为什么需要这些指标

做 NLG 任务（翻译、摘要、RAG 问答）时，人工评估太贵太慢，需要自动化指标。但不同指标衡量的维度完全不同，面试时要能说清楚每个指标"在度量什么"以及"什么场景下会失效"。

## BLEU（Bilingual Evaluation Understudy）

核心思想：计算生成文本和参考文本之间的 **n-gram 精确率**（precision）。

```
BLEU = BP × exp(Σ wₙ log pₙ)

其中：
- pₙ = modified n-gram precision（用 clipping 防止重复词刷分）
- BP = brevity penalty，惩罚过短的生成结果
- wₙ 通常取均匀权重 1/N
```

我的理解：BLEU 本质上在问"生成的词组有多少出现在参考答案里"，是 precision 导向的。所以它**不关心召回**——你漏掉了重要信息它不管。

踩过的坑：
- BLEU 是语料级别（corpus-level）指标，单句 BLEU 波动很大，不太靠谱
- 一开始我以为 modified precision 就是普通 precision，其实它对每个 n-gram 的计数做了 clip（取 min），防止生成 "the the the..." 刷高分

面试高频考点：BLEU 衡量精确率，ROUGE 衡量召回率，这是最核心的区别。

## ROUGE（Recall-Oriented Understudy for Gisting Evaluation）

专为摘要任务设计，关注**召回率**——参考文本中的 n-gram 有多少被生成文本覆盖了。

常见变体：
- **ROUGE-N**：n-gram 召回率（ROUGE-1, ROUGE-2 最常用）
- **ROUGE-L**：基于最长公共子序列（LCS），不要求连续匹配
- **ROUGE-Lsum**：对多句摘要按句分别算 LCS 再汇总

```
ROUGE-N recall = |matched n-grams| / |n-grams in reference|
```

和 BLEU 对比记忆：BLEU 是"我生成的东西对不对"（precision），ROUGE 是"参考答案的内容我覆盖了多少"（recall）。实际使用中 ROUGE 也会报 F1。

## BERTScore

上面两个都是基于字面匹配的，同义词换一下分数就掉。BERTScore 用预训练模型的 embedding 做 **token 级别的语义相似度匹配**。

流程：
1. 用 BERT 对 candidate 和 reference 分别编码，得到每个 token 的 contextual embedding
2. 对 candidate 中每个 token，找 reference 中余弦相似度最高的 token（贪心匹配）
3. 汇总得到 Precision、Recall、F1

```python
# 伪代码
sim_matrix = cosine_similarity(candidate_embeddings, reference_embeddings)
P = mean(max(sim_matrix, dim=1))  # 每个 candidate token 的最佳匹配
R = mean(max(sim_matrix, dim=0))  # 每个 reference token 的最佳匹配
F1 = 2 * P * R / (P + R)
```

这里我一开始理解错了：BERTScore 不是句子级别的 cosine similarity（那是 SentenceBERT 干的事），而是 token 级别的贪心对齐，粒度更细。

和 Transformer 注意力机制的联系：BERTScore 的 sim_matrix 本质上就是一个 cross-attention 矩阵，只不过它用 max pooling 而不是 softmax 来做聚合。

## RAGAS 评估框架

这是专门为 **RAG（Retrieval-Augmented Generation）** 系统设计的评估框架，不是单一指标而是一组指标。面试中如果聊到 RAG 系统评估，RAGAS 基本是必提的。

核心维度：

| 指标 | 衡量什么 | 输入 |
|------|----------|------|
| Faithfulness | 生成的答案是否忠于检索到的上下文（不幻觉） | answer + contexts |
| Answer Relevancy | 答案和问题的相关性 | question + answer |
| Context Precision | 检索到的上下文中，排在前面的是否更相关 | question + contexts + ground_truth |
| Context Recall | 参考答案中的信息是否都能在检索上下文中找到 | contexts + ground_truth |

关键设计思路：RAGAS 用 LLM-as-Judge 来做评估（比如让 GPT-4 判断每个 claim 是否被 context 支持），所以它本身也有评估噪声。

```python
# Faithfulness 计算伪代码
claims = llm_extract_claims(answer)
supported = [c for c in claims if llm_verify(c, contexts)]
faithfulness = len(supported) / len(claims)
```

我的理解：RAGAS 把 RAG 的评估拆成了"检索质量"和"生成质量"两个独立维度，这个解耦很重要。检索烂但生成好（靠模型自身知识回答）或者检索好但生成烂（幻觉）都能被分别捕捉到。

和 RAG 系统设计的交叉引用：Context Precision 直接关联到检索阶段的 reranking 策略——如果你用了 Cross-Encoder 做重排序，这个指标应该会显著提升。

## 指标选择的实践建议

- 机器翻译：BLEU（行业惯例） + BERTScore（语义补充）
- 文本摘要：ROUGE-L + BERTScore
- RAG 问答：RAGAS 全家桶，重点看 Faithfulness（幻觉是 RAG 的核心痛点）
- 开放式对话：这些指标都不太行，更多靠人工评估或 LLM-as-Judge

## 面试要点速记

1. BLEU = n-gram precision + brevity penalty，语料级指标
2. ROUGE = n-gram recall，摘要任务标配
3. BERTScore = token 级语义匹配，解决同义词问题
4. RAGAS = RAG 专用，拆解为 faithfulness / relevancy / context quality
5. 所有基于字面匹配的指标都有一个根本缺陷：语义等价但表述不同时会低估质量
6. LLM-as-Judge 是趋势，但引入了评估者自身的 bias（位置偏好、冗长偏好等）
