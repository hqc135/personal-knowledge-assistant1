
# 文档分块策略：固定窗口、语义分块、递归分块

## 为什么分块这么重要

RAG 的检索单元就是 chunk。分块太大，噪声多，embedding 被稀释；分块太小，上下文断裂，语义不完整。这是一个 **精确性 vs 完整性** 的 tradeoff。

> 面试高频考点：分块策略直接影响检索质量，是 RAG 系统中最容易被忽视但影响最大的环节之一。

## 固定窗口分块（Fixed-size Chunking）

最简单粗暴的方式：按固定字符数/token 数切分，可以设置 overlap。

```python
def fixed_size_chunk(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # 滑动窗口
    return chunks
```

**优点**：实现简单、chunk 大小可控、便于批处理
**缺点**：可能从句子中间切断、不尊重文档结构

overlap 的作用：防止关键信息恰好在切割边界被截断。一般设 10%-20%。

> 踩过的坑：overlap 设太大会导致大量重复内容被索引，检索时 top-k 里全是同一段话的不同切片，浪费了检索名额。

## 递归分块（Recursive Character Splitting）

LangChain 的默认策略。核心思想：**按优先级尝试不同分隔符**，尽量保持语义完整。

```python
# LangChain 的分隔符优先级
separators = ["\n\n", "\n", " ", ""]

def recursive_split(text, chunk_size, separators):
    # 如果文本已经够短，直接返回
    if len(text) <= chunk_size:
        return [text]
    
    # 尝试用当前最高优先级的分隔符切分
    for sep in separators:
        if sep in text:
            splits = text.split(sep)
            # 合并小块，确保每个 chunk 接近 chunk_size
            chunks = merge_splits(splits, chunk_size, sep)
            # 对超长的 chunk 递归处理（用下一级分隔符）
            return [recursive_split(c, chunk_size, separators[1:]) 
                    if len(c) > chunk_size else c 
                    for c in chunks]
    
    # 所有分隔符都不行，强制按字符切
    return fixed_size_chunk(text, chunk_size)
```

**逻辑**：先尝试按段落切（\n\n），段落太长就按行切（\n），行太长就按空格切，最后才按字符切。

> 这个递归的思路很优雅——和归并排序的分治思想类似，但这里是"尽量在高层级解决问题，解决不了再下沉"。

## 语义分块（Semantic Chunking）

核心思想：**让内容本身决定在哪里切分**，而不是人为设定规则。

### 方法一：基于 embedding 相似度

```python
def semantic_chunk(sentences, threshold=0.75):
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # 计算当前句子和当前 chunk 的相似度
        sim = cosine_similarity(
            embed(sentences[i]), 
            embed(' '.join(current_chunk))
        )
        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            # 相似度低于阈值，开始新 chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
    
    chunks.append(' '.join(current_chunk))
    return chunks
```

### 方法二：基于 LLM 的分块

让 LLM 判断哪里是语义边界。效果好但成本高，一般只在离线处理时用。

### 方法三：Greg Kamradt 的梯度分块

计算相邻句子 embedding 的相似度序列，找到相似度骤降的位置作为切分点（类似边缘检测）。

```
相似度: [0.9, 0.85, 0.88, 0.3, 0.92, 0.87, 0.25, ...]
                              ↑                    ↑
                          切分点               切分点
```

> 这里我一开始理解错了：以为语义分块一定比固定分块好。实际上语义分块的 chunk 大小不可控，可能产生特别短或特别长的 chunk，需要额外处理。而且计算 embedding 本身就有成本。

## 特殊文档的分块策略

### Markdown/HTML

按标题层级切分，保留结构信息：

```python
# 按 ## 标题切分
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
```

### 代码

按函数/类切分，不能从函数中间切断。可以用 AST 解析。

### 表格

表格不能切分，要作为整体保留。如果表格太大，按行切分但每个 chunk 都保留表头。

## 分块大小的选择

| 场景 | 建议大小 | 理由 |
|------|----------|------|
| QA 问答 | 256-512 tokens | 精确匹配，减少噪声 |
| 摘要生成 | 1024-2048 tokens | 需要更多上下文 |
| 对话系统 | 512-1024 tokens | 平衡精确性和上下文 |

> 关键约束：chunk 大小不能超过 embedding 模型的 max_length（通常 512 tokens）。超过的部分会被截断，等于白分了。这和 embedding 模型选型是强耦合的。

## 进阶技巧

### 1. 添加元数据

给每个 chunk 附加来源信息（文件名、页码、标题层级），检索时可以做过滤。

### 2. 父子文档策略

- 用小 chunk 做检索（精确匹配）
- 检索到后返回其父 chunk（更大的上下文）给 LLM

```
Parent chunk (2000 tokens) → 提供给 LLM
  ├── Child chunk 1 (200 tokens) → 用于检索
  ├── Child chunk 2 (200 tokens) → 用于检索
  └── Child chunk 3 (200 tokens) → 用于检索
```

### 3. 命题化分块（Proposition Chunking）

用 LLM 把文档拆成独立的命题（每个命题是一个自包含的事实陈述），然后对命题做 embedding。

> 和混合检索的关系：分块策略会影响 BM25 的效果。太短的 chunk 可能缺少关键词的共现信息，导致 BM25 召回率下降。所以混合检索场景下，分块大小需要同时考虑向量检索和关键词检索的需求。

## 我的实践总结

1. 先用递归分块跑 baseline，大部分场景够用
2. 如果检索质量不行，再尝试语义分块
3. chunk_size 和 overlap 是需要调的超参，没有万能值
4. 一定要看几个实际的 chunk 样本，确认切分是否合理
5. 父子文档策略在实际项目中效果提升明显，推荐尝试