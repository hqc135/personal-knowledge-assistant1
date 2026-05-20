
# 分词算法：BPE、WordPiece、SentencePiece

> 分词看起来是个"底层"问题，但实际上直接影响模型性能、多语言能力、甚至推理成本。面试中经常作为"基础功"来考。

## 为什么需要子词分词？

两个极端：
- **字符级**：词表小（几百），但序列太长，语义信息稀疏
- **词级**：语义清晰，但词表爆炸（几十万），OOV（未登录词）问题严重

**子词分词（Subword Tokenization）** 是折中：常见词保持完整，罕见词拆成子词片段。

```
"unhappiness" → ["un", "happiness"]  或 ["un", "happ", "iness"]
"tokenization" → ["token", "ization"]
```

## BPE（Byte Pair Encoding）

### 原始算法（GPT 系列使用）

来自数据压缩领域，核心思想：**反复合并最频繁的相邻字符对**。

```python
# BPE 训练过程
vocab = set(all_characters)
corpus = split_into_characters(training_data)

for i in range(num_merges):
    # 统计所有相邻 pair 的频率
    pairs = count_pairs(corpus)
    # 找到最频繁的 pair
    best_pair = argmax(pairs)
    # 合并这个 pair 为新 token
    vocab.add(merge(best_pair))
    corpus = apply_merge(corpus, best_pair)
```

### 具体例子

```
初始词表: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', ...]
语料: "low lower newest"

Step 1: 最频繁 pair ('l','o') → 合并为 'lo'
Step 2: 最频繁 pair ('lo','w') → 合并为 'low'
Step 3: 最频繁 pair ('e','s') → 合并为 'es'
Step 4: 最频繁 pair ('es','t') → 合并为 'est'
...
```

### BPE 分词（推理时）

训练好后，对新文本分词：按照合并规则的**学习顺序**，贪心地应用合并。

```python
def tokenize_bpe(word, merges):
    tokens = list(word)
    for merge_rule in merges:  # 按学习顺序
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == merge_rule:
                tokens[i] = tokens[i] + tokens[i+1]
                del tokens[i+1]
            else:
                i += 1
    return tokens
```

### Byte-level BPE（GPT-2/3/4 使用）

普通 BPE 的问题：基础字符集依赖语言，遇到未知字符就 GG。

解决方案：以 **byte**（256 个）作为基础单元，任何 UTF-8 文本都能表示。

```
优点：永远不会有 OOV
缺点：非 ASCII 字符（如中文）会被拆成多个 byte token，效率低
```

这就是为什么 GPT 系列处理中文时 token 数量比英文多很多——一个汉字通常被编码为 2-3 个 byte token。**面试高频考点**。

## WordPiece

### Google 提出（BERT 使用）

和 BPE 很像，但合并策略不同：

- BPE：合并**频率最高**的 pair
- WordPiece：合并能**最大化语言模型似然**的 pair

```
合并标准: score(pair) = freq(pair) / (freq(first) * freq(second))
```

直觉：不只看 pair 出现多少次，还要看组成 pair 的两个 token 各自有多频繁。如果两个本身就很频繁的 token 经常一起出现，那合并它们的收益更大。

### WordPiece 分词（推理时）

用**最长匹配**（longest match first）策略：

```python
def tokenize_wordpiece(word, vocab):
    tokens = []
    start = 0
    while start < len(word):
        end = len(word)
        found = False
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = "##" + substr  # 非首字符加 ## 前缀
            if substr in vocab:
                tokens.append(substr)
                found = True
                break
            end -= 1
        if not found:
            tokens.append("[UNK]")
            start += 1
        else:
            start = end
    return tokens
```

注意 `##` 前缀：表示这个 token 不是词的开头。比如 "playing" → ["play", "##ing"]。

### BPE vs WordPiece 对比

| | BPE | WordPiece |
|---|---|---|
| 合并策略 | 频率最高 | 似然增益最大 |
| 分词策略 | 按合并顺序贪心 | 最长匹配 |
| 代表模型 | GPT 系列 | BERT |
| 子词标记 | 无特殊标记（或用 Ġ 表示空格） | ## 前缀 |

这里我一开始搞混了：训练时的合并策略和推理时的分词策略是两回事。BPE 训练和推理都用合并规则，WordPiece 训练用似然但推理用最长匹配。

## SentencePiece

### 解决的问题

BPE 和 WordPiece 都假设输入已经被**预分词**（pre-tokenized）了——即已经按空格分成了词。这对中文、日文等没有空格的语言不友好。

SentencePiece 的做法：**把空格也当作普通字符处理**（用 ▁ 替代空格），直接在原始文本上训练。

```
"Hello World" → ["▁Hello", "▁World"]
"今天天气好" → ["▁今天", "天气", "好"]
```

### 两种模式

SentencePiece 支持两种子词算法：
1. **BPE 模式**：和标准 BPE 一样，但在原始文本上操作
2. **Unigram 模式**：完全不同的思路

### Unigram Language Model

和 BPE 相反的方向：
- BPE：从小词表开始，逐步合并（bottom-up）
- Unigram：从大词表开始，逐步删除（top-down）

```python
# Unigram 训练
vocab = initialize_large_vocab()  # 比如所有出现过的子串
for _ in range(iterations):
    # 用 EM 算法估计每个 token 的概率
    probs = estimate_unigram_probs(vocab, corpus)
    # 计算删除每个 token 后 loss 的增加量
    losses = compute_removal_loss(vocab, probs)
    # 删除 loss 增加最小的 token（保留重要的）
    vocab = remove_least_important(vocab, losses, keep_ratio=0.8)
```

分词时用 **Viterbi 算法**找到概率最大的分词方式：

```
P(tokenization) = argmax ∏ P(token_i)
```

**和 Transformer 的联系**：分词结果直接决定了输入序列的长度，进而影响注意力计算的复杂度（O(n²)）。一个好的分词器应该在词表大小和序列长度之间找到平衡。这和推理优化笔记中讨论的 KV Cache 显存开销直接相关——token 越多，cache 越大。

## 实际选择

| 模型 | 分词器 | 词表大小 |
|---|---|---|
| GPT-2 | Byte-level BPE | 50,257 |
| GPT-4 | Byte-level BPE (cl100k) | 100,256 |
| BERT | WordPiece | 30,522 |
| LLaMA | SentencePiece (BPE) | 32,000 |
| LLaMA-3 | Byte-level BPE (tiktoken) | 128,256 |

趋势：词表越来越大。大词表 → 序列更短 → 推理更快，但 embedding 层参数更多。

## 踩过的坑和注意事项

1. **分词不可逆的情况**：某些 tokenizer 的 decode(encode(text)) ≠ text，特别是处理特殊字符时
2. **中文分词效率**：早期模型（GPT-2）对中文极不友好，一个汉字可能 3 个 token。新模型（GPT-4、LLaMA-3）通过扩大词表和加入中文语料大幅改善
3. **数字处理**：很多 tokenizer 把数字逐位拆分（"2024" → ["2","0","2","4"]），这对数学推理很不利。有些模型专门优化了数字的 tokenization
4. **特殊 token**：[CLS]、[SEP]、<|endoftext|> 等，不同模型不同，容易搞混

## 面试常见追问

- Q：为什么不用字符级？→ 序列太长，注意力 O(n²)，且每个字符的语义信息太少
- Q：词表大小怎么选？→ trade-off：大词表序列短但参数多，通常 32K-128K
- Q：BPE 的时间复杂度？→ 训练 O(n × num_merges)，推理 O(n × |merges|)，实际用优化数据结构可以更快
- Q：如何处理新词/新领域？→ 子词天然支持（拆成已知片段），但效率可能低。可以扩展词表（vocabulary expansion）但需要重新训练 embedding
