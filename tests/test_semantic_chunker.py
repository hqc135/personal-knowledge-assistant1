"""
单元测试：SemanticChunker 语义分块逻辑

完全使用 Mock Embedder，不依赖真实 API，可离线运行。
"""
import numpy as np
import pytest

from data_pipeline import SemanticChunker


# ──────────────────────────────────────────────────────────────────────────────
# Mock Embedder：可控的假向量，用于精确测试断点检测逻辑
# ──────────────────────────────────────────────────────────────────────────────

class MockEmbedder:
    """
    给定一组预设向量序列，按顺序依次返回。
    encode([s1, s2, ...]) → np.ndarray (N, D)
    """
    def __init__(self, vectors: np.ndarray):
        self._vectors = vectors  # shape (N, D)
        self._call_count = 0

    def encode(self, texts, normalize_embeddings=True, **_kwargs):
        n = len(texts)
        result = self._vectors[self._call_count: self._call_count + n]
        self._call_count += n
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            result = result / norms
        return result


def _unit_vec(d=8, **overrides) -> np.ndarray:
    """返回一个 d 维单位向量，overrides 指定坐标轴方向。"""
    v = np.zeros(d, dtype=np.float32)
    for k, val in overrides.items():
        v[int(k)] = val
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _make_chunker(vectors: np.ndarray, **kwargs) -> SemanticChunker:
    """辅助：用给定向量创建 SemanticChunker，可覆盖部分配置。"""
    import config

    chunker = SemanticChunker.__new__(SemanticChunker)
    chunker.embedder = MockEmbedder(vectors)
    chunker.percentile = kwargs.get("percentile", config.SEMANTIC_BREAKPOINT_PERCENTILE)
    chunker.window = kwargs.get("window", 1)  # 测试中用窗口=1 关闭平滑
    chunker.min_size = kwargs.get("min_size", 5)   # 测试中放宽下限
    chunker.max_size = kwargs.get("max_size", config.SEMANTIC_MAX_CHUNK_SIZE)
    return chunker


# ──────────────────────────────────────────────────────────────────────────────
# 1. 句子拆分
# ──────────────────────────────────────────────────────────────────────────────

class TestSentenceSplit:
    def _chunker(self):
        vecs = np.eye(4, dtype=np.float32)
        return _make_chunker(vecs)

    def test_split_on_chinese_punctuation(self):
        c = self._chunker()
        sents = c._sentence_split("这是第一句话。这是第二句话！这是第三句话？")
        assert len(sents) == 3

    def test_split_on_blank_line(self):
        c = self._chunker()
        sents = c._sentence_split("第一段内容在这里。\n\n第二段内容在这里。")
        assert len(sents) >= 2

    def test_filter_short_noise(self):
        c = self._chunker()
        # 孤立的单字符不应出现在结果中
        sents = c._sentence_split("正常句子。！\n\n下一段的正常内容。")
        assert all(len(s["text"]) >= 5 for s in sents)

    def test_empty_text(self):
        c = self._chunker()
        sents = c._sentence_split("")
        assert sents == []

    def test_markdown_heading_split(self):
        c = self._chunker()
        text = "这是导言部分，说明了一些内容。\n## 新章节\n这是新章节的内容。"
        sents = c._sentence_split(text)
        # 标题行之后应该形成新句子
        assert len(sents) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# 2. 余弦相似度计算
# ──────────────────────────────────────────────────────────────────────────────

class TestCosineSims:
    def test_identical_vectors_sim_is_one(self):
        v = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        sims = SemanticChunker._cosine_sims(v)
        assert len(sims) == 1
        assert abs(sims[0] - 1.0) < 1e-5

    def test_orthogonal_vectors_sim_is_zero(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sims = SemanticChunker._cosine_sims(v)
        assert abs(sims[0]) < 1e-5

    def test_opposite_vectors_sim_is_minus_one(self):
        v = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        sims = SemanticChunker._cosine_sims(v)
        assert abs(sims[0] - (-1.0)) < 1e-5

    def test_length_is_n_minus_one(self):
        v = np.random.randn(7, 16).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        sims = SemanticChunker._cosine_sims(v)
        assert len(sims) == 6


# ──────────────────────────────────────────────────────────────────────────────
# 3. 断点检测
# ──────────────────────────────────────────────────────────────────────────────

class TestFindBreakpoints:
    def _chunker(self, percentile=50.0):
        vecs = np.eye(4, dtype=np.float32)
        return _make_chunker(vecs, percentile=percentile)

    def test_breakpoint_at_low_similarity(self):
        c = self._chunker(percentile=50)
        sims = np.array([0.9, 0.85, 0.1, 0.88], dtype=np.float32)
        bps = c._find_breakpoints(sims)
        # 0.1 明显低于中位数，应被选为断点
        assert 2 in bps

    def test_no_breakpoint_when_all_similar(self):
        c = self._chunker(percentile=10)  # 只有最低 10% 触发
        sims = np.array([0.95, 0.93, 0.94, 0.92], dtype=np.float32)
        bps = c._find_breakpoints(sims)
        # 最多 1 个断点（最低那个）
        assert len(bps) <= 1

    def test_returns_list(self):
        c = self._chunker()
        bps = c._find_breakpoints(np.array([0.5, 0.8, 0.2], dtype=np.float32))
        assert isinstance(bps, list)


# ──────────────────────────────────────────────────────────────────────────────
# 4. split_text 端到端
# ──────────────────────────────────────────────────────────────────────────────

class TestSplitText:
    """端到端测试，用 Mock Embedder 模拟"话题转移"场景。"""

    def _make_topic_switch_chunker(self):
        """
        构造一个 3 句场景：
          句0 ↔ 句1：高相似（同一话题） cosim≈1
          句1 ↔ 句2：低相似（话题转移） cosim≈0
        """
        # 句0 和句1 方向相同（(1,0,0,0)）
        # 句2 方向完全不同（(0,1,0,0)）
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        return _make_chunker(vecs, percentile=50, window=1, min_size=5)

    def test_topic_switch_creates_two_chunks(self):
        chunker = self._make_topic_switch_chunker()
        # 三句话：前两句同话题，第三句话题转移
        text = "第一句话描述话题A的内容。第二句话继续话题A的讨论！第三句话突然转向了完全不同的话题。"
        chunks = chunker.split_text(text)
        # 应在话题转移处切开，产生 2 个 chunk
        assert len(chunks) == 2

    def test_short_text_returns_single_chunk(self):
        vecs = np.eye(4, dtype=np.float32)
        chunker = _make_chunker(vecs)
        text = "这是一段很短的文字。"  # 只有 1 句，< 3 句
        chunks = chunker.split_text(text)
        assert len(chunks) == 1
        assert "短的文字" in chunks[0]["text"]

    def test_empty_text_returns_empty(self):
        vecs = np.eye(4, dtype=np.float32)
        chunker = _make_chunker(vecs)
        chunks = chunker.split_text("")
        assert chunks == []

    def test_output_is_non_empty_strings(self):
        chunker = self._make_topic_switch_chunker()
        text = "正常句子在这里，包含了一些实质性的内容。另一个句子继续讨论同样的主题！现在换一个完全不同的话题来讨论。"
        chunks = chunker.split_text(text)
        assert all(isinstance(c, dict) and c["text"].strip() for c in chunks)

    def test_no_sentence_truncation(self):
        """验证：每个 chunk 内不应出现跨句截断（内容完整性）"""
        chunker = self._make_topic_switch_chunker()
        text = "话题A的第一句完整内容。话题A的第二句完整内容！话题B的完整句子出现在这里。"
        chunks = chunker.split_text(text)
        # 合并后的全文应包含原始所有内容
        joined = " ".join([c["text"] for c in chunks])
        for fragment in ["第一句", "第二句", "话题B"]:
            assert fragment in joined

    def test_code_block_shielding(self):
        """验证代码块屏蔽逻辑：代码块不会被拆分，保持原子性。"""
        chunker = self._make_topic_switch_chunker()
        text = "这是自然语言。\n```python\nprint('hello')\nprint('world')\n```\n结束语。"
        sents = chunker._sentence_split(text)
        # 应该有一个 is_code=True 的 sentence
        code_sents = [s for s in sents if s["is_code"]]
        assert len(code_sents) == 1
        assert "python" in code_sents[0]["code_language"]
        assert "print('hello')" in code_sents[0]["text"]


# ──────────────────────────────────────────────────────────────────────────────
# 5. 边界保护：min/max chunk size
# ──────────────────────────────────────────────────────────────────────────────

class TestChunkSizeBounds:
    def test_merge_small_chunks(self):
        """过小的碎片 chunk 应向前合并。"""
        vecs = np.eye(4, dtype=np.float32)
        # min_size=100 强制所有短句合并
        chunker = _make_chunker(vecs, min_size=100, max_size=9999)
        small_sents = [[{"text": "短句A", "is_code": False}], [{"text": "短句B", "is_code": False}], [{"text": "短句C", "is_code": False}]]
        merged = chunker._merge_small(small_sents)
        # 三个短片段应合并为 1 组
        assert len(merged) == 1

    def test_split_oversized(self):
        """超过 max_size 的 chunk 应被强制切分。"""
        vecs = np.eye(4, dtype=np.float32)
        chunker = _make_chunker(vecs, max_size=20)
        # 每段 10 字，合起来 40 字 > max_size=20
        oversized = [
            {"text": "AAAAAAAAAAAAAAAAAAAAA", "is_code": False, "code_language": None},  # 21 字
            {"text": "BBBBBBBBBBBBBBBBBBBBB", "is_code": False, "code_language": None},  # 21 字
        ]
        result = chunker._split_oversized(oversized)
        assert len(result) >= 2
        assert all(len(r["text"]) <= 20 for r in result)

    def test_single_overlong_sentence_hard_cut(self):
        """单个超长句无法二分时，应按字符截断。"""
        vecs = np.eye(4, dtype=np.float32)
        chunker = _make_chunker(vecs, max_size=10)
        long_sent = [{"text": "A" * 35, "is_code": False, "code_language": None}]
        result = chunker._split_oversized(long_sent)
        assert all(len(r["text"]) <= 10 for r in result)
        # 内容不丢失
        assert "".join([r["text"] for r in result]) == "A" * 35


# ──────────────────────────────────────────────────────────────────────────────
# 6. Embedder 异常降级
# ──────────────────────────────────────────────────────────────────────────────

class TestEmbedderFallback:
    def test_embedder_error_returns_raw_text(self):
        """Embedder 抛出异常时，split_text 应返回原文而非崩溃。"""

        class BrokenEmbedder:
            def encode(self, texts, **_):
                raise RuntimeError("模拟 API 超时")

        import config

        chunker = SemanticChunker.__new__(SemanticChunker)
        chunker.embedder = BrokenEmbedder()
        chunker.percentile = config.SEMANTIC_BREAKPOINT_PERCENTILE
        chunker.window = 1
        chunker.min_size = 5
        chunker.max_size = 9999

        text = "第一句话内容详细。第二句话继续。第三句话结束。"
        result = chunker.split_text(text)
        # 不应抛出异常；返回内容包含原文信息
        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], dict)
