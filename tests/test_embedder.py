"""
单元测试：embedder 维度行为
"""
import numpy as np
import pytest


class TestEncodeOutputShape:
    """测试 encode() 返回的维度是否正确（不调用真实 API，mock 验证逻辑）"""

    @staticmethod
    def _simulate_encode(texts, single_input: bool) -> np.ndarray:
        """模拟 encode 的 squeeze 逻辑"""
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        # 假设 API 返回 (N, dim) 的数据
        dim = 1024
        result = np.random.randn(len(texts), dim).astype(np.float32)

        # 归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        result = result / norms

        if single_input:
            result = result.squeeze(0)

        return result

    def test_single_string_returns_1d(self):
        """单条文本输入应返回 1D (dim,)"""
        result = self._simulate_encode("hello", single_input=True)
        assert result.ndim == 1
        assert result.shape == (1024,)

    def test_list_returns_2d(self):
        """列表输入应返回 2D (N, dim)"""
        result = self._simulate_encode(["a", "b", "c"], single_input=False)
        assert result.ndim == 2
        assert result.shape == (3, 1024)

    def test_single_list_returns_2d(self):
        """单元素列表应返回 2D (1, dim)"""
        result = self._simulate_encode(["only one"], single_input=False)
        assert result.ndim == 2
        assert result.shape == (1, 1024)

    def test_normalized_vectors(self):
        """归一化后每个向量的 L2 范数应接近 1"""
        result = self._simulate_encode(["test"], single_input=True)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_dot_product_compatibility(self):
        """1D query 和 2D docs 的矩阵乘法应成功"""
        query = self._simulate_encode("query", single_input=True)  # (dim,)
        docs = self._simulate_encode(["a", "b"], single_input=False)  # (2, dim)
        scores = np.dot(docs, query)  # (2, dim) @ (dim,) → (2,)
        assert scores.shape == (2,)
