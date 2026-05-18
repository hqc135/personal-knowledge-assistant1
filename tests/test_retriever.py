"""
单元测试：retriever 核心逻辑 (RRF 融合)
"""
import pytest


class TestRRFFusion:
    """测试 Reciprocal Rank Fusion 算法"""

    @staticmethod
    def _rrf_fusion(*result_lists, k: int = 60) -> list[dict]:
        """复制自 Retriever._rrf_fusion（避免依赖完整 init）"""
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for results in result_lists:
            for rank, item in enumerate(results):
                doc_id = item["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
                doc_map[doc_id] = item

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [{**doc_map[did], "score": rrf_scores[did]} for did in sorted_ids]

    def test_single_source(self):
        """单路结果直接排序"""
        results = [
            {"id": "a", "text": "doc a", "metadata": {}, "score": 0.9},
            {"id": "b", "text": "doc b", "metadata": {}, "score": 0.8},
        ]
        fused = self._rrf_fusion(results)
        assert len(fused) == 2
        assert fused[0]["id"] == "a"  # rank 0 → 更高 RRF 分

    def test_overlapping_documents_boosted(self):
        """同时出现在两路的文档应获得更高 RRF 分"""
        vector = [
            {"id": "a", "text": "a", "metadata": {}, "score": 0.9},
            {"id": "b", "text": "b", "metadata": {}, "score": 0.7},
        ]
        bm25 = [
            {"id": "b", "text": "b", "metadata": {}, "score": 3.0},
            {"id": "c", "text": "c", "metadata": {}, "score": 2.0},
        ]
        fused = self._rrf_fusion(vector, bm25)
        # "b" 出现在两路，应排第一
        assert fused[0]["id"] == "b"
        assert len(fused) == 3

    def test_empty_inputs(self):
        """空输入返回空列表"""
        assert self._rrf_fusion([], []) == []
        assert self._rrf_fusion([]) == []

    def test_rrf_scores_decrease_with_rank(self):
        """RRF 分数应随 rank 递减"""
        results = [
            {"id": f"doc_{i}", "text": "", "metadata": {}, "score": 0}
            for i in range(5)
        ]
        fused = self._rrf_fusion(results)
        scores = [d["score"] for d in fused]
        assert scores == sorted(scores, reverse=True)

    def test_k_parameter_affects_scores(self):
        """不同 k 值应产生不同分数"""
        results = [{"id": "a", "text": "", "metadata": {}, "score": 0}]
        fused_k10 = self._rrf_fusion(results, k=10)
        fused_k60 = self._rrf_fusion(results, k=60)
        # k=10 → 1/(10+1) = 0.0909; k=60 → 1/(60+1) = 0.0164
        assert fused_k10[0]["score"] > fused_k60[0]["score"]
