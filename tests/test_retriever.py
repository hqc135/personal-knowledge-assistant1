"""Unit tests for retriever fusion and neighbor expansion."""

from retriever import Retriever


class TestRRFFusion:
    """测试加权 Reciprocal Rank Fusion 算法"""

    @staticmethod
    def _weighted_rrf(*scored_lists, k: int = 60, weights: dict[str, float] | None = None) -> list[dict]:
        return Retriever._merge_candidates(*scored_lists, k=k, weights=weights)

    def test_single_source(self):
        """单路结果直接排序"""
        results = [
            {"id": "a", "text": "doc a", "metadata": {}, "score": 0.9},
            {"id": "b", "text": "doc b", "metadata": {}, "score": 0.8},
        ]
        fused = self._weighted_rrf(("vector", results), weights={"vector": 1.0})
        assert len(fused) == 2
        assert fused[0]["id"] == "a"

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
        fused = self._weighted_rrf(
            ("vector", vector),
            ("bm25", bm25),
            weights={"vector": 1.0, "bm25": 1.0},
        )
        assert fused[0]["id"] == "b"
        assert len(fused) == 3
        assert set(fused[0]["channels"]) == {"bm25", "vector"}

    def test_empty_inputs(self):
        """空输入返回空列表"""
        assert self._weighted_rrf(("vector", []), ("bm25", [])) == []
        assert self._weighted_rrf(("vector", [])) == []

    def test_rrf_scores_decrease_with_rank(self):
        """RRF 分数应随 rank 递减"""
        results = [
            {"id": f"doc_{i}", "text": "", "metadata": {}, "score": 0}
            for i in range(5)
        ]
        fused = self._weighted_rrf(("vector", results), weights={"vector": 1.0})
        scores = [d["score"] for d in fused]
        assert scores == sorted(scores, reverse=True)

    def test_k_parameter_affects_scores(self):
        """不同 k 值应产生不同分数"""
        results = [{"id": "a", "text": "", "metadata": {}, "score": 0}]
        fused_k10 = self._weighted_rrf(("vector", results), k=10, weights={"vector": 1.0})
        fused_k60 = self._weighted_rrf(("vector", results), k=60, weights={"vector": 1.0})
        assert fused_k10[0]["score"] > fused_k60[0]["score"]

    def test_channel_weight_changes_ranking(self):
        """通道权重应影响最终排序"""
        vector = [
            {"id": "a", "text": "a", "metadata": {}, "score": 0.9},
            {"id": "b", "text": "b", "metadata": {}, "score": 0.8},
        ]
        kg = [
            {"id": "b", "text": "b", "metadata": {}, "score": 0.1},
        ]

        fused = self._weighted_rrf(
            ("vector", vector),
            ("kg", kg),
            weights={"vector": 1.0, "kg": 3.0},
        )

        assert fused[0]["id"] == "b"
        assert "kg" in fused[0]["channels"]


class TestNeighborExpansion:
    def test_expand_neighbor_chunks_uses_chunk_index_and_source(self):
        retriever = Retriever.__new__(Retriever)
        retriever.collection = _FakeCollection({
            "notes_demo_0": {"text": "seed", "metadata": {"source": "notes/demo.md", "chunk_index": 0}},
            "notes_demo_1": {"text": "next", "metadata": {"source": "notes/demo.md", "chunk_index": 1}},
        })

        seeds = [{"id": "notes_demo_0", "text": "seed", "metadata": {"source": "notes/demo.md", "chunk_index": 0}, "score": 1.0}]
        expanded = Retriever._expand_neighbor_chunks(retriever, seeds, window=1, budget=1)

        assert expanded[0]["id"] == "notes_demo_1"
        assert expanded[0]["channels"] == ["neighbor"]

    def test_expand_neighbor_chunks_respects_budget(self):
        retriever = Retriever.__new__(Retriever)
        retriever.collection = _FakeCollection({
            "notes_demo_1": {"text": "next", "metadata": {"source": "notes/demo.md", "chunk_index": 1}},
            "notes_demo_2": {"text": "next next", "metadata": {"source": "notes/demo.md", "chunk_index": 2}},
        })

        seeds = [{"id": "notes_demo_0", "text": "seed", "metadata": {"source": "notes/demo.md", "chunk_index": 0}, "score": 1.0}]
        expanded = Retriever._expand_neighbor_chunks(retriever, seeds, window=2, budget=1)

        assert len(expanded) == 1


class _FakeCollection:
    def __init__(self, docs: dict[str, dict]):
        self.docs = docs

    def get(self, ids, include=None):
        found_ids = []
        documents = []
        metadatas = []
        for doc_id in ids:
            doc = self.docs.get(doc_id)
            if not doc:
                continue
            found_ids.append(doc_id)
            documents.append(doc["text"])
            metadatas.append(doc["metadata"])
        return {"ids": found_ids, "documents": documents, "metadatas": metadatas}
