from __future__ import annotations

from types import SimpleNamespace

from query_aligner import QueryAligner


class _FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def test_align_falls_back_deterministically_without_alignment_signal():
    client = _FakeClient("{}")
    aligner = QueryAligner(
        client=client,
        min_query_length=4,
        min_confidence=0.5,
        max_expansions=3,
        max_terms=4,
    )

    result = aligner.align("你好")

    assert result.should_expand is False
    assert result.search_query == "你好"
    assert result.fallback_reason == "too_short"
    assert client.calls == 0


def test_align_expands_bilingual_terms_when_gate_passes():
    client = _FakeClient(
        '{"should_expand": true, "confidence": 0.88, "search_query": "向量数据库 vector database", '
        '"expansions": ["vector database", "vector store"], "aligned_terms": [], "reason": "aligned"}'
    )
    aligner = QueryAligner(
        client=client,
        min_query_length=4,
        min_confidence=0.5,
        max_expansions=3,
        max_terms=4,
    )

    result = aligner.align("向量数据库怎么选")

    assert result.should_expand is True
    assert result.fallback_reason is None
    assert result.search_query == "向量数据库 vector database"
    assert result.expansions == ["vector database", "vector store"]
    assert client.calls == 1


def test_align_falls_back_on_low_confidence():
    client = _FakeClient(
        '{"should_expand": true, "confidence": 0.2, "search_query": "RAG retrieval augmented generation", '
        '"expansions": ["retrieval augmented generation"], "aligned_terms": [], "reason": "weak"}'
    )
    aligner = QueryAligner(
        client=client,
        min_query_length=4,
        min_confidence=0.5,
        max_expansions=3,
        max_terms=4,
    )

    result = aligner.align("RAG 是什么")

    assert result.should_expand is False
    assert result.fallback_reason == "low_confidence"
    assert result.search_query == "RAG 是什么"