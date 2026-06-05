"""Unit tests for evaluator helpers."""

from evaluator import _keyword_coverage


def test_keyword_coverage_counts_matches_in_query_context_or_answer():
    coverage = _keyword_coverage(
        query="总结 RAG",
        contexts=["这里提到检索和生成"],
        answer="回答涵盖 RAG",
        expected_keywords=["RAG", "检索", "生成"],
    )

    assert coverage == 1.0


def test_keyword_coverage_returns_zero_for_empty_keywords():
    assert _keyword_coverage("q", [], "a", []) == 0.0