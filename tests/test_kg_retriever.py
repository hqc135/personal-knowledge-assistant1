"""Unit tests for KG retrieval recall helpers."""

from __future__ import annotations

import kg_retriever


class _FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        def vector_for(text: str):
            lowered = text.lower()
            if "agent framework" in lowered or "agent" in lowered:
                return [1.0, 0.0, 0.0]
            if "react" in lowered:
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

        vectors = [vector_for(text) for text in texts]
        if len(vectors) == 1:
            return vectors[0]
        return vectors


def test_entity_vector_index_returns_semantic_match(monkeypatch):
    retriever = kg_retriever.KGRetriever.__new__(kg_retriever.KGRetriever)
    retriever._graph = kg_retriever.build_graph([
        {"head": "Agent Framework", "relation": "is", "tail": "framework", "source": "s", "chunk_id": "c1"},
        {"head": "ReAct", "relation": "is", "tail": "prompting strategy", "source": "s", "chunk_id": "c2"},
    ])
    retriever._entities = set(retriever._graph.nodes)
    retriever._entity_list = sorted(retriever._entities)
    retriever._embedder = _FakeEmbedder()
    retriever._entity_vector_index = kg_retriever._EntityVectorIndex.build(retriever._entity_list, retriever._embedder)

    matches = retriever._match_entities("agent")

    assert "Agent Framework" in matches


def test_entity_vector_index_handles_refresh_cache(monkeypatch):
    retriever = kg_retriever.KGRetriever.__new__(kg_retriever.KGRetriever)
    retriever._graph = kg_retriever.build_graph([])
    retriever._entities = set()
    retriever._entity_list = []
    retriever._entity_vector_index = None
    retriever._embedder = _FakeEmbedder()

    assert retriever._get_entity_vector_index() is None