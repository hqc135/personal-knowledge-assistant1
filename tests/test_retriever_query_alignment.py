from __future__ import annotations

from types import SimpleNamespace

import retriever as retriever_module
from retriever import Retriever


class _FakeIntentRouter:
    def __init__(self, events: list[str]):
        self.calls = []
        self.events = events

    def route(self, query: str):
        self.calls.append(query)
        self.events.append("route")
        return "local", {"reason": "routed"}


class _FakeAligner:
    def __init__(self, events: list[str]):
        self.calls = []
        self.events = events

    def align(self, query: str):
        self.calls.append(query)
        self.events.append("align")
        return SimpleNamespace(
            search_query=f"{query} vector database",
            to_dict=lambda: {"original_query": query, "search_query": f"{query} vector database"},
        )


def test_retriever_calls_intent_router_before_query_aligner(monkeypatch):
    events: list[str] = []
    retriever = Retriever.__new__(Retriever)
    retriever.use_intent_router = True
    retriever.use_query_aligner = True
    retriever.use_query_rewriter = False
    retriever.use_hybrid = False
    retriever.use_kg = False
    retriever.use_reranker = False
    retriever.last_timing = {}
    retriever.intent_router = _FakeIntentRouter(events)
    retriever.query_aligner = _FakeAligner(events)

    monkeypatch.setattr(Retriever, "_vector_search", lambda self, query, top_k: [
        {"id": "doc-1", "text": query, "metadata": {}, "score": 1.0}
    ])
    monkeypatch.setattr(Retriever, "_bm25_search", lambda self, query, top_k: [])
    monkeypatch.setattr(Retriever, "_merge_candidates", lambda *args, **kwargs: args[0][1])
    monkeypatch.setattr(Retriever, "_sort_local_contexts", lambda self, contexts: contexts)
    monkeypatch.setattr(retriever_module.config, "RETRIEVER_RECALL_TOP_K", 1)
    monkeypatch.setattr(retriever_module.config, "RETRIEVER_FINAL_K", 1)

    retriever.retrieve("向量数据库怎么选", mode="auto", use_rerank=False, top_k=1, final_k=1)

    assert retriever.intent_router.calls == ["向量数据库怎么选"]
    assert retriever.query_aligner.calls == ["向量数据库怎么选"]
    assert events == ["route", "align"]
    assert retriever.last_timing["aligned_query"] == "向量数据库怎么选 vector database"
    assert retriever.last_timing["route"] == "local"