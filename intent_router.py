"""
Intent routing based on prototype embeddings.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)

DEFAULT_PROTOTYPES: dict[str, list[str]] = {
    "global": [
        "Give me an overview of this topic",
        "Summarize the key ideas in these notes",
        "Compare two approaches and their trade-offs",
        "What are the trends in this field",
        "Provide a high-level review",
    ],
    "local": [
        "What is BM25",
        "Explain the RRF formula",
        "How do I set CHROMA_DB_PATH",
        "Where is KG top_k configured",
        "What does chunk_overlap do",
    ],
}


class IntentRouter:
    def __init__(
        self,
        embedder,
        prototypes: dict[str, list[str]] | None = None,
        min_score: float | None = None,
        min_margin: float | None = None,
        agg: str | None = None,
        fallback: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.min_score = config.INTENT_ROUTER_MIN_SCORE if min_score is None else min_score
        self.min_margin = config.INTENT_ROUTER_MIN_MARGIN if min_margin is None else min_margin
        self.agg = config.INTENT_ROUTER_AGG if agg is None else agg
        self.fallback = config.INTENT_ROUTER_FALLBACK if fallback is None else fallback
        self.prototypes = prototypes or self._load_prototypes()
        self.prototype_vectors = self._embed_prototypes(self.prototypes)

    def _load_prototypes(self) -> dict[str, list[str]]:
        path = Path(config.INTENT_ROUTER_PROTOTYPES_PATH)
        if not path.exists():
            logger.warning("Intent prototypes not found: %s", path)
            return dict(DEFAULT_PROTOTYPES)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read intent prototypes: %s", exc)
            return dict(DEFAULT_PROTOTYPES)

        if not isinstance(data, dict):
            logger.warning("Intent prototypes malformed, fallback to defaults")
            return dict(DEFAULT_PROTOTYPES)

        cleaned: dict[str, list[str]] = {}
        for key, value in data.items():
            if not isinstance(value, list):
                continue
            cleaned[key] = [str(v) for v in value if str(v).strip()]

        if not cleaned:
            return dict(DEFAULT_PROTOTYPES)
        return cleaned

    def _embed_prototypes(self, prototypes: dict[str, list[str]]) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for intent, items in prototypes.items():
            if not items:
                continue
            emb = self.embedder.encode(items, normalize_embeddings=True)
            vectors[intent] = np.asarray(emb, dtype=np.float32)
        return vectors

    def _aggregate(self, sims: np.ndarray) -> float:
        if sims.size == 0:
            return float("-inf")
        if self.agg == "top2":
            top = np.sort(sims)[-2:]
            return float(np.mean(top))
        return float(np.max(sims))

    def route(self, query: str) -> tuple[str, dict[str, Any]]:
        if not self.prototype_vectors:
            return self.fallback, {"reason": "no_prototypes"}

        query_vec = self.embedder.encode(query, normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype=np.float32)

        scores: dict[str, float] = {}
        for intent, vecs in self.prototype_vectors.items():
            sims = np.dot(vecs, query_vec)
            scores[intent] = self._aggregate(sims)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else float("-inf")
        margin = best_score - second_score

        if best_score < self.min_score or margin < self.min_margin:
            return self.fallback, {
                "reason": "fallback",
                "scores": scores,
                "best": best_intent,
                "best_score": best_score,
                "margin": margin,
            }

        return best_intent, {
            "reason": "routed",
            "scores": scores,
            "best": best_intent,
            "best_score": best_score,
            "margin": margin,
        }
