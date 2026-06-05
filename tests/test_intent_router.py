import numpy as np

from intent_router import IntentRouter


class _FakeEmbedder:
    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, list):
            vectors = [self.mapping[t] for t in texts]
            return np.vstack(vectors)
        return self.mapping[texts]


def test_route_global_with_clear_margin():
    vectors = {
        "g1": np.array([1.0, 0.0], dtype=np.float32),
        "g2": np.array([0.9, 0.1], dtype=np.float32),
        "l1": np.array([0.0, 1.0], dtype=np.float32),
        "l2": np.array([0.1, 0.9], dtype=np.float32),
        "q": np.array([1.0, 0.0], dtype=np.float32),
    }
    router = IntentRouter(
        _FakeEmbedder(vectors),
        prototypes={"global": ["g1", "g2"], "local": ["l1", "l2"]},
        min_score=0.2,
        min_margin=0.2,
        agg="max",
        fallback="local",
    )

    route, info = router.route("q")

    assert route == "global"
    assert info["reason"] == "routed"


def test_route_fallback_when_margin_low():
    vectors = {
        "g1": np.array([1.0, 0.0], dtype=np.float32),
        "l1": np.array([0.0, 1.0], dtype=np.float32),
        "q": np.array([0.7, 0.7], dtype=np.float32),
    }
    router = IntentRouter(
        _FakeEmbedder(vectors),
        prototypes={"global": ["g1"], "local": ["l1"]},
        min_score=0.2,
        min_margin=0.2,
        agg="max",
        fallback="local",
    )

    route, info = router.route("q")

    assert route == "local"
    assert info["reason"] == "fallback"
