"""
Reranker 封装：CrossEncoder 推理 + 线程安全 LRU 缓存。

从 retriever.py 中解耦，职责单一：
- ThreadSafeLRUCache 拦截高频 (query, chunk_id) 重排分数
- Reranker.rerank() 批量推理并返回排序后的候选列表
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ThreadSafeLRUCache:
    """线程安全的 LRU 缓存，用于拦截高频 (Query, Chunk) 的重排分数。"""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value) -> None:
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


class Reranker:
    """CrossEncoder Reranker，内置 LRU 缓存加速重复推理。"""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        max_length: int = 512,
        cache_capacity: int = 10000,
    ) -> None:
        logger.info("加载 Reranker: %s", model_name)
        self._model = CrossEncoder(model_name, max_length=max_length)
        self._batch_size = batch_size
        self._cache = ThreadSafeLRUCache(capacity=cache_capacity)

    def warmup(self) -> None:
        """预热模型推理（触发 PyTorch/CUDA 初始化）。"""
        try:
            self._model.predict([["预热", "测试"]])
        except Exception as exc:
            logger.warning("Reranker 预热异常（不影响运行）: %s", exc)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
    ) -> tuple[list[dict], float]:
        """
        对候选列表进行重排，返回 (已排序候选列表, rerank_ms)。

        candidates 会被原地修改（写入 score 和 metadata），
        调用方应传入副本或已接受此行为。
        """
        import time

        if not candidates:
            return candidates, 0.0

        t0 = time.perf_counter()

        scores: list[Optional[float]] = [None] * len(candidates)
        uncached_pairs: list[list[str]] = []
        uncached_indices: list[int] = []

        # 1. 查询 LRU 缓存
        for i, c in enumerate(candidates):
            doc_id = c["id"]
            cached = self._cache.get((query, doc_id))
            if cached is not None:
                scores[i] = cached
            else:
                uncached_pairs.append([query, c["text"]])
                uncached_indices.append(i)

        # 2. 批量推理未命中缓存的候选
        if uncached_pairs:
            batch_scores = self._model.predict(
                uncached_pairs, batch_size=self._batch_size
            )
            if isinstance(batch_scores, float) or np.isscalar(batch_scores):
                batch_scores = [batch_scores]
            for idx, score in zip(uncached_indices, batch_scores):
                scores[idx] = float(score)
                self._cache.put((query, candidates[idx]["id"]), float(score))

        # 3. 写入候选的 score 和 metadata
        for c, s in zip(candidates, scores):
            c.setdefault("metadata", {})["raw_rerank_logit"] = float(s)
            c["metadata"]["score_type"] = "rerank_compressed_score"
            c["score"] = float(1 / (1 + np.exp(-s)))

        candidates.sort(key=lambda x: x["score"], reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return candidates, elapsed_ms
