"""
可观测性模块：记录每次请求的各阶段耗时和统计信息
"""
import time
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Any

import numpy as np
import config


class Timer:
    """计时上下文管理器"""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


@dataclass
class RequestMetrics:
    """单次请求的完整指标"""

    timestamp: float
    query: str
    # 各阶段耗时 (ms)
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0
    # 结果信息
    num_sources: int = 0
    answer_length: int = 0
    retrieve_mode: str = "hybrid"


class MetricsCollector:
    """线程安全的指标收集器"""

    def __init__(self, max_history: Optional[int] = None):
        max_history = max_history or config.METRICS_MAX_HISTORY
        self._history: deque[RequestMetrics] = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def record(self, metrics: RequestMetrics):
        with self._lock:
            self._history.append(metrics)

    @property
    def history(self) -> list[RequestMetrics]:
        with self._lock:
            return list(self._history)

    def summary(self) -> dict:
        """计算汇总统计"""
        h = self.history
        if not h:
            return {
                "total_requests": 0,
                "avg_total_ms": 0,
                "avg_embed_ms": 0,
                "avg_search_ms": 0,
                "avg_rerank_ms": 0,
                "avg_generate_ms": 0,
                "p95_total_ms": 0,
            }

        totals = [m.total_ms for m in h]
        return {
            "total_requests": len(h),
            "avg_total_ms": float(np.mean(totals)),
            "avg_embed_ms": float(np.mean([m.embed_ms for m in h])),
            "avg_search_ms": float(np.mean([m.search_ms for m in h])),
            "avg_rerank_ms": float(np.mean([m.rerank_ms for m in h])),
            "avg_generate_ms": float(np.mean([m.generate_ms for m in h])),
            "p95_total_ms": float(np.percentile(totals, 95)),
        }


# 全局指标收集器实例
collector = MetricsCollector()
