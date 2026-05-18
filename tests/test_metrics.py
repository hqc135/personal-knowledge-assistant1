"""
单元测试：metrics 模块
"""
import time

import pytest
from metrics import Timer, RequestMetrics, MetricsCollector


class TestTimer:
    def test_measures_elapsed_time(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms >= 40  # 至少 40ms (留余量)
        assert t.elapsed_ms < 200  # 不应超过 200ms

    def test_zero_duration(self):
        with Timer() as t:
            pass
        assert t.elapsed_ms >= 0
        assert t.elapsed_ms < 50


class TestMetricsCollector:
    def _make_metrics(self, query: str = "test", total_ms: float = 100) -> RequestMetrics:
        return RequestMetrics(
            timestamp=time.time(),
            query=query,
            embed_ms=10,
            search_ms=20,
            rerank_ms=30,
            generate_ms=40,
            total_ms=total_ms,
            num_sources=3,
            answer_length=100,
        )

    def test_record_and_history(self):
        c = MetricsCollector(max_history=10)
        c.record(self._make_metrics("q1"))
        c.record(self._make_metrics("q2"))
        assert len(c.history) == 2

    def test_max_history_limit(self):
        c = MetricsCollector(max_history=3)
        for i in range(5):
            c.record(self._make_metrics(f"q{i}"))
        assert len(c.history) == 3
        assert c.history[0].query == "q2"  # 最早的两条被淘汰

    def test_summary_empty(self):
        c = MetricsCollector(max_history=10)
        s = c.summary()
        assert s["total_requests"] == 0

    def test_summary_averages(self):
        c = MetricsCollector(max_history=10)
        c.record(self._make_metrics(total_ms=100))
        c.record(self._make_metrics(total_ms=200))
        s = c.summary()
        assert s["total_requests"] == 2
        assert s["avg_total_ms"] == 150.0

    def test_thread_safety(self):
        """并发写入不应出错"""
        import threading

        c = MetricsCollector(max_history=100)

        def worker():
            for _ in range(20):
                c.record(self._make_metrics())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(c.history) == 100  # 5 * 20 = 100
