"""
跨层数据契约：定义 RAG 管线中各层之间传递的结构化数据对象。

设计原则：
- 消除 retriever.last_timing 隐式副作用传递
- 上层（app.py / pipeline.py）不再需要穿透访问检索器内部状态
- 所有字段均有默认值，便于测试中按需覆盖
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PipelineTrace:
    """检索管线执行轨迹，记录各阶段的决策和耗时。"""
    # ── 路由决策 ──
    route_decision: str = "local"   # "local" | "global"
    mode: str = "vector"            # "vector" | "bm25" | "hybrid" | "global"

    # ── 查询重写 ──
    rewrite_triggered: bool = False
    original_query: str = ""
    rewritten_query: str = ""
    rewrite_ms: float = 0.0

    # ── 查询对齐 ──
    align_triggered: bool = False
    aligned_query: str = ""
    query_alignment: dict = field(default_factory=dict)
    align_ms: float = 0.0

    # ── Rerank ──
    rerank_active: bool = False
    rerank_ms: float = 0.0

    # ── 其他耗时 ──
    embed_ms: float = 0.0
    pure_search_ms: float = 0.0

    # ── 路由元信息（可选，用于调试） ──
    route_info: dict = field(default_factory=dict)
    kg_ms: float = 0.0

    def to_dict(self) -> dict:
        """向后兼容：转换为旧版 last_timing dict 格式，供遗留代码使用。"""
        return {
            "route_decision": self.route_decision,
            "mode": self.mode,
            "route": self.route_decision,
            "rewrite_triggered": self.rewrite_triggered,
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "rewrite_ms": self.rewrite_ms,
            "align_triggered": self.align_triggered,
            "aligned_query": self.aligned_query,
            "query_alignment": self.query_alignment,
            "align_ms": self.align_ms,
            "rerank_active": self.rerank_active,
            "rerank_ms": self.rerank_ms,
            "embed_ms": self.embed_ms,
            "pure_search_ms": self.pure_search_ms,
            "route_info": self.route_info,
            "kg_ms": self.kg_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineTrace":
        """从旧版 last_timing dict 构造 PipelineTrace（向后兼容）。"""
        return cls(
            route_decision=str(
                d.get("route_decision") or d.get("route") or d.get("mode") or "local"
            ),
            mode=str(d.get("mode") or "vector"),
            rewrite_triggered=bool(d.get("rewrite_triggered", False)),
            original_query=str(d.get("original_query") or ""),
            rewritten_query=str(d.get("rewritten_query") or ""),
            rewrite_ms=float(d.get("rewrite_ms") or 0.0),
            align_triggered=bool(d.get("align_triggered", False)),
            aligned_query=str(d.get("aligned_query") or ""),
            query_alignment=dict(d.get("query_alignment") or {}),
            align_ms=float(d.get("align_ms") or 0.0),
            rerank_active=bool(d.get("rerank_active", False)),
            rerank_ms=float(d.get("rerank_ms") or 0.0),
            embed_ms=float(d.get("embed_ms") or 0.0),
            pure_search_ms=float(d.get("pure_search_ms") or 0.0),
            route_info=dict(d.get("route_info") or {}),
            kg_ms=float(d.get("kg_ms") or 0.0),
        )


@dataclass
class RetrievalResult:
    """
    检索结果，封装上下文列表和管线执行轨迹。

    - contexts: 与旧版 retriever.retrieve() 返回值格式完全兼容
    - trace: 替代 retriever.last_timing，无需副作用穿透
    """
    contexts: list[dict]
    trace: PipelineTrace = field(default_factory=PipelineTrace)
