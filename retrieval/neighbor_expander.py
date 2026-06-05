"""
邻居块扩展：为种子文档检索前后相邻的 chunk，提供更完整的上下文。

从 retriever.py 中解耦，职责单一。
"""
from __future__ import annotations

import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def chunk_id(source: str, chunk_index: int) -> str:
    """根据 source 路径和 chunk_index 构建 chunk ID（与 data_pipeline 保持一致）。"""
    parts = list(Path(source).with_suffix("").parts)
    return "_".join(parts) + f"_{chunk_index}"


class NeighborExpander:
    """从 ChromaDB collection 中查找种子 chunk 的相邻块并返回。"""

    def __init__(self, collection) -> None:
        self._collection = collection

    def expand(
        self,
        seeds: list[dict],
        window: int | None = None,
        budget: int | None = None,
    ) -> list[dict]:
        """
        为每个 seed 块查找相邻 chunk。

        Args:
            seeds: 种子文档列表（需含 metadata.source 和 metadata.chunk_index）
            window: 每侧扩展的 chunk 数量
            budget: 最多返回的相邻 chunk 总数

        Returns:
            新增的相邻 chunk 列表（不含原始 seeds）
        """
        if not seeds:
            return []

        window = config.RETRIEVER_NEIGHBOR_WINDOW if window is None else window
        budget = config.RETRIEVER_NEIGHBOR_BUDGET if budget is None else budget
        if window <= 0 or budget <= 0:
            return []

        neighbor_ids: list[str] = []
        seen_ids = {item["id"] for item in seeds}

        for seed in seeds:
            metadata = seed.get("metadata") or {}
            source = metadata.get("source")
            chunk_index = metadata.get("chunk_index")
            if source is None or chunk_index is None:
                continue

            for offset in range(1, window + 1):
                for neighbor_index in (chunk_index - offset, chunk_index + offset):
                    if neighbor_index < 0:
                        continue
                    neighbor_id = chunk_id(source, neighbor_index)
                    if neighbor_id in seen_ids:
                        continue
                    seen_ids.add(neighbor_id)
                    neighbor_ids.append(neighbor_id)
                    if len(neighbor_ids) >= budget:
                        break
                if len(neighbor_ids) >= budget:
                    break
            if len(neighbor_ids) >= budget:
                break

        if not neighbor_ids:
            return []

        results = self._collection.get(
            ids=neighbor_ids, include=["documents", "metadatas"]
        )
        expanded: list[dict] = []
        for rank, (doc_id, doc, meta) in enumerate(
            zip(results["ids"], results["documents"], results["metadatas"])
        ):
            if doc is None:
                continue
            meta_dict = dict(meta or {})
            meta_dict["score_type"] = "neighbor_boost_score"
            expanded.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta_dict,
                    "score": config.RETRIEVER_WEIGHT_NEIGHBOR / (rank + 1),
                    "channels": ["neighbor"],
                }
            )
        return expanded
