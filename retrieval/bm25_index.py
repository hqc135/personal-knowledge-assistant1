"""
BM25 索引封装：独立管理索引构建与关键词检索。

从 retriever.py 中解耦，职责单一：
- 从 ChromaDB collection 加载文档并构建 jieba + BM25 索引
- 提供线程安全的 search() 接口
"""
from __future__ import annotations

import logging
import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# 降低 jieba 日志噪音
jieba.setLogLevel(logging.WARNING)


class BM25Index:
    """BM25 关键词检索索引。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []
        self._bm25: BM25Okapi | None = None

    # ── 构建 ─────────────────────────────────────────────

    def build(self, collection) -> None:
        """从 ChromaDB collection 加载全部文档并构建 BM25 索引。"""
        logger.info("构建 BM25 索引...")
        all_data = collection.get(include=["documents", "metadatas"])
        self._ids = all_data["ids"]
        self._docs = all_data["documents"]
        self._metas = all_data["metadatas"]

        if not self._docs:
            logger.warning("集合为空，BM25 索引跳过")
            self._bm25 = None
            return

        tokenized = [list(jieba.cut(doc)) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 索引完成: %d 篇文档", len(self._docs))

    # ── 检索 ─────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[dict]:
        """BM25 关键词检索，返回与 vector_search 格式相同的 list[dict]。"""
        if self._bm25 is None:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]

        return [
            {
                "id": self._ids[i],
                "text": self._docs[i],
                "metadata": self._metas[i],
                "score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0
        ]

    @property
    def is_ready(self) -> bool:
        """索引是否已构建完成。"""
        return self._bm25 is not None
