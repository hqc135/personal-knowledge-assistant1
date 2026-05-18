"""
检索模块：支持向量检索、BM25 关键词检索、混合检索 (RRF 融合) + Rerank
"""
import logging

import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import chromadb

from embedder import ZhipuEmbedder
from metrics import Timer
import config
from typing import Optional

logger = logging.getLogger(__name__)

# 降低 jieba 日志级别
jieba.setLogLevel(logging.WARNING)


class Retriever:
    def __init__(self):
        self.embedder = ZhipuEmbedder()
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_collection(config.COLLECTION_NAME)

        # Reranker
        self.use_reranker = config.USE_RERANKER
        if self.use_reranker:
            logger.info("加载 Reranker: %s", config.RERANKER_MODEL)
            self.reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)

        # BM25 索引 (混合检索)
        self.use_hybrid = config.USE_HYBRID_SEARCH
        if self.use_hybrid:
            self._build_bm25_index()

        # 上次检索的分步耗时 (供可观测性使用)
        self.last_timing = {}

    def _build_bm25_index(self):
        """从 ChromaDB 加载全部文档，构建 BM25 索引"""
        logger.info("构建 BM25 索引...")
        all_data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = all_data["ids"]
        self._bm25_docs = all_data["documents"]
        self._bm25_metas = all_data["metadatas"]

        if not self._bm25_docs:
            logger.warning("集合为空，BM25 索引跳过")
            self.bm25 = None
            return

        tokenized = [list(jieba.cut(doc)) for doc in self._bm25_docs]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 索引完成: %d 篇文档", len(self._bm25_docs))

    # ── 检索策略 ──────────────────────────────────────────

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """纯向量检索"""
        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {"id": id_, "text": doc, "metadata": meta, "score": 1 - dist}
            for id_, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 关键词检索"""
        if not self.bm25:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]

        return [
            {
                "id": self._bm25_ids[i],
                "text": self._bm25_docs[i],
                "metadata": self._bm25_metas[i],
                "score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _rrf_fusion(*result_lists, k: int = 60) -> list[dict]:
        """
        Reciprocal Rank Fusion：融合多路检索结果。
        RRF(d) = Σ 1 / (k + rank_i(d))
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for results in result_lists:
            for rank, item in enumerate(results):
                doc_id = item["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
                doc_map[doc_id] = item

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [{**doc_map[did], "score": rrf_scores[did]} for did in sorted_ids]

    # ── 主检索入口 ────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        mode: str = "auto",
        use_rerank: Optional[bool] = None,
        top_k: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> list[dict]:
        """
        检索入口。
        mode: "auto"(按配置), "vector", "bm25", "hybrid"
        use_rerank: 覆盖配置的 reranker 开关 (用于消融实验)
        """
        top_k = top_k or config.RETRIEVER_TOP_K
        final_k = final_k or config.RETRIEVER_FINAL_K
        if use_rerank is None:
            use_rerank = self.use_reranker
        if mode == "auto":
            mode = "hybrid" if self.use_hybrid else "vector"

        self.last_timing = {}

        # ── Embedding + 检索 ──
        with Timer() as t_embed:
            if mode == "bm25":
                candidates = self._bm25_search(query, top_k)
            elif mode == "hybrid":
                vector_results = self._vector_search(query, top_k)
                bm25_results = self._bm25_search(query, top_k)
                candidates = self._rrf_fusion(
                    vector_results, bm25_results, k=config.RRF_K
                )
            else:  # vector
                candidates = self._vector_search(query, top_k)

        self.last_timing["embed_ms"] = t_embed.elapsed_ms

        logger.debug("检索模式=%s, 候选数=%d", mode, len(candidates))

        # ── Rerank ──
        rerank_ms = 0.0
        if use_rerank and self.use_reranker and candidates:
            with Timer() as t_rerank:
                pairs = [[query, c["text"]] for c in candidates]
                scores = self.reranker.predict(pairs)
                for c, s in zip(candidates, scores):
                    c["score"] = float(s)
                candidates.sort(key=lambda x: x["score"], reverse=True)
            rerank_ms = t_rerank.elapsed_ms

        self.last_timing["rerank_ms"] = rerank_ms
        self.last_timing["mode"] = mode

        # 去掉内部 id 字段再返回
        return [
            {"text": c["text"], "metadata": c["metadata"], "score": c["score"]}
            for c in candidates
        ][:final_k]
