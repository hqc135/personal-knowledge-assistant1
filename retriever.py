"""
检索模块：支持向量检索、BM25 关键词检索、混合检索 (RRF 融合) + Rerank
"""
import logging
from collections import defaultdict
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import chromadb
from openai import OpenAI

from embedder import ZhipuEmbedder
from kg_retriever import KGRetriever
from intent_router import IntentRouter
from query_aligner import QueryAligner, QueryAlignmentResult
from query_rewriter import QueryRewriter
from metrics import Timer
import config
from typing import Optional

import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

# 降低 jieba 日志级别
jieba.setLogLevel(logging.WARNING)

class ThreadSafeLRUCache:
    """线程安全的 LRU 缓存，用于拦截高频 (Query, Chunk) 的重排分数"""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


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
            self.rerank_cache = ThreadSafeLRUCache(capacity=10000)

        # BM25 索引 (混合检索)
        self.use_hybrid = config.USE_HYBRID_SEARCH
        if self.use_hybrid:
            self._build_bm25_index()

        # KG 检索
        self.use_kg = config.USE_KG_RETRIEVAL
        if self.use_kg:
            self.kg = KGRetriever()

        # Intent Router
        self.use_intent_router = config.USE_INTENT_ROUTER
        if self.use_intent_router:
            self.intent_router = IntentRouter(self.embedder)

        # Query Aligner
        self.use_query_aligner = config.USE_QUERY_ALIGNER
        if self.use_query_aligner:
            self.query_aligner = QueryAligner()
            
        # Query Rewriter
        self.use_query_rewriter = config.USE_QUERY_REWRITER
        if self.use_query_rewriter:
            self.query_rewriter = QueryRewriter()

        self.summary_client = None
        if config.INTENT_ROUTER_GLOBAL_MODE == "summary":
            self.summary_client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.LLM_BASE_URL,
            )

        # 上次检索的分步耗时 (供可观测性使用)
        self.last_timing = {}

        # ── 启动后台预热线程，解耦生命周期 ──
        import threading
        threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self):
        """后台预热，避免阻塞主线程启动，提前建立所有必要缓存和连接"""
        logger.info("后台预热线程已启动...")
        try:
            # 1. 预热 jieba 分词
            jieba.initialize()
            
            # 2. 预热 Reranker 模型推理 (触发 PyTorch/CUDA 初始化耗时)
            if getattr(self, "use_reranker", False) and hasattr(self, "reranker"):
                self.reranker.predict([["预热", "测试"]])
                
            # 3. 预热 ChromaDB 向量查询连接
            dummy_embedding = [0.0] * config.EMBEDDING_DIMENSIONS
            self.collection.query(query_embeddings=[dummy_embedding], n_results=1)
            
            # 4. 预热 KG 实体向量索引 (这是首问最严重的耗时点)
            if getattr(self, "use_kg", False) and hasattr(self, "kg"):
                self.kg._get_entity_vector_index()
                
            logger.info("后台预热全部完成！首次查询将极速响应。")
        except Exception as e:
            logger.warning("后台预热过程出现异常，但不影响正常运行: %s", e)

    def _align_query(self, query: str) -> tuple[str, QueryAlignmentResult | None]:
        if not self.use_query_aligner:
            return query, None

        with Timer() as t_align:
            alignment = self.query_aligner.align(query)
        self.last_timing["align_ms"] = t_align.elapsed_ms
        self.last_timing["align_triggered"] = bool(
            getattr(alignment, "should_expand", False) and getattr(alignment, "expansions", [])
        )
        self.last_timing["query_alignment"] = alignment.to_dict()
        return alignment.search_query, alignment

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

    def _kg_search(self, query: str, top_k: int) -> list[dict]:
        if not self.use_kg:
            return []

        chunk_ids = self.kg.retrieve_chunk_ids(query, top_k=top_k)
        if not chunk_ids:
            return []

        results = self.collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        output: list[dict] = []
        for rank, (doc_id, doc, meta) in enumerate(
            zip(results["ids"], results["documents"], results["metadatas"])
        ):
            if doc is None:
                continue
            score = config.KG_SCORE_BASE + (max(top_k - rank, 0) / max(top_k, 1)) * 0.1
            output.append({"id": doc_id, "text": doc, "metadata": meta, "score": score})
        return output

    @staticmethod
    def _chunk_id(source: str, chunk_index: int) -> str:
        parts = list(Path(source).with_suffix("").parts)
        return "_".join(parts) + f"_{chunk_index}"

    @staticmethod
    def _merge_metadata(*metadata_items: dict | None) -> dict:
        merged: dict = {}
        channel_metadata: dict[str, dict] = {}

        for index, metadata in enumerate(metadata_items):
            if not metadata:
                continue
            channel_name = metadata.get("channel") if isinstance(metadata, dict) else None
            if channel_name is None:
                channel_name = f"channel_{index}"
            channel_metadata[channel_name] = dict(metadata)
            for key, value in metadata.items():
                if key == "channel":
                    continue
                if key not in merged:
                    merged[key] = value

        if channel_metadata:
            merged["channel_metadata"] = channel_metadata
        return merged

    @staticmethod
    def _merge_candidates(
        *scored_lists: tuple[str, list[dict]],
        k: int = 60,
        weights: dict[str, float] | None = None,
    ) -> list[dict]:
        weights = weights or {}
        fused_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, dict] = {}
        channel_hits: dict[str, set[str]] = defaultdict(set)
        channel_scores: dict[str, dict[str, float]] = defaultdict(dict)
        channel_metadata: dict[str, dict[str, dict]] = defaultdict(dict)

        for channel, results in scored_lists:
            channel_weight = weights.get(channel, 1.0)
            for rank, item in enumerate(results):
                doc_id = item["id"]
                fused_scores[doc_id] += channel_weight / (k + rank + 1)
                channel_scores[doc_id][channel] = float(item.get("score", 0.0))
                channel_metadata[doc_id][channel] = dict(item.get("metadata") or {})
                if doc_id not in doc_map or item["score"] > doc_map[doc_id].get("score", float("-inf")):
                    doc_map[doc_id] = item
                channel_hits[doc_id].add(channel)

        merged: list[dict] = []
        for doc_id, score in fused_scores.items():
            item = dict(doc_map[doc_id])
            item["metadata"] = Retriever._merge_metadata(
                *[
                    {**metadata, "channel": channel}
                    for channel, metadata in channel_metadata[doc_id].items()
                ]
            )
            item["metadata"]["score_type"] = "rrf_position_score"
            item["score"] = score
            item["channels"] = sorted(channel_hits[doc_id])
            item["channel_scores"] = dict(channel_scores[doc_id])
            merged.append(item)
        return sorted(merged, key=lambda x: x["score"], reverse=True)

    def _expand_neighbor_chunks(
        self,
        seeds: list[dict],
        window: int | None = None,
        budget: int | None = None,
    ) -> list[dict]:
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
                    neighbor_id = self._chunk_id(source, neighbor_index)
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

        results = self.collection.get(ids=neighbor_ids, include=["documents", "metadatas"])
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

    @staticmethod
    def _sort_local_contexts(contexts: list[dict]) -> list[dict]:
        grouped: dict[str, list[tuple[int, int, dict]]] = {}
        source_order: list[str] = []
        passthrough: list[tuple[int, dict]] = []

        for original_index, item in enumerate(contexts):
            metadata = item.get("metadata") or {}
            source = metadata.get("source")
            chunk_index = metadata.get("chunk_index")
            if source is None or chunk_index is None:
                passthrough.append((original_index, item))
                continue

            if source not in grouped:
                grouped[source] = []
                source_order.append(source)

            grouped[source].append((int(chunk_index), original_index, item))

        sorted_contexts: list[dict] = []
        for source in source_order:
            sorted_contexts.extend(
                item
                for _, _, item in sorted(
                    grouped[source], key=lambda value: (value[0], value[1])
                )
            )

        sorted_contexts.extend(
            item for _, item in sorted(passthrough, key=lambda value: value[0])
        )
        return sorted_contexts

    def _global_retrieve(
        self,
        query: str,
        top_k: int,
        max_sources: int,
        max_chars: int,
    ) -> list[dict]:
        if self.use_hybrid:
            vector_results = self._vector_search(query, top_k)
            bm25_results = self._bm25_search(query, top_k)
            candidates = self._merge_candidates(
                ("vector", vector_results),
                ("bm25", bm25_results),
                k=config.RRF_K,
                weights={
                    "vector": config.RETRIEVER_WEIGHT_VECTOR,
                    "bm25": config.RETRIEVER_WEIGHT_BM25,
                },
            )
        else:
            candidates = self._vector_search(query, top_k)

        source_best: dict[str, dict] = {}
        for item in candidates:
            source = (item.get("metadata") or {}).get("source")
            if not source:
                continue
            if source not in source_best or item["score"] > source_best[source]["score"]:
                source_best[source] = item

        ranked_sources = sorted(
            source_best.values(), key=lambda x: x["score"], reverse=True
        )[:max_sources]

        contexts: list[dict] = []
        for item in ranked_sources:
            source = item["metadata"]["source"]
            full_path = Path(config.NOTES_DIR) / source
            text = item["text"]
            if full_path.exists():
                try:
                    text = full_path.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.warning("读取文档失败 %s: %s", full_path, exc)
            if max_chars > 0 and len(text) > max_chars:
                text = text[:max_chars].rstrip() + "\n... [truncated]"

            contexts.append(
                {
                    "id": f"doc::{source}",
                    "text": text,
                    "metadata": {
                        "source": source,
                        "scope": "global",
                        "score_type": "rrf_position_score" if self.use_hybrid else "vector_similarity",
                    },
                    "score": item["score"],
                }
            )

        if config.INTENT_ROUTER_GLOBAL_MODE == "summary":
            summary = self._summarize_global_contexts(query, contexts)
            if summary:
                source_list = [c["metadata"]["source"] for c in contexts]
                return [
                    {
                        "id": "global_summary",
                        "text": summary,
                        "metadata": {
                            "source": "global_summary",
                            "scope": "global_summary",
                            "sources": source_list,
                            "score_type": "derived_context_max_score",
                            "is_global_summary_block": True,
                        },
                        "score": max((c["score"] for c in contexts), default=0.0),
                    }
                ]

        return contexts

    def _summarize_global_contexts(self, query: str, contexts: list[dict]) -> str:
        if not self.summary_client or not contexts:
            return ""

        joined = "\n\n---\n\n".join(
            f"[来源: {c['metadata']['source']}]\n{c['text']}" for c in contexts
        )
        prompt = (
            "你是一个知识库摘要助手。请基于提供的材料回答用户问题，"
            "输出一段简洁摘要，并尽量在关键要点后用 [来源] 标注来源文件名。"
        )

        try:
            response = self.summary_client.chat.completions.create(
                model=config.INTENT_ROUTER_GLOBAL_SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"问题: {query}\n\n材料:\n{joined}"},
                ],
                temperature=0.2,
                max_tokens=config.INTENT_ROUTER_GLOBAL_SUMMARY_MAX_TOKENS,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("全局摘要生成失败: %s", exc)
            return ""

    # ── 主检索入口 ────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        mode: str = "auto",
        use_rerank: Optional[bool] = None,
        top_k: Optional[int] = None,
        final_k: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        检索入口。
        mode: "auto"(按配置), "vector", "bm25", "hybrid"
        use_rerank: 覆盖配置的 reranker 开关 (用于消融实验)
        history: 多轮对话历史，用于查询重写和指代消解
        """
        top_k = top_k or config.RETRIEVER_RECALL_TOP_K
        final_k = final_k or config.RETRIEVER_FINAL_K
        if use_rerank is None:
            use_rerank = self.use_reranker
        self.last_timing = {
            "rewrite_ms": 0.0,
            "align_ms": 0.0,
            "rerank_ms": 0.0,
            "route_decision": "local",
            "align_triggered": False,
            "rerank_active": False,
            "rewrite_triggered": False,
        }

        # ── 0. 查询重写 (指代消解) ──
        if self.use_query_rewriter and history:
            with Timer() as t_rewrite:
                rewritten_query = self.query_rewriter.rewrite(query, history)
            self.last_timing["rewrite_ms"] = t_rewrite.elapsed_ms
            if rewritten_query != query:
                self.last_timing["rewrite_triggered"] = True
                self.last_timing["original_query"] = query
                self.last_timing["rewritten_query"] = rewritten_query
                query = rewritten_query

        if mode == "auto" and self.use_intent_router:
            route, route_info = self.intent_router.route(query)
            self.last_timing["route"] = route
            self.last_timing["route_decision"] = route
            self.last_timing["route_info"] = route_info

            if route == "global":
                aligned_query, _alignment = self._align_query(query)
                self.last_timing["aligned_query"] = aligned_query
                self.last_timing["route_decision"] = "global"
                with Timer() as t_embed:
                    candidates = self._global_retrieve(
                        aligned_query,
                        top_k=config.INTENT_ROUTER_GLOBAL_TOP_K,
                        max_sources=config.INTENT_ROUTER_GLOBAL_MAX_SOURCES,
                        max_chars=config.INTENT_ROUTER_GLOBAL_MAX_CHARS,
                    )
                self.last_timing["embed_ms"] = t_embed.elapsed_ms
                self.last_timing["mode"] = "global"
                self.last_timing["rerank_active"] = False
            # 💡 扩充出口载荷，透传 channels 和 channel_scores
                return [
                    {
                        "text": c["text"], 
                        "metadata": c["metadata"], 
                        "score": c["score"],
                        "channels": c.get("channels", []),
                        "channel_scores": c.get("channel_scores", {})
                    }
                    for c in candidates
                ]

        aligned_query, _alignment = self._align_query(query)
        self.last_timing["aligned_query"] = aligned_query
        self.last_timing["route_decision"] = "local"

        if mode == "auto":
            mode = "hybrid" if self.use_hybrid else "vector"

        # ── Embedding + 检索 ──
        with Timer() as t_embed:
            if mode == "bm25":
                candidates = self._bm25_search(aligned_query, top_k)
            elif mode == "hybrid":
                vector_results = self._vector_search(aligned_query, top_k)
                bm25_results = self._bm25_search(aligned_query, top_k)
                candidates = self._merge_candidates(
                    ("vector", vector_results),
                    ("bm25", bm25_results),
                    k=config.RRF_K,
                    weights={
                        "vector": config.RETRIEVER_WEIGHT_VECTOR,
                        "bm25": config.RETRIEVER_WEIGHT_BM25,
                    },
                )
            else:  # vector
                candidates = self._vector_search(aligned_query, top_k)

        self.last_timing["embed_ms"] = t_embed.elapsed_ms

        if self.use_kg:
            base_count = len(candidates)
            with Timer() as t_kg:
                kg_candidates = self._kg_search(aligned_query, config.KG_TOP_K)
            self.last_timing["kg_ms"] = t_kg.elapsed_ms
            if kg_candidates:
                candidates = self._merge_candidates(
                    ("base", candidates),
                    ("kg", kg_candidates),
                    k=config.RRF_K,
                    weights={"base": 1.0, "kg": config.RETRIEVER_WEIGHT_KG},
                )
            merged_count = len(candidates)
            hit_rate = len(kg_candidates) / max(merged_count, 1)
            logger.info(
                "KG hits=%d, base=%d, merged=%d, hit_rate=%.2f",
                len(kg_candidates),
                base_count,
                merged_count,
                hit_rate,
            )

        logger.debug("检索模式=%s, 候选数=%d", mode, len(candidates))

        # ── Rerank ──
        rerank_ms = 0.0
        seed_candidates = candidates[:final_k]

        if use_rerank and self.use_reranker and seed_candidates:
            with Timer() as t_rerank:
                scores = [None] * len(seed_candidates)
                uncached_pairs = []
                uncached_indices = []

                # 1. 查询缓存 (LRU Cache拦截)
                for i, c in enumerate(seed_candidates):
                    doc_id = c["id"]
                    cached_score = self.rerank_cache.get((aligned_query, doc_id))
                    if cached_score is not None:
                        scores[i] = cached_score
                    else:
                        uncached_pairs.append([aligned_query, c["text"]])
                        uncached_indices.append(i)

                # 2. 对未命中缓存的候选块进行矩阵式批量推理 (Batching)
                if uncached_pairs:
                    batch_size = getattr(config, "RERANKER_BATCH_SIZE", 32)
                    batch_scores = self.reranker.predict(uncached_pairs, batch_size=batch_size)
                    
                    if isinstance(batch_scores, float) or np.isscalar(batch_scores):
                        batch_scores = [batch_scores]
                        
                    for idx, score in zip(uncached_indices, batch_scores):
                        scores[idx] = float(score)
                        self.rerank_cache.put((aligned_query, seed_candidates[idx]["id"]), float(score))

                # 3. 汇总与映射分数
                for c, s in zip(seed_candidates, scores):
                    c.setdefault("metadata", {})["raw_rerank_logit"] = float(s)
                    c["metadata"]["score_type"] = "rerank_compressed_score"
                    c["score"] = float(1 / (1 + np.exp(-s)))
                
                seed_candidates.sort(key=lambda x: x["score"], reverse=True)
            rerank_ms = t_rerank.elapsed_ms
            self.last_timing["rerank_active"] = True
        else:
            self.last_timing["rerank_active"] = False

        expanded_neighbors = self._expand_neighbor_chunks(seed_candidates)
        candidates = [*seed_candidates, *expanded_neighbors]

        deduped: list[dict] = []
        seen_ids: set[str] = set()
        for item in candidates:
            doc_id = item["id"]
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            deduped.append(item)
        candidates = deduped

        self.last_timing["rerank_ms"] = rerank_ms
        self.last_timing["mode"] = mode

# 去掉内部 id 字段再返回
        candidates = self._sort_local_contexts(candidates)
        # 💡 扩充出口载荷，确保精排、粗排后的多维通道特征完整流入 generator.py
        return [
            {
                "text": c["text"], 
                "metadata": c["metadata"], 
                "score": c["score"],
                "channels": c.get("channels", []),
                "channel_scores": c.get("channel_scores", {})
            }
            for c in candidates
        ]