"""
Retriever 工厂：负责按配置组装所有检索子组件并注入 Retriever。

- 原始 retriever.py 保留向后兼容的 Retriever 类（kwargs 初始化方式不变）
- 本工厂供 app.py / pipeline.py 使用，负责依赖组装，并启动后台预热线程
- ablation.py 可继续使用 retriever.Retriever(use_reranker=True, ...) 的旧方式
"""
from __future__ import annotations

import logging
import threading

import chromadb

from embedder import ZhipuEmbedder
from kg_retriever import KGRetriever
from intent_router import IntentRouter
from query_aligner import QueryAligner
from query_rewriter import QueryRewriter
from retrieval.bm25_index import BM25Index
from retrieval.reranker import Reranker
from retrieval.neighbor_expander import NeighborExpander
import config

logger = logging.getLogger(__name__)


def build_retriever(**overrides):
    """
    组装并返回一个完全初始化的 Retriever 实例。

    overrides 可覆盖任何 config.* 开关，与 ablation.py 的 kwargs 语义一致：
        use_reranker, use_hybrid, use_kg, use_intent_router,
        use_query_aligner, use_query_rewriter
    """
    # 延迟导入，避免循环依赖
    from retriever import Retriever

    flags = {
        "use_reranker": overrides.get("use_reranker", config.USE_RERANKER),
        "use_hybrid": overrides.get("use_hybrid", config.USE_HYBRID_SEARCH),
        "use_kg": overrides.get("use_kg", config.USE_KG_RETRIEVAL),
        "use_intent_router": overrides.get("use_intent_router", config.USE_INTENT_ROUTER),
        "use_query_aligner": overrides.get("use_query_aligner", config.USE_QUERY_ALIGNER),
        "use_query_rewriter": overrides.get("use_query_rewriter", config.USE_QUERY_REWRITER),
    }

    retriever = Retriever(**flags)
    return retriever


def _warmup_worker(retriever) -> None:
    """后台预热线程目标函数，解耦启动生命周期。"""
    import jieba

    logger.info("后台预热线程已启动...")
    try:
        # 1. jieba 分词
        jieba.initialize()

        # 2. Reranker
        if getattr(retriever, "use_reranker", False) and hasattr(retriever, "reranker"):
            retriever.reranker.predict([["预热", "测试"]])

        # 3. ChromaDB
        dummy_embedding = [0.0] * config.EMBEDDING_DIMENSIONS
        retriever.collection.query(query_embeddings=[dummy_embedding], n_results=1)

        # 4. KG 实体向量索引
        if getattr(retriever, "use_kg", False) and hasattr(retriever, "kg"):
            retriever.kg._get_entity_vector_index()

        logger.info("后台预热全部完成！首次查询将极速响应。")
    except Exception as exc:
        logger.warning("后台预热出现异常（不影响运行）: %s", exc)
