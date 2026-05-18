"""
简单评估框架：不依赖 GPT-4 打分，用规则 + embedding 相似度
"""
import logging

import numpy as np
import json
from embedder import ZhipuEmbedder

logger = logging.getLogger(__name__)


class SimpleEvaluator:
    def __init__(self):
        self.embedder = ZhipuEmbedder()

    def relevance_score(self, query: str, retrieved_texts: list[str]) -> float:
        """检索相关性：query 与 retrieved chunks 的平均余弦相似度"""
        # encode(str) 返回 1D (dim,)，encode(list) 返回 2D (N, dim)
        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        doc_embs = self.embedder.encode(retrieved_texts, normalize_embeddings=True)
        # (N, dim) @ (dim,) → (N,)
        scores = np.dot(doc_embs, query_emb)
        return float(np.mean(scores))

    def faithfulness_check(self, answer: str, contexts: list[str]) -> dict:
        """忠实度检查：答案是否基于上下文（简单版）"""
        answer_emb = self.embedder.encode(answer, normalize_embeddings=True)
        ctx_embs = self.embedder.encode(contexts, normalize_embeddings=True)
        # (N, dim) @ (dim,) → (N,)
        max_sim = float(np.max(np.dot(ctx_embs, answer_emb)))
        return {
            "max_context_similarity": max_sim,
            "likely_grounded": max_sim > 0.5,
        }

    def run_eval(self, test_cases: list[dict], retriever, generator) -> dict:
        """批量评估"""
        results = []
        for i, case in enumerate(test_cases):
            query = case["query"]
            logger.info("评估 [%d/%d]: %s", i + 1, len(test_cases), query)

            try:
                contexts = retriever.retrieve(query)
                answer = generator.generate(query, contexts)

                relevance = self.relevance_score(
                    query, [c["text"] for c in contexts]
                )
                faithfulness = self.faithfulness_check(
                    answer, [c["text"] for c in contexts]
                )

                results.append({
                    "query": query,
                    "relevance": relevance,
                    "faithfulness": faithfulness,
                    "answer_length": len(answer),
                })
            except Exception as e:
                logger.error("评估失败 [%s]: %s", query, e)
                results.append({
                    "query": query,
                    "relevance": 0.0,
                    "faithfulness": {"max_context_similarity": 0, "likely_grounded": False},
                    "answer_length": 0,
                    "error": str(e),
                })

        avg_relevance = np.mean([r["relevance"] for r in results])
        grounded_rate = np.mean([
            r["faithfulness"]["likely_grounded"] for r in results
        ])

        summary = {
            "avg_relevance": float(avg_relevance),
            "grounded_rate": float(grounded_rate),
            "num_cases": len(results),
            "details": results,
        }

        logger.info(
            "评估完成: 平均相关性=%.3f, 忠实率=%.1f%%",
            summary["avg_relevance"],
            summary["grounded_rate"] * 100,
        )

        return summary
