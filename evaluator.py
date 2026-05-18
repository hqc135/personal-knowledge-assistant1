"""
简单评估框架：不依赖 GPT-4 打分，用规则 + embedding 相似度
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class SimpleEvaluator:
    def __init__(self):
        self.embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def relevance_score(self, query: str, retrieved_texts: list[str]) -> float:
        """检索相关性：query 与 retrieved chunks 的平均余弦相似度"""
        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        doc_embs = self.embedder.encode(retrieved_texts, normalize_embeddings=True)
        scores = np.dot(doc_embs, query_emb)
        return float(np.mean(scores))

    def faithfulness_check(self, answer: str, contexts: list[str]) -> dict:
        """忠实度检查：答案是否基于上下文（简单版）"""
        answer_emb = self.embedder.encode(answer, normalize_embeddings=True)
        ctx_embs = self.embedder.encode(contexts, normalize_embeddings=True)
        max_sim = float(np.max(np.dot(ctx_embs, answer_emb)))
        return {
            "max_context_similarity": max_sim,
            "likely_grounded": max_sim > 0.5
        }

    def run_eval(self, test_cases: list[dict], retriever, generator) -> dict:
        """批量评估"""
        results = []
        for case in test_cases:
            query = case["query"]
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
                "answer_length": len(answer)
            })

        avg_relevance = np.mean([r["relevance"] for r in results])
        grounded_rate = np.mean([
            r["faithfulness"]["likely_grounded"] for r in results
        ])

        return {
            "avg_relevance": float(avg_relevance),
            "grounded_rate": float(grounded_rate),
            "num_cases": len(results),
            "details": results
        }
