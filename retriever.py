import logging

from sentence_transformers import CrossEncoder
import chromadb
from embedder import ZhipuEmbedder
import config

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        self.embedder = ZhipuEmbedder()
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_collection(config.COLLECTION_NAME)

        self.use_reranker = config.USE_RERANKER
        if self.use_reranker:
            logger.info("加载 Reranker 模型: %s", config.RERANKER_MODEL)
            self.reranker = CrossEncoder(
                config.RERANKER_MODEL,
                max_length=512,
            )

    def retrieve(self, query: str, top_k: int = None, final_k: int = None) -> list:
        """向量检索 + rerank"""
        top_k = top_k or config.RETRIEVER_TOP_K
        final_k = final_k or config.RETRIEVER_FINAL_K

        # encode 单条返回 1D (dim,)，转为列表后是 [float, ...]
        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        logger.debug("检索到 %d 个候选文档", len(documents))

        if not self.use_reranker:
            return [
                {"text": doc, "metadata": meta, "score": 1 - dist}
                for doc, meta, dist in zip(documents, metadatas, distances)
            ][:final_k]

        # Rerank
        pairs = [[query, doc] for doc in documents]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(documents, metadatas, rerank_scores),
            key=lambda x: x[2],
            reverse=True,
        )

        logger.debug("Rerank 完成，top score: %.3f", ranked[0][2] if ranked else 0)

        return [
            {"text": doc, "metadata": meta, "score": float(score)}
            for doc, meta, score in ranked
        ][:final_k]
