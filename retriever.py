from sentence_transformers import CrossEncoder
import chromadb
from embedder import ZhipuEmbedder
import config

class Retriever:
    def __init__(self):
        self.embedder = ZhipuEmbedder()
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_collection(config.COLLECTION_NAME)
        
        self.use_reranker = config.USE_RERANKER
        if self.use_reranker:
            # 轻量 reranker，CPU 可跑
            self.reranker = CrossEncoder(
                config.RERANKER_MODEL, 
                max_length=512
            )

    def retrieve(self, query: str, top_k: int = None, final_k: int = None) -> list:
        """向量检索 + rerank"""
        top_k = top_k or config.RETRIEVER_TOP_K
        final_k = final_k or config.RETRIEVER_FINAL_K

        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

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
            reverse=True
        )

        return [
            {"text": doc, "metadata": meta, "score": float(score)}
            for doc, meta, score in ranked
        ][:final_k]
