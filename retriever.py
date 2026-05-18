from sentence_transformers import CrossEncoder
import chromadb
from embedder import ZhipuEmbedder

class Retriever:
    def __init__(self, collection_name="my_knowledge_base", use_reranker=True):
        self.embedder = ZhipuEmbedder()
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection(collection_name)
        
        self.use_reranker = use_reranker
        if use_reranker:
            # 轻量 reranker，CPU 可跑
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-v2-m3", 
                max_length=512
            )

    def retrieve(self, query: str, top_k: int = 5, final_k: int = 3) -> list:
        """向量检索 + rerank"""
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
