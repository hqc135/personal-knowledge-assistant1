import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from embedder import ZhipuEmbedder
import config

class DocumentProcessor:
    def __init__(self):
        self.embedder = ZhipuEmbedder()
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hf_space": "personal_kb"}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ""]
        )

    def load_markdown_dir(self, dir_path: str):
        """递归加载目录下所有 markdown 文件"""
        docs = []
        for path in Path(dir_path).rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "id": f"{path.stem}_{i}",
                    "text": chunk,
                    "metadata": {
                        "source": str(path),
                        "chunk_index": i
                    }
                })
        return docs

    def index(self, docs: list):
        """批量 embedding 并写入 ChromaDB"""
        texts = [d["text"] for d in docs]
        ids = [d["id"] for d in docs]
        metadatas = [d["metadata"] for d in docs]

        embeddings = self.embedder.encode(
            texts, 
            normalize_embeddings=True,
        ).tolist()

        # ChromaDB 支持 upsert，幂等写入
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Indexed {len(docs)} chunks")


if __name__ == "__main__":
    processor = DocumentProcessor()
    docs = processor.load_markdown_dir(config.NOTES_DIR)
    processor.index(docs)
