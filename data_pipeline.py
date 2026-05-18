"""
数据处理 Pipeline：加载 Markdown、分块、Embedding、写入向量库
支持增量索引（基于文件 hash，只处理变更文件）
"""
import hashlib
import json
import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from embedder import ZhipuEmbedder
import config

logger = logging.getLogger(__name__)

# 索引元数据文件，记录每个文件的 hash
_INDEX_META_PATH = Path(config.CHROMA_DB_PATH) / ".index_meta.json"


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

    @staticmethod
    def _file_hash(path: Path) -> str:
        """计算文件内容的 MD5 hash"""
        return hashlib.md5(path.read_bytes()).hexdigest()

    @staticmethod
    def _make_chunk_id(relative_path: Path, chunk_index: int) -> str:
        """
        用相对路径生成唯一 chunk ID，避免不同子目录同名文件碰撞。
        例: notes/sub1/design.md chunk 0 → "sub1_design_0"
        """
        # 将路径分隔符和 .md 后缀去掉，用下划线连接
        parts = list(relative_path.with_suffix("").parts)
        return "_".join(parts) + f"_{chunk_index}"

    def _load_index_meta(self) -> dict:
        """加载已索引文件的 hash 记录"""
        if _INDEX_META_PATH.exists():
            return json.loads(_INDEX_META_PATH.read_text(encoding="utf-8"))
        return {}

    def _save_index_meta(self, meta: dict):
        """保存索引元数据"""
        _INDEX_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        _INDEX_META_PATH.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_markdown_dir(self, dir_path: str, incremental: bool = True) -> list[dict]:
        """
        递归加载目录下所有 markdown 文件。
        incremental=True 时只处理新增/变更的文件。
        """
        dir_path_obj = Path(dir_path)
        all_files = list(dir_path_obj.rglob("*.md"))
        index_meta = self._load_index_meta() if incremental else {}

        docs = []
        updated_meta: dict[str, str] = {}
        skipped = 0

        for path in all_files:
            relative = path.relative_to(dir_path_obj)
            file_key = str(relative)
            current_hash = self._file_hash(path)
            updated_meta[file_key] = current_hash

            # 增量模式：跳过未变更的文件
            if incremental and index_meta.get(file_key) == current_hash:
                skipped += 1
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error("读取文件失败 %s: %s", path, e)
                continue

            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "id": self._make_chunk_id(relative, i),
                    "text": chunk,
                    "metadata": {
                        "source": str(relative),
                        "chunk_index": i,
                    }
                })

        if skipped:
            logger.info("跳过 %d 个未变更文件", skipped)
        logger.info("待索引: %d 个文件, %d 个 chunks", len(all_files) - skipped, len(docs))

        # 保存元数据供下次增量判断
        self._save_index_meta(updated_meta)

        return docs

    def index(self, docs: list):
        """批量 embedding 并写入 ChromaDB"""
        if not docs:
            logger.info("没有需要索引的文档")
            return

        texts = [d["text"] for d in docs]
        ids = [d["id"] for d in docs]
        metadatas = [d["metadata"] for d in docs]

        logger.info("正在生成 Embedding (%d chunks)...", len(texts))
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

        # ChromaDB 支持 upsert，幂等写入
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("索引完成: %d chunks 已写入", len(docs))


if __name__ == "__main__":
    from logger import setup_logging
    setup_logging()

    processor = DocumentProcessor()
    docs = processor.load_markdown_dir(config.NOTES_DIR)
    processor.index(docs)
