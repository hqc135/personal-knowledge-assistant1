"""
数据处理 Pipeline：加载 Markdown、分块、Embedding、写入向量库
支持增量索引（基于文件 hash，只处理变更文件）

分块策略：
  - 默认：SemanticChunker（语义梯度阈值切块）
  - 降级：RecursiveCharacterTextSplitter（固定字符窗口，fallback）
"""
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from embedder import ZhipuEmbedder
from kg_extractor import extract_triples
from kg_store import append_triples
import config

logger = logging.getLogger(__name__)

# 索引元数据文件，记录每个文件的 hash
_INDEX_META_PATH = Path(config.CHROMA_DB_PATH) / ".index_meta.json"

# ──────────────────────────────────────────────────────────────────────────────
# 句子拆分正则（中英文混合 + Markdown）
# ──────────────────────────────────────────────────────────────────────────────
_SENT_SPLIT_RE = re.compile(
    r"(?<=[。！？!?])"          # 中英文句末标点（后向断言）
    r"|(?<=\n\n)"               # 连续空行（Markdown 段落边界）
    r"|(?<=\n)(?=#{1,6}\s)"     # Markdown 标题行之前
    r"|(?<=[.!?][ \t])(?=[A-Z])",  # 英文句点 + 空格 + 大写首字（英文句尾）
)


class SemanticChunker:
    """
    基于余弦相似度梯度的语义分块器。

    算法流程：
      1. 正则拆分句子
      2. 滑动窗口平滑句向量（降低单句噪声）
      3. 计算相邻句向量余弦相似度
      4. 百分位阈值检测语义断点（话题转移）
      5. 合并过小碎片 / 强制拆分过大 chunk
    """

    def __init__(self, embedder: ZhipuEmbedder):
        self.embedder = embedder
        self.percentile = config.SEMANTIC_BREAKPOINT_PERCENTILE
        self.window = config.SEMANTIC_WINDOW_SIZE
        self.min_size = config.SEMANTIC_MIN_CHUNK_SIZE
        self.max_size = config.SEMANTIC_MAX_CHUNK_SIZE

    # ── 公共入口 ──────────────────────────────────────────────────────────────

    def split_text(self, text: str) -> list[str]:
        """将一段文本切分为语义连贯的 chunk 列表。"""
        sentences = self._sentence_split(text)

        if len(sentences) < 3:
            # 文档过短，无需相似度计算
            chunk = text.strip()
            return [chunk] if chunk else []

        try:
            vecs = self._windowed_embed(sentences)
            sims = self._cosine_sims(vecs)
            breakpoints = self._find_breakpoints(sims)
            chunks = self._build_chunks(sentences, breakpoints)
            return chunks
        except Exception as exc:
            logger.warning(
                "SemanticChunker 内部异常，跳过语义计算直接返回原文: %s", exc
            )
            # 轻量降级：直接把拼接后的全文作为单个 chunk 返回
            # （外层 DocumentProcessor 还有 fallback splitter 作第二重保障）
            return [text.strip()] if text.strip() else []

    # ── 句子拆分 ──────────────────────────────────────────────────────────────

    def _sentence_split(self, text: str) -> list[str]:
        """按标点 / 段落边界拆分，过滤过短噪声片段。"""
        parts = _SENT_SPLIT_RE.split(text)
        sentences = []
        for part in parts:
            part = part.strip()
            if len(part) >= 5:  # 过滤孤立标点、空白等噪声
                sentences.append(part)
        return sentences

    # ── 向量计算 ──────────────────────────────────────────────────────────────

    def _windowed_embed(self, sentences: list[str]) -> np.ndarray:
        """
        批量 Embed 所有句子，再做滑动窗口平均平滑。
        返回归一化后的向量矩阵 (N, D)。
        """
        vecs = self.embedder.encode(sentences, normalize_embeddings=True)  # (N, D)
        half = self.window // 2
        smoothed = []
        for i in range(len(vecs)):
            lo = max(0, i - half)
            hi = min(len(vecs), i + half + 1)
            smoothed.append(vecs[lo:hi].mean(axis=0))
        smoothed = np.array(smoothed, dtype=np.float32)
        # 重新归一化（平均后模长变化）
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return smoothed / norms

    @staticmethod
    def _cosine_sims(vecs: np.ndarray) -> np.ndarray:
        """
        计算相邻句向量的余弦相似度。
        向量已归一化，点积即余弦，返回长度 N-1 的数组。
        """
        return (vecs[:-1] * vecs[1:]).sum(axis=1)

    # ── 断点检测 ──────────────────────────────────────────────────────────────

    def _find_breakpoints(self, sims: np.ndarray) -> list[int]:
        """
        以第 percentile 百分位相似度为阈值，低于该值的位置即为语义断点。
        返回断点索引列表（断点 i 表示在 sentence[i] 和 sentence[i+1] 之间切）。
        """
        threshold = float(np.percentile(sims, self.percentile))
        return [i for i, s in enumerate(sims) if s < threshold]

    # ── chunk 组装 ────────────────────────────────────────────────────────────

    def _build_chunks(
        self, sentences: list[str], breakpoints: list[int]
    ) -> list[str]:
        """
        按断点将句子列表合并为 chunk，再执行：
          - 过小合并（< min_size）
          - 过大拆分（> max_size）
        """
        bp_set = set(breakpoints)
        raw_chunks: list[list[str]] = []
        current: list[str] = []

        for i, sent in enumerate(sentences):
            current.append(sent)
            # 断点在 i 处 → 在 sentence[i] 和 sentence[i+1] 之间切
            if i in bp_set and i < len(sentences) - 1:
                raw_chunks.append(current)
                current = []
        if current:
            raw_chunks.append(current)

        # 合并过小碎片
        merged = self._merge_small(raw_chunks)
        # 拆分过大单体
        result: list[str] = []
        for chunk_sents in merged:
            text = " ".join(chunk_sents)
            if len(text) > self.max_size:
                result.extend(self._split_oversized(chunk_sents))
            else:
                if text.strip():
                    result.append(text.strip())
        return result

    def _merge_small(self, chunks: list[list[str]]) -> list[list[str]]:
        """将字符数不足 min_size 的 chunk 向前合并。"""
        merged: list[list[str]] = []
        for sents in chunks:
            text = " ".join(sents)
            if merged and len(text) < self.min_size:
                merged[-1].extend(sents)
            else:
                merged.append(list(sents))
        return merged

    def _split_oversized(self, sentences: list[str]) -> list[str]:
        """
        对超过 max_size 的 chunk 按句子列表二分递归拆分。
        最小粒度为单句（不再下钻），防止无限递归。
        """
        if len(sentences) <= 1:
            # 单句超长：直接按字符截断（硬切保底）
            text = sentences[0] if sentences else ""
            return [
                text[i: i + self.max_size]
                for i in range(0, len(text), self.max_size)
                if text[i: i + self.max_size].strip()
            ]
        mid = len(sentences) // 2
        left = sentences[:mid]
        right = sentences[mid:]
        result: list[str] = []
        for half in (left, right):
            text = " ".join(half)
            if len(text) > self.max_size:
                result.extend(self._split_oversized(half))
            elif text.strip():
                result.append(text.strip())
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Document Processor
# ──────────────────────────────────────────────────────────────────────────────

class DocumentProcessor:
    def __init__(self):
        self.embedder = ZhipuEmbedder()
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hf_space": "personal_kb"}
        )

        # 根据配置选择分块器
        if config.SEMANTIC_CHUNKER_ENABLED:
            logger.info(
                "语义分块已启用（percentile=%.0f, window=%d, min=%d, max=%d）",
                config.SEMANTIC_BREAKPOINT_PERCENTILE,
                config.SEMANTIC_WINDOW_SIZE,
                config.SEMANTIC_MIN_CHUNK_SIZE,
                config.SEMANTIC_MAX_CHUNK_SIZE,
            )
            self._semantic_chunker: Optional[SemanticChunker] = SemanticChunker(
                self.embedder
            )
        else:
            self._semantic_chunker = None
            logger.info("语义分块已关闭，使用 RecursiveCharacterTextSplitter 作为后备")

        # fallback splitter（语义模式下异常降级时使用）
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ""],
        )

    def _split_document(self, text: str, source: str) -> tuple[list[str], str]:
        """
        对单篇文档执行分块，返回 (chunks, chunker_name)。
        优先语义分块，失败时自动降级至字符切块。
        """
        if self._semantic_chunker is not None:
            try:
                chunks = self._semantic_chunker.split_text(text)
                if chunks:
                    return chunks, "semantic"
                # split_text 返回空列表时视为异常
                logger.warning("语义分块返回空结果，降级处理文件: %s", source)
            except Exception as exc:
                logger.warning(
                    "语义分块失败，降级至字符切块 [%s]: %s", source, exc
                )

        # fallback
        return self._fallback_splitter.split_text(text), "recursive"

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

            chunks, chunker_name = self._split_document(text, str(relative))
            for i, chunk in enumerate(chunks):
                docs.append({
                    "id": self._make_chunk_id(relative, i),
                    "text": chunk,
                    "metadata": {
                        "source": str(relative),
                        "chunk_index": i,
                        "chunker": chunker_name,   # 记录实际使用的分块器
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

        if config.USE_KG_EXTRACTION:
            workers = config.KG_EXTRACTION_WORKERS
            logger.info(
                "开始并发抽取知识图谱三元组（workers=%d, chunks=%d）...",
                workers, len(docs),
            )
            all_triples: list[dict] = []

            def _extract_one(doc: dict) -> list[dict]:
                return extract_triples(
                    doc["text"],
                    source=doc["metadata"]["source"],
                    chunk_id=doc["id"],
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_extract_one, doc): doc for doc in docs}
                for future in as_completed(futures):
                    doc = futures[future]
                    try:
                        triples = future.result()
                        all_triples.extend(triples)
                    except Exception as exc:
                        logger.warning("三元组抽取失败 %s: %s", doc["id"], exc)

            added = append_triples(all_triples)
            logger.info("三元组写入完成: 新增 %d 条", added)


if __name__ == "__main__":
    from logger import setup_logging
    setup_logging()

    processor = DocumentProcessor()
    docs = processor.load_markdown_dir(config.NOTES_DIR)
    processor.index(docs)
