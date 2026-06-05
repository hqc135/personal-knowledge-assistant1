"""
Parent-Child 分层分块器。

将文档拆分为两层 chunk：
  - Parent Chunk：按 Markdown 标题层级 / 段落边界切分的大块（1500~2500 chars），
    保留完整上下文，用于喂给 LLM。
  - Child Chunk：对每个 Parent 内部进一步用 SemanticChunker 切出的细粒度小块
    （200~400 chars），用于向量 / BM25 检索。

两层通过 parent_id / child_ids 元数据双向关联。
"""
import logging
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)

# Markdown 标题行正则（匹配 ## 和 ### 级别）
_HEADING_RE = re.compile(r"^(#{1,3})\s+", re.MULTILINE)


# ──────────────────────────────────────────────────────────────────────────────
# ID 生成工具
# ──────────────────────────────────────────────────────────────────────────────

def _make_parent_id(relative_path: Path, parent_index: int) -> str:
    """
    生成 parent chunk ID。
    例: notes/chunking-strategies.md, index=0 → "chunking-strategies_parent_0"
    """
    parts = list(relative_path.with_suffix("").parts)
    return "_".join(parts) + f"_parent_{parent_index}"


def _make_child_id(relative_path: Path, parent_index: int, child_index: int) -> str:
    """
    生成 child chunk ID。
    例: notes/chunking-strategies.md, p=0, c=2 → "chunking-strategies_child_0_2"
    """
    parts = list(relative_path.with_suffix("").parts)
    return "_".join(parts) + f"_child_{parent_index}_{child_index}"


# ──────────────────────────────────────────────────────────────────────────────
# Parent 切分逻辑
# ──────────────────────────────────────────────────────────────────────────────

def _split_by_headings(text: str) -> list[str]:
    """
    按 Markdown 标题（# / ## / ###）切分文档。

    每个标题行作为切分点，标题本身归入后续段落。
    返回的每个元素是一个"段落块"（从标题到下一个标题之前）。
    """
    # 找到所有标题的起始位置
    positions = [m.start() for m in _HEADING_RE.finditer(text)]

    if not positions:
        # 没有标题，整篇文档作为单个块
        return [text.strip()] if text.strip() else []

    sections: list[str] = []

    # 如果文档开头有标题之前的内容，作为第一个 section
    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            sections.append(preamble)

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections


def _merge_small_sections(
    sections: list[str],
    min_size: int,
    max_size: int,
) -> list[str]:
    """
    合并过小的 section，拆分过大的 section。

    1. 连续的小 section 向前合并，直到达到 min_size
    2. 超过 max_size 的 section 按段落边界强制拆分
    """
    if not sections:
        return []

    # ── 合并过小 ──
    merged: list[str] = []
    for section in sections:
        if merged and len(merged[-1]) < min_size:
            merged[-1] = merged[-1] + "\n\n" + section
        else:
            merged.append(section)

    # ── 拆分过大 ──
    result: list[str] = []
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ""],
    )
    for section in merged:
        if len(section) > max_size:
            sub_chunks = fallback_splitter.split_text(section)
            result.extend(sub_chunks)
        else:
            result.append(section)

    return [s for s in result if s.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# ParentChildChunker
# ──────────────────────────────────────────────────────────────────────────────

class ParentChildChunker:
    """
    两层分块器：

    - 第一层 (Parent): 按 Markdown 标题 / 段落切分，目标大小由
      PARENT_CHUNK_MIN_SIZE ~ PARENT_CHUNK_MAX_SIZE 控制。
    - 第二层 (Child): 对每个 Parent 内容使用 SemanticChunker（或 fallback
      RecursiveCharacterTextSplitter）进一步切分为细粒度块。

    返回的 parent_docs 和 child_docs 通过 parent_id / child_ids 元数据关联。
    """

    def __init__(self, embedder=None):
        """
        Args:
            embedder: ZhipuEmbedder 实例，用于 SemanticChunker。
                      如果为 None 或语义分块未启用，退化为 RecursiveCharacterTextSplitter。
        """
        self._parent_min = config.PARENT_CHUNK_MIN_SIZE
        self._parent_max = config.PARENT_CHUNK_MAX_SIZE
        self._child_min = config.CHILD_CHUNK_MIN_SIZE
        self._child_max = config.CHILD_CHUNK_MAX_SIZE

        # 构建 child 分块器
        self._semantic_chunker = None
        if config.SEMANTIC_CHUNKER_ENABLED and embedder is not None:
            from data_pipeline import SemanticChunker
            # 用 child 级别参数覆盖全局默认值
            self._semantic_chunker = SemanticChunker(embedder)
            self._semantic_chunker.min_size = self._child_min
            self._semantic_chunker.max_size = self._child_max

        # fallback child splitter
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_max,
            chunk_overlap=40,
            separators=["\n\n", "\n", "。", ""],
        )

    # ── 公共入口 ──────────────────────────────────────────────────────────────

    def split_document(
        self,
        text: str,
        source: str,
    ) -> tuple[list[dict], list[dict]]:
        """
        对单篇文档执行 Parent-Child 两层分块。

        Args:
            text: 文档全文
            source: 文件相对路径（如 "chunking-strategies.md"）

        Returns:
            (parent_docs, child_docs)
            parent_docs: [{"id": str, "text": str, "metadata": dict}, ...]
            child_docs:  [{"id": str, "text": str, "metadata": dict}, ...]
        """
        relative_path = Path(source)

        # ── 第一层：Parent 切分 ──
        sections = _split_by_headings(text)
        parent_texts = _merge_small_sections(
            sections, self._parent_min, self._parent_max
        )

        if not parent_texts:
            # 极端情况：空文档
            return [], []

        parent_docs: list[dict] = []
        child_docs: list[dict] = []

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = _make_parent_id(relative_path, p_idx)

            # ── 第二层：Child 切分 ──
            child_dicts = self._split_children(parent_text, source)
            child_ids: list[str] = []

            for c_idx, child_dict in enumerate(child_dicts):
                child_id = _make_child_id(relative_path, p_idx, c_idx)
                child_ids.append(child_id)
                
                c_meta = {
                    "source": source,
                    "parent_id": parent_id,
                    "parent_index": p_idx,
                    "child_index": c_idx,
                    "chunk_type": "child",
                }
                if child_dict.get("is_code"):
                    c_meta["is_code"] = True
                    if child_dict.get("code_language"):
                        c_meta["code_language"] = child_dict["code_language"]

                child_docs.append({
                    "id": child_id,
                    "text": child_dict["text"],
                    "metadata": c_meta,
                })

            is_parent_code = False
            parent_code_lang = None
            parent_text_stripped = parent_text.strip()
            if parent_text_stripped.startswith("```") and parent_text_stripped.endswith("```"):
                parts = parent_text_stripped.split("```")
                if len(parts) == 3 and not parts[0] and not parts[2]:
                    is_parent_code = True
                    lines = parent_text_stripped.split("\n", 1)
                    if len(lines) > 1:
                        parent_code_lang = lines[0][3:].strip()

            p_meta = {
                "source": source,
                "parent_index": p_idx,
                "child_count": len(child_ids),
                "chunk_type": "parent",
            }
            if is_parent_code:
                p_meta["is_code"] = True
                if parent_code_lang:
                    p_meta["code_language"] = parent_code_lang

            parent_docs.append({
                "id": parent_id,
                "text": parent_text,
                "metadata": p_meta,
            })

        logger.info(
            "Parent-Child 分块完成 [%s]: %d parents, %d children",
            source,
            len(parent_docs),
            len(child_docs),
        )
        return parent_docs, child_docs

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _split_children(self, parent_text: str, source: str) -> list[dict]:
        """
        对单个 parent chunk 执行 child 级别细粒度切分。
        优先使用 SemanticChunker，失败时降级至 RecursiveCharacterTextSplitter。
        """
        if self._semantic_chunker is not None:
            try:
                chunks = self._semantic_chunker.split_text(parent_text)
                if chunks:
                    return chunks
                logger.warning(
                    "SemanticChunker 返回空结果，降级处理 parent [%s]", source
                )
            except Exception as exc:
                logger.warning(
                    "SemanticChunker 异常，降级至字符切块 [%s]: %s", source, exc
                )

        fallback_chunks = self._fallback_splitter.split_text(parent_text)
        return [{"text": c, "is_code": False, "code_language": None} for c in fallback_chunks]
