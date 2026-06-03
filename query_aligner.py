"""
查询对齐助手：在检索前做中英术语对齐与受控扩展。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


TERM_ALIASES: list[dict[str, Any]] = [
    {"canonical": "RAG", "aliases": ["rag", "检索增强生成", "检索增强", "retrieval augmented generation"]},
    {"canonical": "vector database", "aliases": ["向量数据库", "向量库", "vector database", "vector store", "chromadb", "chroma db"]},
    {"canonical": "embedding", "aliases": ["embedding", "向量表示", "嵌入", "词向量"]},
    {"canonical": "BM25", "aliases": ["bm25", "关键词检索", "词项检索"]},
    {"canonical": "reranker", "aliases": ["reranker", "重排序", "rerank", "重排"]},
    {"canonical": "knowledge graph", "aliases": ["知识图谱", "knowledge graph", "kg", "graph retrieval"]},
    {"canonical": "intent router", "aliases": ["意图路由", "intent router", "routing"]},
    {"canonical": "chunking", "aliases": ["chunking", "分块", "切分", "文本切分"]},
    {"canonical": "tokenization", "aliases": ["tokenization", "分词", "词元化"]},
    {"canonical": "prompt engineering", "aliases": ["prompt engineering", "提示词", "提示工程"]},
    {"canonical": "transformer", "aliases": ["transformer", "注意力机制", "自注意力"]},
    {"canonical": "retrieval", "aliases": ["retrieval", "检索", "召回"]},
]


@dataclass
class QueryAlignmentResult:
    original_query: str
    search_query: str
    should_expand: bool
    confidence: float
    reason: str
    fallback_reason: str | None = None
    detected_terms: list[dict[str, Any]] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    language: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "search_query": self.search_query,
            "should_expand": self.should_expand,
            "confidence": self.confidence,
            "reason": self.reason,
            "fallback_reason": self.fallback_reason,
            "detected_terms": self.detected_terms,
            "expansions": self.expansions,
            "language": self.language,
        }


class QueryAligner:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        min_query_length: int | None = None,
        min_confidence: float | None = None,
        max_expansions: int | None = None,
        max_terms: int | None = None,
    ) -> None:
        self.client = client or OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=15.0,
            max_retries=1,
        )
        self.model = model or config.QUERY_ALIGNER_MODEL
        self.min_query_length = (
            config.QUERY_ALIGNER_MIN_QUERY_LENGTH if min_query_length is None else min_query_length
        )
        self.min_confidence = (
            config.QUERY_ALIGNER_MIN_CONFIDENCE if min_confidence is None else min_confidence
        )
        self.max_expansions = (
            config.QUERY_ALIGNER_MAX_EXPANSIONS if max_expansions is None else max_expansions
        )
        self.max_terms = config.QUERY_ALIGNER_MAX_TERMS if max_terms is None else max_terms

    @staticmethod
    def _has_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _has_latin(text: str) -> bool:
        return bool(re.search(r"[A-Za-z]", text))

    @staticmethod
    def _looks_like_code_or_path(text: str) -> bool:
        lowered = text.lower()
        return bool(
            re.search(r"https?://|\\|/|::", lowered)
            or any(token in lowered for token in ["def ", "class ", "import ", "lambda ", "function("])
        )

    @staticmethod
    def _extract_terms(query: str) -> list[dict[str, Any]]:
        lowered = query.lower()
        hits: list[dict[str, Any]] = []
        for entry in TERM_ALIASES:
            aliases = entry["aliases"]
            matched_aliases = [alias for alias in aliases if alias.lower() in lowered]
            if matched_aliases:
                hits.append(
                    {
                        "canonical": entry["canonical"],
                        "aliases": matched_aliases[:3],
                    }
                )
        return hits

    def _deterministic_gate(self, query: str) -> tuple[bool, str, list[dict[str, Any]], str]:
        cleaned = query.strip()
        if not cleaned:
            return False, "empty_query", [], "unknown"
        if len(cleaned) < self.min_query_length:
            return False, "too_short", [], "unknown"
        if self._looks_like_code_or_path(cleaned):
            return False, "code_or_path_query", [], "unknown"

        detected_terms = self._extract_terms(cleaned)[: self.max_terms]
        has_cjk = self._has_cjk(cleaned)
        has_latin = self._has_latin(cleaned)
        language = "mixed" if has_cjk and has_latin else ("zh" if has_cjk else ("en" if has_latin else "unknown"))

        if not detected_terms and language != "mixed":
            return False, "no_alignment_signal", [], language

        return True, "gate_passed", detected_terms, language

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        content = text.strip()
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                return json.loads(match.group(0))
            raise

    def _sanitize_expansions(self, payload: dict[str, Any], original_query: str) -> tuple[list[str], float, str, str]:
        expansions: list[str] = []
        raw_expansions = payload.get("expansions", [])
        if isinstance(raw_expansions, dict):
            raw_items: list[Any] = []
            for value in raw_expansions.values():
                if isinstance(value, list):
                    raw_items.extend(value)
                elif isinstance(value, str):
                    raw_items.append(value)
        elif isinstance(raw_expansions, list):
            raw_items = raw_expansions
        else:
            raw_items = []

        for item in raw_items:
            text = str(item).strip()
            if not text or text == original_query:
                continue
            if text not in expansions:
                expansions.append(text)
            if len(expansions) >= self.max_expansions:
                break

        search_query = str(payload.get("search_query") or "").strip()
        if not search_query:
            search_query = original_query if not expansions else f"{original_query} {' '.join(expansions)}"

        confidence = float(payload.get("confidence", 0.0) or 0.0)
        reason = str(payload.get("reason", "")) or "aligned"
        return expansions, confidence, search_query or original_query, reason

    def _fallback(self, query: str, reason: str, language: str = "unknown") -> QueryAlignmentResult:
        return QueryAlignmentResult(
            original_query=query,
            search_query=query,
            should_expand=False,
            confidence=0.0,
            reason="fallback",
            fallback_reason=reason,
            detected_terms=self._extract_terms(query)[: self.max_terms],
            expansions=[],
            language=language,
        )

    def align(self, query: str) -> QueryAlignmentResult:
        gate_passed, gate_reason, detected_terms, language = self._deterministic_gate(query)
        if not gate_passed:
            return self._fallback(query, gate_reason, language)

        prompt = (
            "你是一个查询对齐助手，只做中英文术语对齐与受控扩展。"
            "不要改写用户真实意图，不要扩展成无关长句。"
            "优先围绕专有名词、技术名词、缩写生成最小必要扩展。\n\n"
            f"用户查询: {query}\n\n"
            f"已识别的候选术语: {json.dumps(detected_terms, ensure_ascii=False)}\n\n"
            "请严格输出 JSON，格式如下：\n"
            "{\n"
            '  "should_expand": true,\n'
            '  "confidence": 0.0,\n'
            '  "search_query": "原查询加上必要的中英文术语扩展",\n'
            '  "expansions": ["扩展1", "扩展2"],\n'
            '  "aligned_terms": [{"source": "中文术语", "target": ["english term"]}],\n'
            '  "reason": "简短原因"\n'
            "}\n"
            f"扩展条数最多 {self.max_expansions} 条，confidence 必须是 0 到 1 之间的小数。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            message = response.choices[0].message
            content = message.content or ""

            # DeepSeek 推理模型可能返回空 content + reasoning_content
            if not content.strip():
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning:
                    logger.info("QueryAligner 收到 reasoning-only 响应，尝试从推理内容提取")
                    # 尝试从 reasoning_content 中提取 JSON
                    try:
                        payload = self._extract_json(reasoning)
                    except Exception:
                        logger.warning("QueryAligner reasoning 内容无法解析为 JSON，回退")
                        return self._fallback(query, "reasoning_only_no_json", language)
                else:
                    logger.warning("QueryAligner 收到空响应，回退")
                    return self._fallback(query, "empty_response", language)
            else:
                payload = self._extract_json(content)
        except Exception as exc:
            logger.warning("QueryAligner 失败，回退到原查询: %s", exc)
            return self._fallback(query, "llm_error", language)

        if not isinstance(payload, dict):
            return self._fallback(query, "invalid_payload", language)

        expansions, confidence, search_query, reason = self._sanitize_expansions(payload, query)
        should_expand = bool(payload.get("should_expand", True))
        if not should_expand or confidence < self.min_confidence or not expansions:
            fallback_reason = "model_declined" if not should_expand else (
                "low_confidence" if confidence < self.min_confidence else "empty_expansion"
            )
            return self._fallback(query, fallback_reason, language)

        return QueryAlignmentResult(
            original_query=query,
            search_query=search_query,
            should_expand=True,
            confidence=confidence,
            reason=reason,
            fallback_reason=None,
            detected_terms=detected_terms,
            expansions=expansions,
            language=language,
        )