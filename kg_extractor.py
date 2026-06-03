"""LLM-based triple extraction for knowledge graph building."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterable

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_PROMPT = """You are an information extraction system.
Extract up to {max_triples} triples from the text. Each triple is (head, relation, tail).
Canonicalization rules for entities:
1) head and tail MUST use a single canonical entity form, not aliases, pronouns, or paraphrases.
2) If the same entity appears multiple times, use the same exact surface form everywhere.
3) Prefer the most specific stable name in the text; do not invent new names.
4) Keep entity text concise and normalized, but do not over-compress meaningful technical terms.
5) If a mention is ambiguous, pick the clearest canonical name that best matches the text context.
6) Relation should be a concise normalized predicate phrase.
Return ONLY valid JSON in one of the following formats:
1) [{{"head": "...", "relation": "...", "tail": "..."}}, ...]
2) {{"triples": [{{"head": "...", "relation": "...", "tail": "..."}}, ...]}}

Text:
{text}
"""


def _extract_json(text: str) -> object:
    content = text.strip()
    if not content:
        raise ValueError("empty model response")

    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            decoder = json.JSONDecoder()
            parsed, end = decoder.raw_decode(content)
            if content[end:].strip():
                raise json.JSONDecodeError("extra data", content, end)
            return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", content)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_item(item: dict) -> dict | None:
    head = item.get("head") or item.get("h") or item.get("entity1")
    relation = item.get("relation") or item.get("rel") or item.get("predicate")
    tail = item.get("tail") or item.get("t") or item.get("entity2")
    if not head or not relation or not tail:
        return None
    return {"head": str(head).strip(), "relation": str(relation).strip(), "tail": str(tail).strip()}


def _finalize_triples(triples: Iterable[dict], source: str, chunk_id: str) -> list[dict]:
    output: list[dict] = []
    for item in triples:
        normalized = _normalize_item(item)
        if not normalized:
            continue
        output.append({
            "head": normalized["head"],
            "relation": normalized["relation"],
            "tail": normalized["tail"],
            "source": source,
            "chunk_id": chunk_id,
        })
    return output


def _postprocess_triples(triples: list[dict]) -> list[dict]:
    """简单后处理规则：
    - 去重（head,relation,tail）
    - 过滤 head/tail/ relation 过短或过长的项
    - 规范 relation 空白
    """
    seen = set()
    out: list[dict] = []
    for t in triples:
        h = (t.get("head") or "").strip()
        r = (t.get("relation") or "").strip()
        ta = (t.get("tail") or "").strip()

        # normalize whitespace in relation
        r = " ".join(r.split())

        # length filters: require non-empty head/tail; allow single-character entities
        if len(h) == 0 or len(ta) == 0:
            continue
        if len(h) > 200 or len(ta) > 500 or len(r) > 120:
            continue

        key = (h.lower(), r.lower(), ta.lower())
        if key in seen:
            continue
        seen.add(key)

        out.append({"head": h, "relation": r, "tail": ta, "source": t.get("source"), "chunk_id": t.get("chunk_id")})

    return out


def _create_client() -> OpenAI:
    return OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.LLM_BASE_URL,
        timeout=30.0,
        max_retries=0,
    )


def _request_kwargs(*, model: str, use_response_format: bool, prompt: str) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500,
    }
    if use_response_format:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def _supports_response_format(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message


def extract_triples(text: str, *, source: str, chunk_id: str) -> list[dict]:
    if not config.USE_KG_EXTRACTION:
        return []

    prompt = _PROMPT.format(text=text, max_triples=config.KG_MAX_TRIPLES_PER_CHUNK)
    client = _create_client()
    use_response_format = True
    model = config.KG_LLM_MODEL
    fallback_model = config.KG_FALLBACK_MODEL
    last_error: Exception | None = None

    data = None
    for attempt in range(1, 4):
        try:
            request_kwargs = _request_kwargs(model=model, use_response_format=use_response_format, prompt=prompt)
            response = client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message
            content = message.content or ""
            if not content:
                finish_reason = response.choices[0].finish_reason
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content:
                    # 尝试用模型的 internal reasoning 作为上下文，直接请求只输出 JSON
                    logger.info("KG extraction received reasoning-only output for %s, attempting direct JSON re-request", chunk_id)
                    follow_messages = [
                        {"role": "assistant", "content": reasoning_content},
                        {
                            "role": "user",
                            "content": (
                                "基于上面的内部推理结果，\n" 
                                "请只返回严格的 JSON，格式为数组或 {'triples': [...] }，\n"
                                "每个 triple 为 {\"head\": ..., \"relation\": ..., \"tail\": ...}，\n"
                                "不要带任何额外文字或解释。"
                            ),
                        },
                    ]
                    try:
                        follow_kwargs = {
                            "model": model,
                            "messages": follow_messages,
                            "temperature": 0,
                            "max_tokens": 500,
                        }
                        if use_response_format:
                            follow_kwargs["response_format"] = {"type": "json_object"}
                        follow_resp = client.chat.completions.create(**follow_kwargs)
                        follow_msg = follow_resp.choices[0].message
                        follow_content = follow_msg.content or ""
                        if follow_content:
                            data = _extract_json(follow_content)
                        else:
                            raise ValueError("empty model response after follow-up")
                    except Exception as exc_follow:
                        # 把 follow-up 的异常当作原始异常继续重试/回退逻辑
                        raise exc_follow
                else:
                    raise ValueError("empty model response")

            # 如果上面没有通过 follow-up 填充 data，则在这里解析 content
            if data is None:
                data = _extract_json(content)

            if isinstance(data, dict) and "triples" in data:
                data = data["triples"]

            if not isinstance(data, list):
                return []

            triples = _finalize_triples(data, source, chunk_id)
            processed = _postprocess_triples(triples)
            return processed[: config.KG_MAX_TRIPLES_PER_CHUNK]
        except Exception as exc:
            last_error = exc
            logger.warning(
                "KG extraction failed for %s (attempt %d/3): %s",
                chunk_id,
                attempt,
                exc,
            )
            if (
                "reasoning-only output" in str(exc)
                and model != fallback_model
                and fallback_model
            ):
                logger.info(
                    "KG extraction fallback model switch for %s: %s -> %s",
                    chunk_id,
                    model,
                    fallback_model,
                )
                model = fallback_model
            if use_response_format and _supports_response_format(exc):
                use_response_format = False
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))

    logger.warning("KG extraction failed for %s after retries: %s", chunk_id, last_error)
    return []
