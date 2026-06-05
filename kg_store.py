"""Knowledge graph triple storage using local JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import config

logger = logging.getLogger(__name__)


def _path() -> Path:
    return Path(config.KG_TRIPLES_PATH)


def _normalize(value: str) -> str:
    return value.strip().lower()


def _triple_key(triple: dict) -> tuple[str, str, str, str, str]:
    return (
        _normalize(triple.get("head", "")),
        _normalize(triple.get("relation", "")),
        _normalize(triple.get("tail", "")),
        _normalize(triple.get("source", "")),
        _normalize(triple.get("chunk_id", "")),
    )


def load_triples() -> list[dict]:
    path = _path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load KG triples: %s", exc)
        return []


def dedupe(triples: Iterable[dict]) -> list[dict]:
    seen: set[tuple[str, str, str, str, str]] = set()
    output: list[dict] = []
    for triple in triples:
        key = _triple_key(triple)
        if key in seen:
            continue
        seen.add(key)
        output.append(triple)
    return output


def append_triples(triples: Iterable[dict]) -> int:
    new_triples = [t for t in triples if t]
    if not new_triples:
        return 0

    path = _path()
    existing = load_triples()
    merged = dedupe([*existing, *new_triples])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(merged) - len(existing)
