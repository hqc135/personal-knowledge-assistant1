"""KG 抽取诊断脚本

用法: python scripts/diagnose_kg.py [notes_dir]
会对目录中的所有 .md 文件分块并对每个 chunk 调用一次 LLM，记录失败类型和返回元信息。
"""
from pathlib import Path
import json
import sys
import time
from collections import Counter

# Ensure project root is on sys.path when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import kg_extractor


def diagnose(notes_dir: str):
    p = Path(notes_dir)
    files = list(p.rglob("*.md"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ""],
    )

    stats = Counter()
    examples = {}

    client = kg_extractor._create_client()

    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for i, ch in enumerate(chunks):
            chunk_id = f"{f.name}_{i}"
            prompt = kg_extractor._PROMPT.format(text=ch, max_triples=config.KG_MAX_TRIPLES_PER_CHUNK)
            try:
                kwargs = kg_extractor._request_kwargs(model=config.KG_LLM_MODEL, use_response_format=True, prompt=prompt)
                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                content = msg.content or ""
                if not content:
                    reason = resp.choices[0].finish_reason or "empty"
                    if getattr(msg, "reasoning_content", None):
                        key = f"reasoning_only|{reason}"
                    else:
                        key = f"empty_content|{reason}"
                    stats[key] += 1
                    examples.setdefault(key, []).append({"file": str(f), "chunk": i, "finish_reason": reason, "reasoning": getattr(msg, "reasoning_content", None)})
                    continue

                # try parse
                try:
                    data = kg_extractor._extract_json(content)
                except Exception as e:
                    stats["json_parse_error"] += 1
                    examples.setdefault("json_parse_error", []).append({"file": str(f), "chunk": i, "error": str(e), "content_sample": content[:200]})
                    continue

                # structural checks
                if isinstance(data, dict) and "triples" in data:
                    data = data["triples"]
                if not isinstance(data, list):
                    stats["non_list_payload"] += 1
                    examples.setdefault("non_list_payload", []).append({"file": str(f), "chunk": i, "sample": str(data)[:200]})
                    continue

                if not data:
                    stats["empty_triples_list"] += 1
                    examples.setdefault("empty_triples_list", []).append({"file": str(f), "chunk": i})
                    continue

                # success
                stats["success"] += 1
            except Exception as e:
                stats[type(e).__name__] += 1
                examples.setdefault(type(e).__name__, []).append({"file": str(f), "chunk": i, "error": str(e)})
            time.sleep(0.5)

    out = {
        "summary": dict(stats),
        "examples": {k: v[:5] for k, v in examples.items()},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    notes_dir = sys.argv[1] if len(sys.argv) > 1 else "notes"
    diagnose(notes_dir)
