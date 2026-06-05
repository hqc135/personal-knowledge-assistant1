"""Prompt A/B 测试脚本

用法: python scripts/ab_test_prompts.py [sample_size]

脚本会从 `notes/` 采样若干 chunk（默认 20），对每个 chunk 用两个 Prompt 变体调用模型，统计成功率与失败类型，结果保存到 `eval_reports/`。
"""
from pathlib import Path
import sys
import json
import time
from datetime import datetime
from collections import Counter, defaultdict

# ensure project root is importable when running script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import kg_extractor
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_prompts():
    base = kg_extractor._PROMPT

    def strict_template(text: str, max_triples: int):
        return (
            "You are an information extraction system.\n"
            f"Extract up to {max_triples} triples from the text. Each triple is (head, relation, tail).\n"
            "RETURN ONLY a single JSON value and NOTHING else. The JSON MUST be either:\n"
            "1) an array of objects: [{\"head\":..., \"relation\":..., \"tail\":...}, ...]\n"
            "OR 2) an object: {\"triples\": [ ... ]}.\n"
            "Fields must be exactly 'head','relation','tail' (strings). Do not add commentary, markdown, or code fences.\n"
            "Example output:\n[{\"head\": \"LangGraph\", \"relation\": \"models\", \"tail\": \"Agent\"}]\n\n"
            f"Text:\n{text}"
        )

    return lambda text, max_triples: base.format(text=text, max_triples=max_triples), strict_template


def sample_chunks(notes_dir: Path, max_chunks: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ""],
    )
    chunks = []
    for f in notes_dir.rglob("*.md"):
        text = f.read_text(encoding="utf-8")
        for i, ch in enumerate(splitter.split_text(text)):
            chunks.append((str(f), i, ch))
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def call_model(client, model, prompt, use_response_format=True):
    try:
        kwargs = kg_extractor._request_kwargs(model=model, use_response_format=use_response_format, prompt=prompt)
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        finish_reason = resp.choices[0].finish_reason
        reasoning = getattr(msg, "reasoning_content", None)
        return {"content": content, "finish_reason": finish_reason, "reasoning": reasoning}
    except Exception as e:
        return {"error": str(e)}


def analyze_response(resp):
    if "error" in resp:
        return "exception", resp["error"]
    content = resp.get("content", "")
    reasoning = resp.get("reasoning")
    if not content:
        if reasoning:
            return "reasoning_only", repr(reasoning)[:200]
        return "empty_content", ""
    try:
        data = kg_extractor._extract_json(content)
    except Exception as e:
        return "json_parse_error", str(e)
    if isinstance(data, dict) and "triples" in data:
        data = data["triples"]
    if not isinstance(data, list):
        return "non_list_payload", str(type(data))
    if not data:
        return "empty_triples_list", ""
    return "success", len(data)


def run_ab(sample_size: int = 20):
    notes_dir = Path("notes")
    chunks = sample_chunks(notes_dir, sample_size)
    if not chunks:
        print("no chunks found in notes/")
        return

    promptA_fn, promptB_fn = build_prompts()
    client = kg_extractor._create_client()
    model = config.KG_LLM_MODEL

    results = {"A": [], "B": []}
    stats = {"A": Counter(), "B": Counter()}

    for idx, (fname, ci, ch) in enumerate(chunks):
        for label in ("A", "B"):
            if label == "A":
                prompt = promptA_fn(ch, config.KG_MAX_TRIPLES_PER_CHUNK)
            else:
                prompt = promptB_fn(ch, config.KG_MAX_TRIPLES_PER_CHUNK)
            resp = call_model(client, model, prompt)
            outcome, info = analyze_response(resp)
            stats[label][outcome] += 1
            results[label].append({"file": fname, "chunk": ci, "outcome": outcome, "info": info})
            # small sleep to avoid burst
            time.sleep(0.6)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"sample_size": len(chunks), "stats": {k: dict(v) for k, v in stats.items()}, "results": results}
    Path("eval_reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("eval_reports") / f"prompt_ab_{now}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps(out["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_ab(n)
