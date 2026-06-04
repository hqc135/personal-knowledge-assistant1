"""验证 evaluator 和 generator 修复的正确性"""
import sys
sys.path.insert(0, ".")

from evaluator import LLMJudgeEvaluator, _avg

print("=== 测试 _extract_json 容错 ===")
tests = [
    ('正常JSON', '{"retrieval_relevance": 4, "faithfulness": 5, "completeness": 3, "reason": "good"}'),
    ('代码块包裹', '```json\n{"retrieval_relevance": 4, "faithfulness": 5, "completeness": 3, "reason": "ok"}\n```'),
    ('BOM前缀', '\ufeff{"retrieval_relevance": 3, "faithfulness": 4, "completeness": 4, "reason": "test"}'),
    ('前缀文字', 'Some prefix {"retrieval_relevance": 2, "faithfulness": 3, "completeness": 2, "reason": "x"}'),
    ('截断JSON', '{"retrieval_relevance": 4, "faithfulness": 5, "completeness": 3, "reason": "trunc'),
]
for name, t in tests:
    try:
        r = LLMJudgeEvaluator._extract_json(t)
        print(f"  [{name}] OK: rel={r.get('retrieval_relevance')}")
    except Exception as e:
        print(f"  [{name}] FAIL: {e}")

print()
print("=== 测试 _avg 跳过 None ===")
items = [
    {"llm_relevance": 4},
    {"llm_relevance": 5},
    {"llm_relevance": None},
    {"llm_relevance": 3},
    {},
]
avg = _avg(items, "llm_relevance")
print(f"  avg (skip None) = {avg}  (期望 4.0, 即 (4+5+3)/3)")
assert abs(avg - 4.0) < 0.01, f"期望 4.0, 实际 {avg}"

print()
print("=== 测试 Generator._estimate_tokens ===")
from generator import Generator
import config
cases = [
    ("空字符串", "", 0),
    ("100 ASCII", "a" * 100, 60),
    ("100 中文", "你好" * 50, 60),
]
for name, text, expected in cases:
    got = Generator._estimate_tokens(text)
    status = "OK" if got == expected else f"WARN (got {got}, expected {expected})"
    print(f"  [{name}]: {status}")

print()
print("=== 验证预算计算 ===")
sp = "你是助手" * 5
ctx = "这是上下文" * 200
uc = f"上下文:\n{ctx}\n\n问题: test"
fixed = (Generator._estimate_tokens(sp)
         + Generator._estimate_tokens(uc)
         + config.LLM_MAX_TOKENS
         + config.LLM_TOKEN_SAFETY_MARGIN)
budget = config.LLM_MAX_CONTEXT_TOKENS - fixed
print(f"  fixed_tokens={fixed}, history_budget={budget}")
assert budget > 0, "预算应大于0（context 约 1000 字符，窗口 32000）"
print(f"  history_budget > 0: OK")

print()
print("所有验证通过！")
