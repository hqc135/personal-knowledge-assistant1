"""
评估框架：
  - EmbeddingEvaluator: 基于 embedding 相似度的快速评估
  - LLMJudgeEvaluator: 基于 LLM 打分的深度评估 (检索相关性/回答忠实度/回答完整度)
  - run_eval: 统一评估入口，支持模式/rerank 参数覆盖 (用于消融实验)
"""
import json
import logging
import re
import time

import numpy as np
from openai import OpenAI

from embedder import ZhipuEmbedder
import config
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ── LLM-as-Judge 评分 Prompt ─────────────────────────────

_JUDGE_PROMPT = """\
你是一个 RAG 系统评估专家。请根据以下信息，分三个维度打分 (1-5 分)。

## 用户问题
{query}

## 检索到的上下文
{contexts}

## 系统回答
{answer}

## 评分维度
1. **检索相关性** (retrieval_relevance): 检索到的文档与问题的相关程度
   - 5分: 所有文档高度相关  1分: 完全不相关
2. **回答忠实度** (faithfulness): 回答是否基于上下文，有无编造
   - 5分: 完全基于上下文  1分: 大量编造
3. **回答完整度** (completeness): 回答是否充分回答了问题
   - 5分: 全面完整  1分: 几乎没回答

请严格以如下 JSON 格式输出 (不要添加其他内容)，reason 字段不超过 50 个字：
{{"retrieval_relevance": <int>, "faithfulness": <int>, "completeness": <int>, "reason": "<简短理由>"}}
"""


class EmbeddingEvaluator:
    """基于 Embedding 相似度的快速评估"""

    def __init__(self):
        self.embedder = ZhipuEmbedder()

    def relevance_score(self, query: str, retrieved_texts: list[str]) -> float:
        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        doc_embs = self.embedder.encode(retrieved_texts, normalize_embeddings=True)
        scores = np.dot(doc_embs, query_emb)
        return float(np.mean(scores))

    def faithfulness_score(self, answer: str, contexts: list[str]) -> float:
        answer_emb = self.embedder.encode(answer, normalize_embeddings=True)
        ctx_embs = self.embedder.encode(contexts, normalize_embeddings=True)
        return float(np.max(np.dot(ctx_embs, answer_emb)))


class LLMJudgeEvaluator:
    """基于 LLM (DeepSeek) 打分的深度评估"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.LLM_BASE_URL,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从模型输出中提取 JSON 对象，增强容错：支持代码块、BOM、控制字符、截断 JSON。"""
        # 去除 BOM 和首尾空白
        content = text.strip().lstrip("\ufeff")

        # 去除非法控制字符（除 \t \n \r 外），避免 JSON 解析崩溃
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)

        # 处理 markdown 代码块包裹
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                block = parts[1].strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                content = block

        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 提取第一个 {...} 块
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # 如果 JSON 被截断（末尾缺少 "}"），尝试补齐后再解析
                if not candidate.rstrip().endswith("}"):
                    try:
                        return json.loads(candidate.rstrip().rstrip(",") + "}")
                    except json.JSONDecodeError:
                        pass

        raise ValueError(f"无法从输出中提取合法 JSON: {text[:200]!r}")

    def judge(self, query: str, contexts: list[str], answer: str) -> dict:
        # 截断过长的 context 和 answer，防止 judge 请求超出 token 窗口导致空响应
        max_ctx = config.EVAL_JUDGE_MAX_CTX_CHARS
        max_ans = config.EVAL_JUDGE_MAX_ANS_CHARS
        ctx_text = "\n---\n".join(contexts)
        if len(ctx_text) > max_ctx:
            ctx_text = ctx_text[:max_ctx] + "\n...[上下文已截断]"
        if len(answer) > max_ans:
            answer = answer[:max_ans] + "...[回答已截断]"

        prompt = _JUDGE_PROMPT.format(query=query, contexts=ctx_text, answer=answer)
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512,
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("LLM Judge 返回了空响应（可能超出 token 或 rate limit）")
                return self._extract_json(content)
            except Exception as e:
                last_error = e
                logger.warning("LLM Judge 解析失败 (第 %d/3 次): %s", attempt, e)
                if attempt < 3:
                    time.sleep(1.0 * attempt)

        return {
            "retrieval_relevance": None,
            "faithfulness": None,
            "completeness": None,
            "reason": f"评估失败: {last_error}",
        }


def run_eval(
    test_cases: list[dict],
    retriever,
    generator,
    retrieve_mode: str = "auto",
    use_rerank: Optional[bool] = None,
    use_llm_judge: bool = True,
    **kwargs
) -> dict:
    """
    统一评估入口。

    支持 retrieve_mode / use_rerank 覆盖，用于消融实验对比不同策略。
    use_llm_judge=True 时启用 LLM-as-Judge 深度评估。
    """
    emb_eval = EmbeddingEvaluator()
    llm_eval = LLMJudgeEvaluator() if use_llm_judge else None

    results = []
    for i, case in enumerate(test_cases):
        query = case["query"]
        expected_keywords = case.get("expected_keywords", [])
        logger.info("评估 [%d/%d] mode=%s rerank=%s: %s",
                     i + 1, len(test_cases), retrieve_mode, use_rerank, query)

        try:
            contexts = retriever.retrieve(
                query, mode=retrieve_mode, use_rerank=use_rerank, **kwargs
            )
            answer = generator.generate(query, contexts)
            ctx_texts = [c["text"] for c in contexts]
            keyword_coverage = _keyword_coverage(query, ctx_texts, answer, expected_keywords)

            # Embedding 评分
            emb_relevance = emb_eval.relevance_score(query, ctx_texts)
            emb_faithfulness = emb_eval.faithfulness_score(answer, ctx_texts)

            entry = {
                "query": query,
                "emb_relevance": round(emb_relevance, 4),
                "emb_faithfulness": round(emb_faithfulness, 4),
                "answer_length": len(answer),
                "keyword_coverage": round(keyword_coverage, 4),
            }

            # LLM Judge 评分
            if llm_eval:
                judge = llm_eval.judge(query, ctx_texts, answer)
                entry["llm_relevance"] = judge.get("retrieval_relevance", 0)
                entry["llm_faithfulness"] = judge.get("faithfulness", 0)
                entry["llm_completeness"] = judge.get("completeness", 0)
                entry["llm_reason"] = judge.get("reason", "")

            results.append(entry)

        except Exception as e:
            logger.error("评估失败 [%s]: %s", query, e)
            results.append({"query": query, "error": str(e)})

    # ── 汇总 ──
    valid = [r for r in results if "error" not in r]
    summary: dict[str, Any] = {
        "retrieve_mode": retrieve_mode,
        "use_rerank": use_rerank,
        "num_cases": len(test_cases),
        "num_success": len(valid),
        "emb_relevance_avg": _avg(valid, "emb_relevance"),
        "emb_faithfulness_avg": _avg(valid, "emb_faithfulness"),
        "keyword_coverage_avg": _avg(valid, "keyword_coverage"),
    }

    if use_llm_judge and valid and "llm_relevance" in valid[0]:
        summary["llm_relevance_avg"] = _avg(valid, "llm_relevance")
        summary["llm_faithfulness_avg"] = _avg(valid, "llm_faithfulness")
        summary["llm_completeness_avg"] = _avg(valid, "llm_completeness")

    summary["details"] = results

    logger.info(
        "评估完成 [mode=%s rerank=%s]: emb_rel=%.3f emb_faith=%.3f",
        retrieve_mode, use_rerank,
        summary["emb_relevance_avg"], summary["emb_faithfulness_avg"],
    )
    return summary


def _avg(items: list[dict], key: str) -> float:
    """计算均值，跳过 None 值（技术失败不计入统计，避免虚假拉低分数）"""
    vals = [r[key] for r in items if key in r and r[key] is not None]
    return round(float(np.mean(vals)), 4) if vals else 0.0


def _keyword_coverage(query: str, contexts: list[str], answer: str, expected_keywords: list[str]) -> float:
    """计算预期关键词覆盖率：关键词只要出现在上下文或回答中即算命中。"""
    if not expected_keywords:
        return 0.0

    haystack = "\n".join([query, answer, *contexts]).lower()
    hits = 0
    for keyword in expected_keywords:
        if str(keyword).lower() in haystack:
            hits += 1
    return hits / len(expected_keywords)
