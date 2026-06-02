"""
消融实验：对比不同检索策略的效果

对比维度：
  - Vector Only (无 Reranker)
  - Vector + Reranker
  - BM25 Only
  - Hybrid (Vector + BM25 RRF 融合)
  - Hybrid + Reranker

输出：对比表格 + JSON 详细报告
"""
import json
import logging
from pathlib import Path
from datetime import datetime

from logger import setup_logging
from retriever import Retriever
from generator import Generator
from evaluator import run_eval
from typing import TypedDict, Optional
import config

logger = logging.getLogger(__name__)

# ── 实验配置 ──────────────────────────────────────────────

class Experiment(TypedDict):
    name: str
    mode: str
    rerank: Optional[bool]
    use_intent_router: Optional[bool]
    use_kg: Optional[bool]
    neighbor_window: Optional[int]
    neighbor_budget: Optional[int]
    global_mode: Optional[str]

EXPERIMENTS: list[Experiment] = [
    {
        "name": "Auto (Main Pipeline)",
        "mode": "auto",
        "rerank": None,
        "use_intent_router": True,
        "use_kg": None,
        "neighbor_window": None,
        "neighbor_budget": None,
        "global_mode": None,
    },
    {
        "name": "Auto (No Intent Router)",
        "mode": "auto",
        "rerank": None,
        "use_intent_router": False,
        "use_kg": False,
        "neighbor_window": 0,
        "neighbor_budget": 0,
        "global_mode": None,
    },
    {
        "name": "Vector Only",
        "mode": "vector",
        "rerank": False,
        "use_intent_router": None,
        "use_kg": False,
        "neighbor_window": 0,
        "neighbor_budget": 0,
        "global_mode": None,
    },
    {
        "name": "Vector + Reranker",
        "mode": "vector",
        "rerank": True,
        "use_intent_router": None,
        "use_kg": False,
        "neighbor_window": 0,
        "neighbor_budget": 0,
        "global_mode": None,
    },
    {
        "name": "BM25 Only",
        "mode": "bm25",
        "rerank": False,
        "use_intent_router": None,
        "use_kg": False,
        "neighbor_window": 0,
        "neighbor_budget": 0,
        "global_mode": None,
    },
    {
        "name": "Hybrid (RRF)",
        "mode": "hybrid",
        "rerank": False,
        "use_intent_router": None,
        "use_kg": False,
        "neighbor_window": 0,
        "neighbor_budget": 0,
        "global_mode": None,
    },
    {
        "name": "Hybrid + Reranker",
        "mode": "hybrid",
        "rerank": True,
        "use_intent_router": None,
        "use_kg": False,
        "neighbor_window": 0,
        "neighbor_budget": 0,
        "global_mode": None,
    },
]


def _apply_overrides(exp: Experiment) -> dict[str, object]:
    overrides: dict[str, object] = {
        "USE_INTENT_ROUTER": exp["use_intent_router"],
        "USE_KG_RETRIEVAL": exp["use_kg"],
        "RETRIEVER_NEIGHBOR_WINDOW": exp["neighbor_window"],
        "RETRIEVER_NEIGHBOR_BUDGET": exp["neighbor_budget"],
        "INTENT_ROUTER_GLOBAL_MODE": exp["global_mode"],
    }

    prev: dict[str, object] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        prev[key] = getattr(config, key)
        setattr(config, key, value)
    return prev


def _restore_overrides(prev: dict[str, object]) -> None:
    for key, value in prev.items():
        setattr(config, key, value)


def run_ablation(
    eval_cases_path: str = "eval_cases.json",
    use_llm_judge: bool = True,
):
    """运行消融实验，对比所有检索策略"""
    with open(eval_cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    logger.info("加载 %d 个测试用例", len(cases))
    logger.info("LLM Judge: %s", "开启" if use_llm_judge else "关闭")

    retriever = Retriever()
    generator = Generator()

    all_results = []

    for exp in EXPERIMENTS:
        logger.info("=" * 50)
        logger.info("实验: %s", exp["name"])
        logger.info("=" * 50)

        prev = _apply_overrides(exp)
        if prev:
            logger.info("实验覆盖配置: %s", {k: getattr(config, k) for k in prev})

        try:
            retriever = Retriever()
            generator = Generator()
            result = run_eval(
                cases,
                retriever,
                generator,
                retrieve_mode=exp["mode"],
                use_rerank=exp["rerank"],
                use_llm_judge=use_llm_judge,
            )
        finally:
            _restore_overrides(prev)
        result["experiment"] = exp["name"]
        all_results.append(result)

    # ── 打印对比表格 ──
    _print_comparison(all_results, use_llm_judge)

    # ── 保存详细报告 ──
    report_path = _save_report(all_results)
    logger.info("详细报告已保存: %s", report_path)


def _print_comparison(results: list[dict], use_llm_judge: bool):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("📊 消融实验结果对比")
    print("=" * 80)

    # 表头
    header = f"{'实验':<22} {'Emb Rel':>8} {'Emb Faith':>10}"
    if use_llm_judge:
        header += f" {'LLM Rel':>8} {'LLM Faith':>10} {'LLM Comp':>9}"
    print(header)
    print("-" * len(header))

    # 数据行
    best_metric = max(r["emb_relevance_avg"] for r in results)
    for r in results:
        is_best = r["emb_relevance_avg"] == best_metric
        marker = " ★" if is_best else ""

        line = (
            f"{r['experiment']:<22} "
            f"{r['emb_relevance_avg']:>8.4f} "
            f"{r['emb_faithfulness_avg']:>10.4f}"
        )
        if use_llm_judge and "llm_relevance_avg" in r:
            line += (
                f" {r['llm_relevance_avg']:>8.1f} "
                f"{r['llm_faithfulness_avg']:>10.1f} "
                f"{r['llm_completeness_avg']:>9.1f}"
            )
        print(line + marker)

    print("=" * len(header))
    print("★ = 最佳 Embedding 检索相关性\n")


def _save_report(results: list[dict]) -> str:
    """保存 JSON 详细报告"""
    report_dir = Path("eval_reports")
    report_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"ablation_{timestamp}.json"
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


if __name__ == "__main__":
    setup_logging()
    run_ablation()
