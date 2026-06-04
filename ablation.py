"""
消融实验：对比不同检索策略的效果，并利用线程池实现并行实验。

对比维度：
  - Base: Vector Only
  - Base + Hybrid
  - Base + Hybrid + Reranker
  - Base + Hybrid + Reranker + KG
  - Base + Hybrid + Reranker + KG + Query Aligner
  - Base + Hybrid + Reranker + KG + Query Aligner + Query Rewriter
  - Full Pipeline: 上述所有 + Intent Router
  - Full Pipeline w/o Intent Router
  - Full Pipeline w/o KG
  - Full Pipeline w/o Query Aligner
  - Full Pipeline w/o Query Rewriter

输出：对比表格 + JSON 详细报告
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Optional

from logger import setup_logging
from retriever import Retriever
from generator import Generator
from evaluator import run_eval
import config

logger = logging.getLogger(__name__)

# ── 实验配置 ──────────────────────────────────────────────

class Experiment(TypedDict):
    name: str
    mode: str
    rerank: Optional[bool]
    use_intent_router: bool
    use_kg: bool
    use_query_aligner: bool
    use_query_rewriter: bool
    use_hybrid: bool

EXPERIMENTS: list[Experiment] = [
    {
        "name": "Base (Vector Only)",
        "mode": "vector",
        "rerank": False,
        "use_intent_router": False,
        "use_kg": False,
        "use_query_aligner": False,
        "use_query_rewriter": False,
        "use_hybrid": False,
    },
    {
        "name": "Base + Hybrid",
        "mode": "hybrid",
        "rerank": False,
        "use_intent_router": False,
        "use_kg": False,
        "use_query_aligner": False,
        "use_query_rewriter": False,
        "use_hybrid": True,
    },
    {
        "name": "Base + Hybrid + Reranker",
        "mode": "hybrid",
        "rerank": True,
        "use_intent_router": False,
        "use_kg": False,
        "use_query_aligner": False,
        "use_query_rewriter": False,
        "use_hybrid": True,
    },
    {
        "name": "Base + Hybrid + Reranker + KG",
        "mode": "hybrid",
        "rerank": True,
        "use_intent_router": False,
        "use_kg": True,
        "use_query_aligner": False,
        "use_query_rewriter": False,
        "use_hybrid": True,
    },
    {
        "name": "Base + H + R + KG + Aligner",
        "mode": "hybrid",
        "rerank": True,
        "use_intent_router": False,
        "use_kg": True,
        "use_query_aligner": True,
        "use_query_rewriter": False,
        "use_hybrid": True,
    },
    {
        "name": "Base + H + R + KG + Aligner + Rewriter",
        "mode": "hybrid",
        "rerank": True,
        "use_intent_router": False,
        "use_kg": True,
        "use_query_aligner": True,
        "use_query_rewriter": True,
        "use_hybrid": True,
    },
    {
        "name": "Full Pipeline (Auto Mode)",
        "mode": "auto",
        "rerank": True,
        "use_intent_router": True,
        "use_kg": True,
        "use_query_aligner": True,
        "use_query_rewriter": True,
        "use_hybrid": True,
    },
    {
        "name": "Full w/o Intent Router",
        "mode": "hybrid",
        "rerank": True,
        "use_intent_router": False,
        "use_kg": True,
        "use_query_aligner": True,
        "use_query_rewriter": True,
        "use_hybrid": True,
    },
    {
        "name": "Full w/o KG",
        "mode": "auto",
        "rerank": True,
        "use_intent_router": True,
        "use_kg": False,
        "use_query_aligner": True,
        "use_query_rewriter": True,
        "use_hybrid": True,
    },
    {
        "name": "Full w/o Query Aligner",
        "mode": "auto",
        "rerank": True,
        "use_intent_router": True,
        "use_kg": True,
        "use_query_aligner": False,
        "use_query_rewriter": True,
        "use_hybrid": True,
    },
    {
        "name": "Full w/o Query Rewriter",
        "mode": "auto",
        "rerank": True,
        "use_intent_router": True,
        "use_kg": True,
        "use_query_aligner": True,
        "use_query_rewriter": False,
        "use_hybrid": True,
    },
]

def _run_single_experiment(exp: Experiment, cases: list[dict], use_llm_judge: bool, retriever: Retriever, generator: Generator) -> dict:
    """运行单个消融实验"""
    logger.info("=" * 50)
    logger.info("启动实验: %s", exp["name"])
    logger.info("=" * 50)

    try:
        result = run_eval(
            cases,
            retriever,
            generator,
            retrieve_mode=exp["mode"],
            use_rerank=exp["rerank"],
            use_llm_judge=use_llm_judge,
            use_hybrid=exp["use_hybrid"],
            use_kg=exp["use_kg"],
            use_intent_router=exp["use_intent_router"],
            use_query_aligner=exp["use_query_aligner"],
            use_query_rewriter=exp["use_query_rewriter"]
        )
    except Exception as e:
        logger.error("实验 %s 运行失败: %s", exp["name"], e)
        result = {"error": str(e), "emb_relevance_avg": 0, "emb_faithfulness_avg": 0}
        
    result["experiment"] = exp["name"]
    return result

def run_ablation(
    eval_cases_path: str = "eval_cases.json",
    use_llm_judge: bool = True,
    max_workers: int = 3
):
    """运行并行的消融实验，对比所有检索策略"""
    with open(eval_cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    logger.info("加载 %d 个测试用例", len(cases))
    logger.info("LLM Judge: %s", "开启" if use_llm_judge else "关闭")
    logger.info("并发 Worker 数: %d", max_workers)

    # 全局初始化一次重型组件，避免多线程加载大模型导致显存溢出或死锁
    logger.info("正在初始化全局 Retriever 和 Generator...")
    global_retriever = Retriever(
        use_reranker=True,
        use_hybrid=True,
        use_kg=True,
        use_intent_router=True,
        use_query_aligner=True,
        use_query_rewriter=True
    )
    global_generator = Generator()
    logger.info("全局组件初始化完成。")

    all_results = []

    # 使用 ThreadPoolExecutor 并行运行实验
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(_run_single_experiment, exp, cases, use_llm_judge, global_retriever, global_generator): exp
            for exp in EXPERIMENTS
        }

        for future in as_completed(future_to_exp):
            exp = future_to_exp[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                logger.error("实验 %s 产生异常: %s", exp["name"], exc)
                all_results.append({"experiment": exp["name"], "error": str(exc), "emb_relevance_avg": 0, "emb_faithfulness_avg": 0})

    # 按原始 EXPERIMENTS 顺序排序
    exp_order = {exp["name"]: i for i, exp in enumerate(EXPERIMENTS)}
    all_results.sort(key=lambda r: exp_order.get(r["experiment"], 999))

    # ── 打印对比表格 ──
    _print_comparison(all_results, use_llm_judge)

    # ── 保存详细报告 ──
    report_path = _save_report(all_results)
    logger.info("详细报告已保存: %s", report_path)


def _print_comparison(results: list[dict], use_llm_judge: bool):
    """打印对比表格"""
    print("\n" + "=" * 110)
    print("📊 消融实验结果对比")
    print("=" * 110)

    # 表头
    header = f"{'实验':<40} {'Emb Rel':>8} {'Emb Faith':>10}"
    if use_llm_judge:
        header += f" {'LLM Rel':>8} {'LLM Faith':>10} {'LLM Comp':>9}"
    print(header)
    print("-" * len(header))

    # 数据行
    best_metric = max((r.get("emb_relevance_avg", 0) for r in results), default=0)
    for r in results:
        is_best = r.get("emb_relevance_avg", 0) == best_metric and best_metric > 0
        marker = " ★" if is_best else ""

        if "error" in r and "emb_relevance_avg" not in r:
            print(f"{r['experiment']:<40} [ERROR] {r['error']}")
            continue
            
        line = (
            f"{r['experiment']:<40} "
            f"{r.get('emb_relevance_avg', 0):>8.4f} "
            f"{r.get('emb_faithfulness_avg', 0):>10.4f}"
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
