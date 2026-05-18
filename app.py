"""
Gradio 应用入口：聊天 + 可观测性仪表盘
"""
import logging
import time

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from retriever import Retriever
from generator import Generator
from metrics import Timer, RequestMetrics, collector
from logger import setup_logging

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ── 初始化 ───────────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)

retriever = Retriever()
generator = Generator()

# ── 聊天逻辑 ─────────────────────────────────────────────


def chat(message: str, history: list):
    """流式聊天 + 指标采集"""
    t_start = time.perf_counter()

    # 检索
    try:
        with Timer() as t_retrieve:
            contexts = retriever.retrieve(message)
    except Exception as e:
        logger.error("检索失败: %s", e)
        yield history + [[message, f"⚠️ 检索出错: {e}"]]
        return

    # 来源信息
    sources = "\n".join(
        f"- {c['metadata']['source']} (score: {c['score']:.3f})"
        for c in contexts
    )
    source_block = f"\n\n---\n📎 参考来源:\n{sources}"

    # 流式生成
    answer = ""
    try:
        with Timer() as t_gen:
            for token in generator.generate_stream(message, contexts):
                answer += token
                yield history + [[message, answer + source_block]]
    except Exception as e:
        logger.error("生成失败: %s", e)
        error_msg = answer + f"\n\n⚠️ 生成中断: {e}" if answer else f"⚠️ 生成出错: {e}"
        yield history + [[message, error_msg + source_block]]
        return

    # 记录指标
    total_ms = (time.perf_counter() - t_start) * 1000
    collector.record(RequestMetrics(
        timestamp=time.time(),
        query=message,
        embed_ms=retriever.last_timing.get("embed_ms", 0),
        search_ms=t_retrieve.elapsed_ms,
        rerank_ms=retriever.last_timing.get("rerank_ms", 0),
        generate_ms=t_gen.elapsed_ms,
        total_ms=total_ms,
        num_sources=len(contexts),
        answer_length=len(answer),
        retrieve_mode=retriever.last_timing.get("mode", "unknown"),
    ))

    yield history + [[message, answer + source_block]]


# ── 仪表盘逻辑 ───────────────────────────────────────────


def build_summary_text() -> str:
    """生成仪表盘汇总文本"""
    s = collector.summary()
    if s["total_requests"] == 0:
        return "暂无请求数据。发送几条消息后刷新查看。"

    return (
        f"### 📈 汇总统计\n\n"
        f"| 指标 | 值 |\n"
        f"|------|----|\n"
        f"| 总请求数 | **{s['total_requests']}** |\n"
        f"| 平均总耗时 | **{s['avg_total_ms']:.0f} ms** |\n"
        f"| P95 总耗时 | **{s['p95_total_ms']:.0f} ms** |\n"
        f"| 平均检索耗时 | {s['avg_search_ms']:.0f} ms |\n"
        f"| 平均 Rerank 耗时 | {s['avg_rerank_ms']:.0f} ms |\n"
        f"| 平均生成耗时 | {s['avg_generate_ms']:.0f} ms |\n"
    )


def build_latency_chart():
    """生成延迟分解柱状图"""
    s = collector.summary()
    if s["total_requests"] == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        return fig

    categories = ["Embedding\n+检索", "Rerank", "LLM 生成"]
    values = [s["avg_search_ms"], s["avg_rerank_ms"], s["avg_generate_ms"]]
    colors = ["#4ECDC4", "#FFE66D", "#FF6B6B"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("平均耗时 (ms)")
    ax.set_title("各阶段延迟分解")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def build_trend_chart():
    """生成请求耗时趋势图"""
    h = collector.history
    if len(h) < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "至少需要 2 条请求", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        return fig

    x = list(range(1, len(h) + 1))
    total = [m.total_ms for m in h]
    gen = [m.generate_ms for m in h]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, total, "o-", color="#FF6B6B", label="总耗时", markersize=4)
    ax.plot(x, gen, "s--", color="#4ECDC4", label="LLM 生成", markersize=4)
    ax.set_xlabel("请求序号")
    ax.set_ylabel("耗时 (ms)")
    ax.set_title("请求耗时趋势")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def build_history_table():
    """生成最近请求记录表"""
    h = collector.history[-20:]  # 最近 20 条
    if not h:
        return []
    return [
        [
            m.query[:30] + ("..." if len(m.query) > 30 else ""),
            m.retrieve_mode,
            f"{m.search_ms:.0f}",
            f"{m.rerank_ms:.0f}",
            f"{m.generate_ms:.0f}",
            f"{m.total_ms:.0f}",
            m.num_sources,
        ]
        for m in reversed(h)
    ]


def refresh_dashboard():
    return (
        build_summary_text(),
        build_latency_chart(),
        build_trend_chart(),
        build_history_table(),
    )


# ── Gradio 界面 ──────────────────────────────────────────

with gr.Blocks(
    title="📚 个人知识库助手",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# 📚 个人知识库助手\n基于 RAG 混合检索 + 流式生成")

    with gr.Tab("💬 聊天"):
        chatbot = gr.Chatbot(label="对话", height=480)
        with gr.Row():
            msg = gr.Textbox(
                label="输入问题",
                placeholder="例如: 总结一下我关于 RAG 的笔记",
                scale=9,
                show_label=False,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ 清空")
            gr.Examples(
                examples=["总结一下我关于 RAG 的笔记", "系统设计的关键原则是什么"],
                inputs=msg,
            )

        msg.submit(chat, [msg, chatbot], [chatbot])
        send_btn.click(chat, [msg, chatbot], [chatbot])

    with gr.Tab("📊 仪表盘"):
        refresh_btn = gr.Button("🔄 刷新数据", variant="primary")

        summary_md = gr.Markdown("点击刷新查看数据")

        with gr.Row():
            latency_plot = gr.Plot(label="延迟分解")
            trend_plot = gr.Plot(label="耗时趋势")

        history_df = gr.Dataframe(
            headers=["问题", "模式", "检索ms", "Rerank ms", "生成ms", "总耗时ms", "来源数"],
            label="最近请求记录",
        )

        refresh_btn.click(
            refresh_dashboard,
            outputs=[summary_md, latency_plot, trend_plot, history_df],
        )


if __name__ == "__main__":
    demo.launch()
