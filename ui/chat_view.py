"""
聊天视图：负责聊天 Tab 的 UI 布局和响应内容格式化。

从 app.py 解耦，职责单一：
- 格式化 assistant 回复（参考来源块 + Trace 折叠块）
- 构建 Gradio 聊天 Tab 布局
- chat() 函数调用 pipeline 并做 UI 适配（流式 yield）
"""
from __future__ import annotations

import logging
import time

import gradio as gr

from core.schemas import PipelineTrace
from metrics import Timer, RequestMetrics, collector
import config

logger = logging.getLogger(__name__)

# 流式输出：每累积 N 个字符才 yield 一次，减少 UI 重绘
_STREAM_BATCH = 10


# ── 响应格式化 ─────────────────────────────────────────────


def build_pipeline_status(trace: PipelineTrace) -> str:
    """将 PipelineTrace 转换为人类可读的管线状态描述。"""
    route_decision = trace.route_decision.lower()
    align_triggered = trace.align_triggered
    rerank_active = trace.rerank_active

    if route_decision == "global":
        segments = ["Global"]
        if align_triggered:
            segments.append("🌐 语义对齐")
        segments.append("📝 摘要合成")
        return " ｜ ".join(segments)

    segments = ["Local" if route_decision == "local" else route_decision.title()]
    mode = trace.mode.lower()
    if mode == "hybrid":
        segments.append("🔀 异构召回")
    elif mode == "vector":
        segments.append("🔭 向量召回")
    elif mode == "bm25":
        segments.append("🔤 关键词召回")

    if align_triggered:
        segments.append("🌐 语义对齐")
    if rerank_active:
        segments.append("⚖️ 精排介入")

    return " ｜ ".join(segments)


def build_source_block(contexts: list[dict]) -> str:
    """生成参考来源 Markdown 块。"""
    source_entries: dict[str, tuple[str, float]] = {}

    for item in contexts:
        metadata = item.get("metadata") or {}
        score = float(item.get("score", 0.0) or 0.0)
        score_type = str(metadata.get("score_type") or "unknown")
        raw_sources = metadata.get("sources")
        if isinstance(raw_sources, list) and raw_sources:
            source_names = [str(s).strip() for s in raw_sources if str(s).strip()]
        else:
            source_name = str(metadata.get("source") or "").strip()
            source_names = [source_name] if source_name else []

        for source_name in source_names:
            existing = source_entries.get(source_name)
            if existing is None or score > existing[1]:
                source_entries[source_name] = (score_type, score)

    lines = ["---\n📎 **参考来源**"]
    if not source_entries:
        lines.append("- 暂无可展示来源")
    else:
        for source_name, (score_type, score) in source_entries.items():
            lines.append(
                f'- {source_name} <span style="color:#6c7086;font-size:0.82em">[{score_type}] {score:.4f}</span>'
            )
    return "\n".join(lines)


def build_trace_block(trace: PipelineTrace, history_turns: int = 0) -> str:
    """生成可折叠的 Trace 执行详情 HTML 块。"""
    route_label = {
        "global": "Global",
        "local": "Local",
    }.get(trace.route_decision.lower(), trace.route_decision.title() or "Unknown")

    expansions: list[str] = []
    if isinstance(trace.query_alignment, dict):
        raw_expansions = trace.query_alignment.get("expansions") or []
        if isinstance(raw_expansions, list):
            expansions = [str(e).strip() for e in raw_expansions if str(e).strip()]
    expansion_text = "，".join(expansions) if expansions else "无"

    aligned_query_line = (
        f"<div><strong>对齐后查询</strong>: {trace.aligned_query}</div>"
        if trace.aligned_query
        else ""
    )
    history_line = (
        f"<div><strong>历史轮数</strong>: {history_turns} 轮已注入</div>"
        if history_turns > 0
        else "<div><strong>历史轮数</strong>: 未注入（首轮或已禁用）</div>"
    )
    rewrite_line = (
        f"<div><strong>多轮重写</strong>: {'✅ 已触发' if trace.rewrite_triggered else '⏭️ 未触发'}</div>"
        f"{(f'<div><strong>重写前查询</strong>: {trace.original_query}</div>' if trace.rewrite_triggered else '')}"
        f"{(f'<div><strong>重写后查询</strong>: {trace.rewritten_query}</div>' if trace.rewrite_triggered else '')}"
    )

    return (
        "<details><summary>🔍 检索管线执行详情 (Trace)</summary>"
        "<div style=\"margin-top:0.6em;padding:0 4px\">"
        f"{history_line}"
        f"{rewrite_line}"
        f"<div><strong>意图路由</strong>: {route_label}</div>"
        f"<div><strong>前置对齐</strong>: {'✅ 已触发' if trace.align_triggered else '⏭️ 未触发'}</div>"
        f"<div><strong>对齐扩展词</strong>: {expansion_text}</div>"
        f"{aligned_query_line}"
        f"<div><strong>精排介入</strong>: {'✅ 已触发' if trace.rerank_active else '⏭️ 未触发'}</div>"
        f"<div><strong>重写耗时</strong>: {trace.rewrite_ms:.1f} ms</div>"
        f"<div><strong>对齐耗时</strong>: {trace.align_ms:.1f} ms</div>"
        f"<div><strong>精排耗时</strong>: {trace.rerank_ms:.1f} ms</div>"
        f"<div><strong>纯检索耗时</strong>: {trace.pure_search_ms:.1f} ms</div>"
        "</div></details>"
    )


def build_final_assistant_content(
    answer: str,
    contexts: list[dict],
    trace: PipelineTrace,
    history_turns: int = 0,
) -> str:
    """拼接 answer + 参考来源 + Trace 为最终 assistant 消息内容。"""
    sections = [answer]
    sections.append(build_source_block(contexts))
    sections.append(build_trace_block(trace, history_turns))
    return "\n\n".join(section for section in sections if section)


# ── 聊天回调 ─────────────────────────────────────────────


def make_chat_fn(pipeline):
    """
    工厂：返回绑定了 pipeline 的 chat() 生成器函数。

    将 Gradio 回调与具体 pipeline 实例解耦，便于测试替换。
    """
    def chat(message: str, history: list | None):
        """流式聊天 + 指标采集（批量 yield 减少 UI 重绘）。"""
        t_start = time.perf_counter()
        history = history or []
        base_history = history + [{"role": "user", "content": message}]

        # 检索
        try:
            with Timer() as t_retrieve:
                result = pipeline.retriever.retrieve_with_trace(
                    message, history=history
                )
            contexts = result.contexts
            trace = result.trace
        except Exception as e:
            logger.error("检索失败: %s", e)
            empty_trace = PipelineTrace()
            error_content = build_final_assistant_content(
                f"⚠️ 检索出错: {e}", [], empty_trace
            )
            yield base_history + [{"role": "assistant", "content": error_content}]
            return

        # 计算实际注入的历史轮数
        _valid_history = [
            m for m in history
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]
        history_turns_injected = min(
            len(_valid_history) // 2, config.LLM_HISTORY_TURNS
        )

        # 流式生成
        answer = ""
        char_buf = 0
        generate_error: Exception | None = None
        generate_ms = 0.0
        try:
            with Timer() as t_gen:
                for token in pipeline.generator.generate_stream(
                    message, contexts, history=history
                ):
                    answer += token
                    char_buf += len(token)
                    if char_buf >= _STREAM_BATCH:
                        char_buf = 0
                        yield base_history + [{"role": "assistant", "content": answer}]
            generate_ms = t_gen.elapsed_ms
        except Exception as e:
            logger.error("生成失败: %s", e)
            generate_error = e
            generate_ms = t_gen.elapsed_ms if "t_gen" in locals() else 0.0

        # 计算 pure_search_ms 并回写 trace
        pure_search_ms = max(
            t_retrieve.elapsed_ms - trace.rerank_ms - trace.align_ms, 0.0
        )
        trace.pure_search_ms = pure_search_ms
        pipeline_status = build_pipeline_status(trace)

        # 指标记录
        total_ms = (time.perf_counter() - t_start) * 1000
        collector.record(RequestMetrics(
            timestamp=time.time(),
            query=message,
            embed_ms=trace.embed_ms,
            search_ms=pure_search_ms,
            align_ms=trace.align_ms,
            rerank_ms=trace.rerank_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
            num_sources=len(contexts),
            answer_length=len(answer),
            retrieve_mode=pipeline_status,
        ))

        # 最终回复
        if generate_error is not None:
            error_msg = (
                answer + f"\n\n⚠️ 生成中断: {generate_error}"
                if answer
                else f"⚠️ 生成出错: {generate_error}"
            )
            final_content = build_final_assistant_content(
                error_msg, contexts, trace, history_turns_injected
            )
        else:
            final_content = build_final_assistant_content(
                answer, contexts, trace, history_turns_injected
            )

        yield base_history + [{"role": "assistant", "content": final_content}]

    return chat


# ── Gradio Tab 布局 ───────────────────────────────────────


def build_chat_tab(pipeline) -> None:
    """构建聊天 Tab 的 Gradio 组件（在 gr.Tab 上下文内调用）。"""
    chat_fn = make_chat_fn(pipeline)

    chatbot = gr.Chatbot(
        label="对话",
        height=520,
        elem_id="chatbot-window",
        show_label=False,
        placeholder=(
            "<div style='text-align:center;color:#45475a;padding:60px 0'>"
            "<div style='font-size:2.5rem'>📚</div>"
            "<div style='font-size:1.1rem;margin-top:12px;color:#6c7086'>向我提问，探索你的知识库</div>"
            "</div>"
        ),
        render_markdown=True,
    )

    with gr.Row(equal_height=True):
        msg = gr.Textbox(
            label="",
            placeholder="💬  输入问题，按 Enter 发送…",
            scale=9,
            show_label=False,
            elem_id="msg-input",
            lines=1,
            max_lines=4,
            autofocus=True,
        )
        send_btn = gr.Button(
            "发送 ↑", variant="primary", scale=1, elem_id="send-btn", min_width=80
        )

    with gr.Row():
        gr.ClearButton(
            [msg, chatbot],
            value="🗑 清空对话",
            elem_id="clear-btn",
        )
        gr.Examples(
            examples=[
                "总结一下我关于 RAG 的笔记",
                "系统设计的关键原则是什么",
                "什么是注意力机制",
            ],
            inputs=msg,
            label="示例问题",
        )

    msg.submit(chat_fn, [msg, chatbot], [chatbot])
    send_btn.click(chat_fn, [msg, chatbot], [chatbot])
