"""
Gradio 应用入口：聊天 + 可观测性仪表盘
"""
import logging
import time

import gradio as gr

from retriever import Retriever
from generator import Generator
from metrics import Timer, RequestMetrics, collector
from logger import setup_logging
import config

# ── 初始化 ───────────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)

retriever = Retriever()
generator = Generator()

# 流式输出：每累积 N 个字符才 yield 一次，减少 UI 重绘
_STREAM_BATCH = 10

# ── 自定义 CSS ───────────────────────────────────────────

CUSTOM_CSS = """
/* ── 全局背景 ── */
body, .gradio-container {
    background: #11111b !important;
    font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif !important;
}

/* ── 顶部标题区 ── */
#app-header {
    background: linear-gradient(135deg, #1e1e2e 0%, #181825 60%, #1e1e2e 100%);
    border-bottom: 1px solid #313244;
    padding: 20px 32px 16px;
    margin-bottom: 4px;
}
#app-header h1 {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #cba6f7 0%, #89b4fa 50%, #94e2d5 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 4px 0 !important;
}
#app-header p {
    color: #6c7086 !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
}

/* ── Tab 样式 ── */
.tab-nav button {
    background: transparent !important;
    color: #6c7086 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    color: #cba6f7 !important;
    border-bottom-color: #cba6f7 !important;
    background: transparent !important;
}
.tab-nav button:hover:not(.selected) {
    color: #cdd6f4 !important;
    background: #1e1e2e !important;
}

/* ── 聊天气泡 ── */
.message-wrap {
    background: transparent !important;
}
.message.user {
    background: linear-gradient(135deg, #313244, #45475a) !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #cdd6f4 !important;
    border: 1px solid #45475a !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}
.message.bot {
    background: linear-gradient(135deg, #1e1e2e, #181825) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: #cdd6f4 !important;
    border: 1px solid #313244 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

/* ── 聊天区域 ── */
#chatbot-window {
    background: #181825 !important;
    border: 1px solid #313244 !important;
    border-radius: 16px !important;
}

/* ── 输入框 ── */
#msg-input textarea {
    background: #1e1e2e !important;
    border: 1px solid #45475a !important;
    border-radius: 12px !important;
    color: #cdd6f4 !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s ease !important;
    resize: none !important;
}
#msg-input textarea:focus {
    border-color: #cba6f7 !important;
    box-shadow: 0 0 0 3px rgba(203,166,247,0.15) !important;
    outline: none !important;
}

/* ── 按钮 ── */
#send-btn {
    background: linear-gradient(135deg, #cba6f7, #89b4fa) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #11111b !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(203,166,247,0.3) !important;
}
#send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(203,166,247,0.45) !important;
    filter: brightness(1.08) !important;
}
#send-btn:active {
    transform: translateY(0) !important;
}

#clear-btn {
    background: #1e1e2e !important;
    border: 1px solid #45475a !important;
    border-radius: 12px !important;
    color: #6c7086 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
#clear-btn:hover {
    border-color: #f38ba8 !important;
    color: #f38ba8 !important;
    background: rgba(243,139,168,0.08) !important;
}

#refresh-btn {
    background: linear-gradient(135deg, #a6e3a1, #94e2d5) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #11111b !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(166,227,161,0.25) !important;
}
#refresh-btn:hover {
    transform: translateY(-1px) !important;
    filter: brightness(1.08) !important;
    box-shadow: 0 4px 18px rgba(166,227,161,0.4) !important;
}

/* ── 示例按钮 ── */
.examples-holder .example {
    background: #1e1e2e !important;
    border: 1px solid #313244 !important;
    border-radius: 8px !important;
    color: #89b4fa !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}
.examples-holder .example:hover {
    border-color: #89b4fa !important;
    background: rgba(137,180,250,0.08) !important;
}

/* ── 仪表盘指标卡片 ── */
#stat-cards {
    display: flex !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
    padding: 4px 0 8px !important;
}
.stat-card {
    flex: 1 1 120px;
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 100px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    border-color: #45475a;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
}
.stat-card .sc-label {
    font-size: 0.72rem;
    color: #6c7086;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    font-weight: 500;
}
.stat-card .sc-value {
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.2;
}
.stat-card .sc-unit {
    font-size: 0.72rem;
    color: #6c7086;
    margin-left: 2px;
}
.stat-card.accent-purple .sc-value { color: #cba6f7; }
.stat-card.accent-blue   .sc-value { color: #89b4fa; }
.stat-card.accent-teal   .sc-value { color: #94e2d5; }
.stat-card.accent-yellow .sc-value { color: #f9e2af; }
.stat-card.accent-red    .sc-value { color: #f38ba8; }
.stat-card.accent-green  .sc-value { color: #a6e3a1; }
.stat-card.accent-peach  .sc-value { color: #fab387; }

/* ── 图表容器 ── */
/* ── 数据表格 ── */
.dataframe {
    background: #1e1e2e !important;
    border: 1px solid #313244 !important;
    border-radius: 12px !important;
}
.dataframe thead th {
    background: #313244 !important;
    color: #cba6f7 !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}
.dataframe tbody tr {
    transition: background 0.15s ease !important;
}
.dataframe tbody tr:hover {
    background: rgba(203,166,247,0.06) !important;
}
.dataframe tbody td {
    color: #cdd6f4 !important;
    padding: 8px 14px !important;
    border-bottom: 1px solid #1e1e2e !important;
}

/* ── Trace details 样式 ── */
details summary {
    color: #89b4fa !important;
    cursor: pointer !important;
    font-size: 0.85rem !important;
    padding: 4px 0 !important;
    user-select: none !important;
}
details summary:hover { color: #cba6f7 !important; }
details div { font-size: 0.83rem !important; color: #a6adc8 !important; line-height: 1.8 !important; }

/* ── 参考来源样式 ── */
.message.bot p { margin: 0.3em 0 !important; }
"""

# ── 聊天逻辑 ─────────────────────────────────────────────


def _build_pipeline_status(last_timing: dict) -> str:
    route_decision = str(
        last_timing.get("route_decision")
        or last_timing.get("route")
        or last_timing.get("mode")
        or "unknown"
    ).lower()
    align_triggered = bool(last_timing.get("align_triggered", False))
    rerank_active = bool(last_timing.get("rerank_active", False))

    if route_decision == "global":
        segments = ["Global"]
        if align_triggered:
            segments.append("🌐 语义对齐")
        segments.append("📝 摘要合成")
        return " ｜ ".join(segments)

    segments = ["Local" if route_decision == "local" else route_decision.title()]
    mode = str(last_timing.get("mode") or "").lower()
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


def _build_source_block(contexts: list[dict]) -> str:
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


def _build_trace_block(last_timing: dict, history_turns: int = 0) -> str:
    route_decision = str(
        last_timing.get("route_decision")
        or last_timing.get("route")
        or last_timing.get("mode")
        or "unknown"
    )
    route_label = {
        "global": "Global",
        "local": "Local",
    }.get(route_decision.lower(), route_decision.title() if route_decision else "Unknown")
    
    rewrite_triggered = bool(last_timing.get("rewrite_triggered", False))
    rewrite_ms = float(last_timing.get("rewrite_ms", 0.0) or 0.0)
    original_query = str(last_timing.get("original_query") or "").strip()
    rewritten_query = str(last_timing.get("rewritten_query") or "").strip()
    
    align_triggered = bool(last_timing.get("align_triggered", False))
    rerank_active = bool(last_timing.get("rerank_active", False))
    align_ms = float(last_timing.get("align_ms", 0.0) or 0.0)
    rerank_ms = float(last_timing.get("rerank_ms", 0.0) or 0.0)
    pure_search_ms = float(last_timing.get("pure_search_ms", 0.0) or 0.0)
    aligned_query = str(last_timing.get("aligned_query") or "").strip()
    query_alignment = last_timing.get("query_alignment") or {}
    expansions = []
    if isinstance(query_alignment, dict):
        raw_expansions = query_alignment.get("expansions") or []
        if isinstance(raw_expansions, list):
            expansions = [str(e).strip() for e in raw_expansions if str(e).strip()]

    expansion_text = "，".join(expansions) if expansions else "无"
    aligned_query_line = (
        f"<div><strong>对齐后查询</strong>: {aligned_query}</div>" if aligned_query else ""
    )
    history_line = (
        f"<div><strong>历史轮数</strong>: {history_turns} 轮已注入</div>"
        if history_turns > 0
        else "<div><strong>历史轮数</strong>: 未注入（首轮或已禁用）</div>"
    )
    
    rewrite_line = (
        f"<div><strong>多轮重写</strong>: {'✅ 已触发' if rewrite_triggered else '⏭️ 未触发'}</div>"
        f"{(f'<div><strong>重写前查询</strong>: {original_query}</div>' if rewrite_triggered else '')}"
        f"{(f'<div><strong>重写后查询</strong>: {rewritten_query}</div>' if rewrite_triggered else '')}"
    )

    return (
        "<details><summary>🔍 检索管线执行详情 (Trace)</summary>"
        "<div style=\"margin-top:0.6em;padding:0 4px\">"
        f"{history_line}"
        f"{rewrite_line}"
        f"<div><strong>意图路由</strong>: {route_label}</div>"
        f"<div><strong>前置对齐</strong>: {'✅ 已触发' if align_triggered else '⏭️ 未触发'}</div>"
        f"<div><strong>对齐扩展词</strong>: {expansion_text}</div>"
        f"{aligned_query_line}"
        f"<div><strong>精排介入</strong>: {'✅ 已触发' if rerank_active else '⏭️ 未触发'}</div>"
        f"<div><strong>重写耗时</strong>: {rewrite_ms:.1f} ms</div>"
        f"<div><strong>对齐耗时</strong>: {align_ms:.1f} ms</div>"
        f"<div><strong>精排耗时</strong>: {rerank_ms:.1f} ms</div>"
        f"<div><strong>纯检索耗时</strong>: {pure_search_ms:.1f} ms</div>"
        "</div></details>"
    )


def _build_final_assistant_content(
    answer: str,
    contexts: list[dict],
    last_timing: dict,
    history_turns: int = 0,
) -> str:
    sections = [answer]
    sections.append(_build_source_block(contexts))
    sections.append(_build_trace_block(last_timing, history_turns))
    return "\n\n".join(section for section in sections if section)


def chat(message: str, history: list | None):
    """流式聊天 + 指标采集（批量 yield 减少 UI 重绘）"""
    t_start = time.perf_counter()
    history = history or []
    base_history = history + [{"role": "user", "content": message}]

    # 检索
    try:
        with Timer() as t_retrieve:
            contexts = retriever.retrieve(message, history=history)
    except Exception as e:
        logger.error("检索失败: %s", e)
        error_content = _build_final_assistant_content(f"⚠️ 检索出错: {e}", [], retriever.last_timing)
        yield base_history + [{"role": "assistant", "content": error_content}]
        return

    # 计算实际注入的历史轮数（取有效 user+assistant 对，上限 LLM_HISTORY_TURNS）
    _valid_history = [
        m for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    history_turns_injected = min(len(_valid_history) // 2, config.LLM_HISTORY_TURNS)

    # 流式生成（批量累积后 yield，减少渲染频率）
    answer = ""
    char_buf = 0
    generate_error: Exception | None = None
    generate_ms = 0.0
    try:
        with Timer() as t_gen:
            for token in generator.generate_stream(message, contexts, history=history):
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

    # 指标记录
    rerank_ms = float(retriever.last_timing.get("rerank_ms", 0.0) or 0.0)
    align_ms = float(retriever.last_timing.get("align_ms", 0.0) or 0.0)
    pure_search_ms = max(t_retrieve.elapsed_ms - rerank_ms - align_ms, 0.0)
    retriever.last_timing["pure_search_ms"] = pure_search_ms
    pipeline_status = _build_pipeline_status(retriever.last_timing)

    total_ms = (time.perf_counter() - t_start) * 1000
    collector.record(RequestMetrics(
        timestamp=time.time(),
        query=message,
        embed_ms=retriever.last_timing.get("embed_ms", 0),
        search_ms=pure_search_ms,
        align_ms=align_ms,
        rerank_ms=rerank_ms,
        generate_ms=generate_ms,
        total_ms=total_ms,
        num_sources=len(contexts),
        answer_length=len(answer),
        retrieve_mode=pipeline_status,
    ))

    # 最终回复
    if generate_error is not None:
        error_msg = answer + f"\n\n⚠️ 生成中断: {generate_error}" if answer else f"⚠️ 生成出错: {generate_error}"
        final_content = _build_final_assistant_content(
            error_msg, contexts, retriever.last_timing, history_turns_injected
        )
    else:
        final_content = _build_final_assistant_content(
            answer, contexts, retriever.last_timing, history_turns_injected
        )

    yield base_history + [{"role": "assistant", "content": final_content}]


# ── 仪表盘逻辑（纯 SVG，无 matplotlib）────────────────────

_BAR_COLORS  = ["#89dceb", "#f9e2af", "#f38ba8"]
_LINE_COLORS = ["#f38ba8", "#89dceb"]
_NO_DATA_SVG = (
    '<svg viewBox="0 0 400 160" style="width:100%;height:auto">'
    '<text x="200" y="88" text-anchor="middle" fill="#6c7086" font-size="14">暂无数据</text>'
    '</svg>'
)


def _svg_bar(categories: list, values: list, colors: list) -> str:
    """生成内联 SVG 柱状图，无需 matplotlib"""
    W, H = 420, 180
    pl, pr, pt, pb = 48, 16, 24, 44
    cw, ch = W - pl - pr, H - pt - pb
    max_v = max(values) if any(v > 0 for v in values) else 1
    n = len(categories)
    slot = cw / n
    bw = slot * 0.55

    rects, xlabels, vlabels, gridlines = [], [], [], []
    # 横向网格线（4条）
    for i in range(1, 5):
        gy = pt + ch * i / 4
        gridlines.append(
            f'<line x1="{pl}" y1="{gy:.1f}" x2="{W-pr}" y2="{gy:.1f}" '
            f'stroke="#313244" stroke-width="0.8" stroke-dasharray="4,3"/>'
        )
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        x = pl + slot * i + (slot - bw) / 2
        bh = ch * val / max_v
        y = pt + ch - bh
        rects.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'fill="{color}" rx="4"/>'
        )
        vlabels.append(
            f'<text x="{x + bw/2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
            f'fill="#cdd6f4" font-size="10">{val:.0f}</text>'
        )
        xlabels.append(
            f'<text x="{x + bw/2:.1f}" y="{H - 8}" text-anchor="middle" '
            f'fill="#6c7086" font-size="11">{cat}</text>'
        )
    axes = (
        f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+ch}" stroke="#45475a" stroke-width="1"/>'
        f'<line x1="{pl}" y1="{pt+ch}" x2="{W-pr}" y2="{pt+ch}" stroke="#45475a" stroke-width="1"/>'
    )
    body = "".join(gridlines) + axes + "".join(rects) + "".join(vlabels) + "".join(xlabels)
    return f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">{body}</svg>'


def _svg_line(x_vals: list, series: list[tuple]) -> str:
    """生成内联 SVG 折线图。series = [(label, values, color), ...]"""
    W, H = 420, 180
    pl, pr, pt, pb = 48, 20, 24, 44
    cw, ch = W - pl - pr, H - pt - pb
    all_v = [v for _, vals, _ in series for v in vals]
    max_v = max(all_v) if all_v and max(all_v) > 0 else 1
    n = len(x_vals)

    gridlines, paths, areas, dots, legends = [], [], [], [], []
    for i in range(1, 5):
        gy = pt + ch * i / 4
        gridlines.append(
            f'<line x1="{pl}" y1="{gy:.1f}" x2="{W-pr}" y2="{gy:.1f}" '
            f'stroke="#313244" stroke-width="0.8" stroke-dasharray="4,3"/>'
        )
    for li, (label, vals, color) in enumerate(series):
        pts = [
            (pl + cw * i / max(n - 1, 1), pt + ch * (1 - v / max_v))
            for i, v in enumerate(vals)
        ]
        pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        area_pts = f"{pl},{pt+ch} {pts_str} {pl+cw},{pt+ch}"
        areas.append(f'<polygon points="{area_pts}" fill="{color}" opacity="0.1"/>')
        paths.append(
            f'<polyline points="{pts_str}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )
        dots.extend(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>'
            for px, py in pts
        )
        lx = pl + li * 90
        legends += [
            f'<line x1="{lx}" y1="{H-10}" x2="{lx+14}" y2="{H-10}" stroke="{color}" stroke-width="2"/>',
            f'<text x="{lx+18}" y="{H-6}" fill="#a6adc8" font-size="10">{label}</text>',
        ]
    # X 轴刻度（首/中/尾）
    x_ticks = []
    for i in [0, n // 2, n - 1]:
        if 0 <= i < n:
            px = pl + cw * i / max(n - 1, 1)
            x_ticks.append(
                f'<text x="{px:.1f}" y="{H-26}" text-anchor="middle" fill="#6c7086" font-size="10">{x_vals[i]}</text>'
            )
    axes = (
        f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+ch}" stroke="#45475a" stroke-width="1"/>'
        f'<line x1="{pl}" y1="{pt+ch}" x2="{W-pr}" y2="{pt+ch}" stroke="#45475a" stroke-width="1"/>'
    )
    body = (
        "".join(gridlines) + axes
        + "".join(areas) + "".join(paths) + "".join(dots)
        + "".join(x_ticks) + "".join(legends)
    )
    return f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">{body}</svg>'


def _chart_card(title: str, svg: str) -> str:
    return (
        f'<div style="flex:1;min-width:260px;background:#1e1e2e;border:1px solid #313244;'
        f'border-radius:12px;padding:14px 16px">'
        f'<div style="color:#a6adc8;font-size:0.8rem;font-weight:600;margin-bottom:6px'
        f';letter-spacing:0.04em;text-transform:uppercase">{title}</div>'
        f'{svg}</div>'
    )


def build_dashboard_html() -> str:
    """将整个仪表盘（卡片 + 图表 + 历史表）渲染为单个 HTML 字符串"""
    s = collector.summary()
    h = collector.history

    # ── 指标卡片 ──
    if s["total_requests"] == 0:
        cards_html = (
            '<div id="stat-cards"><div class="stat-card accent-purple" '
            'style="flex:none;width:100%;text-align:center;padding:14px">'
            '<span class="sc-label">💡 暂无数据，发送几条消息后点击刷新</span>'
            '</div></div>'
        )
    else:
        def card(label, value, unit, accent):
            return (
                f'<div class="stat-card {accent}">'
                f'<span class="sc-label">{label}</span>'
                f'<div><span class="sc-value">{value}</span>'
                f'<span class="sc-unit">{unit}</span></div>'
                f'</div>'
            )
        cards = [
            card("总请求数",    s["total_requests"],           "",   "accent-purple"),
            card("均值总耗时",  f"{s['avg_total_ms']:.0f}",    "ms", "accent-blue"),
            card("P95总耗时",   f"{s['p95_total_ms']:.0f}",    "ms", "accent-red"),
            card("纯检索均值",  f"{s['avg_search_ms']:.0f}",   "ms", "accent-teal"),
            card("对齐均值",    f"{s['avg_align_ms']:.0f}",    "ms", "accent-yellow"),
            card("Rerank均值",  f"{s['avg_rerank_ms']:.0f}",   "ms", "accent-peach"),
            card("LLM生成均值", f"{s['avg_generate_ms']:.0f}", "ms", "accent-green"),
        ]
        cards_html = f'<div id="stat-cards">{" ".join(cards)}</div>'

    # ── SVG 图表 ──
    cats   = ["纯检索", "Rerank", "LLM生成"]
    values = [s["avg_search_ms"], s["avg_rerank_ms"], s["avg_generate_ms"]]
    bar_svg  = _svg_bar(cats, values, _BAR_COLORS) if s["total_requests"] > 0 else _NO_DATA_SVG

    if len(h) >= 2:
        x_vals   = [str(i + 1) for i in range(len(h))]
        line_svg = _svg_line(
            x_vals,
            [("总耗时", [m.total_ms for m in h], "#f38ba8"),
             ("LLM生成", [m.generate_ms for m in h], "#89dceb")],
        )
    else:
        line_svg = (
            '<svg viewBox="0 0 400 160" style="width:100%;height:auto">'
            '<text x="200" y="88" text-anchor="middle" fill="#6c7086" font-size="13">'
            '至少需要 2 条请求</text></svg>'
        )

    charts_html = (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px">'
        + _chart_card("延迟分解", bar_svg)
        + _chart_card("耗时趋势", line_svg)
        + '</div>'
    )

    # ── 历史表 ──
    recent = list(reversed(h[-20:]))
    if not recent:
        table_html = (
            '<div style="color:#6c7086;text-align:center;padding:20px;font-size:0.85rem">'
            '暂无历史记录</div>'
        )
    else:
        th_style = (
            'style="background:#313244;color:#cba6f7;font-size:0.78rem;'
            'font-weight:600;padding:8px 12px;text-align:left;'
            'letter-spacing:0.04em;text-transform:uppercase"'
        )
        td_style = 'style="color:#cdd6f4;font-size:0.82rem;padding:7px 12px;border-bottom:1px solid #1e1e2e"'
        headers = ["问题", "管线状态", "纯检索ms", "Rerankms", "生成ms", "总耗时ms", "来源"]
        thead = "<tr>" + "".join(f"<th {th_style}>{h}</th>" for h in headers) + "</tr>"
        tbody_rows = []
        for m in recent:
            q = m.query[:38] + ("…" if len(m.query) > 38 else "")
            row = (
                f"<tr>"
                f"<td {td_style}>{q}</td>"
                f"<td {td_style} style='color:#89b4fa;font-size:0.78rem'>{m.retrieve_mode}</td>"
                f"<td {td_style} style='text-align:right'>{m.search_ms:.0f}</td>"
                f"<td {td_style} style='text-align:right'>{m.rerank_ms:.0f}</td>"
                f"<td {td_style} style='text-align:right'>{m.generate_ms:.0f}</td>"
                f"<td {td_style} style='text-align:right;color:#f9e2af'>{m.total_ms:.0f}</td>"
                f"<td {td_style} style='text-align:center'>{m.num_sources}</td>"
                f"</tr>"
            )
            tbody_rows.append(row)
        table_html = (
            '<div style="margin-top:12px;background:#1e1e2e;border:1px solid #313244;'
            'border-radius:12px;overflow:hidden">'
            '<div style="color:#a6adc8;font-size:0.8rem;font-weight:600;padding:10px 14px 6px;'
            'letter-spacing:0.04em;text-transform:uppercase;border-bottom:1px solid #313244">'
            f'最近 {len(recent)} 条请求记录</div>'
            '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse">'
            f'<thead>{thead}</thead><tbody>{" ".join(tbody_rows)}</tbody>'
            '</table></div></div>'
        )

    return cards_html + charts_html + table_html


# ── Gradio 界面 ──────────────────────────────────────────

_DARK_THEME = gr.themes.Base().set(
    # 背景
    background_fill_primary="#11111b",
    background_fill_primary_dark="#11111b",
    background_fill_secondary="#1e1e2e",
    background_fill_secondary_dark="#1e1e2e",
    body_background_fill="#11111b",
    body_background_fill_dark="#11111b",
    # 区块
    block_background_fill="#1e1e2e",
    block_background_fill_dark="#1e1e2e",
    block_border_color="#313244",
    block_border_color_dark="#313244",
    block_border_width="1px",
    block_radius="14px",
    block_label_background_fill="#1e1e2e",
    block_label_background_fill_dark="#313244",
    block_label_text_color="#a6adc8",
    block_label_text_color_dark="#a6adc8",
    # 输入
    input_background_fill="#181825",
    input_background_fill_dark="#181825",
    input_border_color="#45475a",
    input_border_color_dark="#45475a",
    input_border_color_focus="#cba6f7",
    input_border_color_focus_dark="#cba6f7",
    input_placeholder_color="#6c7086",
    input_placeholder_color_dark="#6c7086",
    input_radius="10px",
    # 按钮
    button_primary_background_fill="linear-gradient(135deg, #cba6f7, #89b4fa)",
    button_primary_background_fill_dark="linear-gradient(135deg, #cba6f7, #89b4fa)",
    button_primary_text_color="#11111b",
    button_primary_text_color_dark="#11111b",
    button_secondary_background_fill="#1e1e2e",
    button_secondary_background_fill_dark="#313244",
    button_secondary_text_color="#a6adc8",
    button_secondary_text_color_dark="#cdd6f4",
    button_secondary_border_color="#45475a",
    button_secondary_border_color_dark="#45475a",
    button_large_radius="12px",
    button_medium_radius="10px",
    button_small_radius="8px",
    # 文字
    body_text_color="#cdd6f4",
    body_text_color_dark="#cdd6f4",
    body_text_color_subdued="#6c7086",
    body_text_color_subdued_dark="#6c7086",
    # 边框
    border_color_primary="#313244",
    border_color_primary_dark="#313244",
    # 阴影
    shadow_drop="0 2px 12px rgba(0,0,0,0.4)",
    shadow_drop_lg="0 4px 24px rgba(0,0,0,0.5)",
    # 链接
    link_text_color="#89b4fa",
    link_text_color_dark="#89b4fa",
    link_text_color_visited="#cba6f7",
    link_text_color_visited_dark="#cba6f7",
    link_text_color_hover="#94e2d5",
    link_text_color_hover_dark="#94e2d5",
    link_text_color_active="#cba6f7",
    link_text_color_active_dark="#cba6f7",
)

with gr.Blocks(
    title="📚 个人知识库助手",
    theme=_DARK_THEME,
    css=CUSTOM_CSS,
    fill_width=True,
) as demo:

    # ── 顶部标题 ──
    with gr.Row(elem_id="app-header"):
        gr.HTML("""
        <div>
            <h1>📚 个人知识库助手</h1>
            <p>基于 RAG 混合检索 · 意图路由 · 流式生成</p>
        </div>
        """)

    # ── 聊天 Tab ──
    with gr.Tab("💬 聊天"):
        chatbot = gr.Chatbot(
            label="对话",
            height=520,
            elem_id="chatbot-window",
            show_label=False,
            placeholder="<div style='text-align:center;color:#45475a;padding:60px 0'>"
                        "<div style='font-size:2.5rem'>📚</div>"
                        "<div style='font-size:1.1rem;margin-top:12px;color:#6c7086'>向我提问，探索你的知识库</div>"
                        "</div>",
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
            send_btn = gr.Button("发送 ↑", variant="primary", scale=1, elem_id="send-btn", min_width=80)

        with gr.Row():
            clear_btn = gr.ClearButton(
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

        submit_event = msg.submit(chat, [msg, chatbot], [chatbot])
        click_event = send_btn.click(chat, [msg, chatbot], [chatbot])

    # ── 仪表盘 Tab ──
    with gr.Tab("📊 可观测仪表盘"):
        with gr.Row(equal_height=True):
            gr.HTML('<span style="color:#6c7086;font-size:0.85rem">点击刷新后查看实时统计</span>')
            refresh_btn = gr.Button(
                "🔄 刷新数据",
                variant="primary",
                scale=0,
                min_width=120,
                elem_id="refresh-btn",
            )

        # 整个仪表盘 = 1 个 HTML 组件（无 gr.Plot / gr.Dataframe）
        dashboard_out = gr.HTML(
            '<div id="stat-cards"><div class="stat-card accent-purple" '
            'style="flex:none;width:100%;text-align:center;padding:14px">'
            '<span class="sc-label">💡 点击「刷新数据」查看统计</span>'
            '</div></div>'
        )

        refresh_btn.click(build_dashboard_html, outputs=[dashboard_out])


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
