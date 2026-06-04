"""
仪表盘视图：负责可观测性仪表盘 Tab 的 UI 布局和 SVG 图表生成。

从 app.py 解耦，职责单一：
- 纯 SVG 图表生成（无 matplotlib 依赖）
- 仪表盘 HTML 组装
- Gradio 仪表盘 Tab 布局
"""
from __future__ import annotations

import gradio as gr
from metrics import collector

# ── SVG 图表常量 ──────────────────────────────────────────
_BAR_COLORS = ["#89dceb", "#f9e2af", "#f38ba8"]
_LINE_COLORS = ["#f38ba8", "#89dceb"]
_NO_DATA_SVG = (
    '<svg viewBox="0 0 400 160" style="width:100%;height:auto">'
    '<text x="200" y="88" text-anchor="middle" fill="#6c7086" font-size="14">暂无数据</text>'
    '</svg>'
)


# ── SVG 图表生成 ──────────────────────────────────────────


def _svg_bar(categories: list, values: list, colors: list) -> str:
    """生成内联 SVG 柱状图，无需 matplotlib。"""
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


# ── 仪表盘 HTML 组装 ──────────────────────────────────────


def build_dashboard_html() -> str:
    """将整个仪表盘（卡片 + 图表 + 历史表）渲染为单个 HTML 字符串。"""
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
    bar_svg = _svg_bar(cats, values, _BAR_COLORS) if s["total_requests"] > 0 else _NO_DATA_SVG

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


# ── Gradio Tab 布局 ───────────────────────────────────────


def build_dashboard_tab() -> None:
    """构建仪表盘 Tab 的 Gradio 组件（在 gr.Tab 上下文内调用）。"""
    with gr.Row(equal_height=True):
        gr.HTML('<span style="color:#6c7086;font-size:0.85rem">点击刷新后查看实时统计</span>')
        refresh_btn = gr.Button(
            "🔄 刷新数据",
            variant="primary",
            scale=0,
            min_width=120,
            elem_id="refresh-btn",
        )

    dashboard_out = gr.HTML(
        '<div id="stat-cards"><div class="stat-card accent-purple" '
        'style="flex:none;width:100%;text-align:center;padding:14px">'
        '<span class="sc-label">💡 点击「刷新数据」查看统计</span>'
        '</div></div>'
    )

    refresh_btn.click(build_dashboard_html, outputs=[dashboard_out])
