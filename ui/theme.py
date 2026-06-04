"""
UI 主题与样式常量：Gradio 主题配置、自定义 CSS、颜色常量。

从 app.py 解耦，职责单一：只负责外观定义，不含业务逻辑。
"""
import gradio as gr

# ── 颜色常量 ─────────────────────────────────────────────
BAR_COLORS = ["#89dceb", "#f9e2af", "#f38ba8"]
LINE_COLORS = ["#f38ba8", "#89dceb"]

# ── 自定义 CSS ────────────────────────────────────────────
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

# ── Gradio 深色主题 ──────────────────────────────────────
DARK_THEME = gr.themes.Base().set(
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
