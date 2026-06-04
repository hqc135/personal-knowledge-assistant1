"""
Gradio 应用入口：聊天 + 可观测性仪表盘

架构层次（由此文件向下依赖）：
  app.py
    └── core/pipeline.py      (RAGPipeline 组装)
    └── ui/theme.py           (CSS / Gradio 主题)
    └── ui/chat_view.py       (聊天 Tab 布局 + 回调)
    └── ui/dashboard_view.py  (仪表盘 Tab 布局 + SVG 图表)
"""
import gradio as gr

from logger import setup_logging
from core.pipeline import build_pipeline
from ui.theme import CUSTOM_CSS, DARK_THEME
from ui.chat_view import build_chat_tab
from ui.dashboard_view import build_dashboard_tab

# ── 初始化 ───────────────────────────────────────────────

setup_logging()
pipeline = build_pipeline()

# ── Gradio 界面 ──────────────────────────────────────────

with gr.Blocks(
    title="📚 个人知识库助手",
    theme=DARK_THEME,
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
        build_chat_tab(pipeline)

    # ── 仪表盘 Tab ──
    with gr.Tab("📊 可观测仪表盘"):
        build_dashboard_tab()


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
