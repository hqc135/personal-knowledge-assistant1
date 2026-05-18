import logging

import gradio as gr
from retriever import Retriever
from generator import Generator
from logger import setup_logging

# 初始化日志
setup_logging()

logger = logging.getLogger(__name__)

retriever = Retriever()
generator = Generator()


def chat(query: str, history: list):
    """流式聊天：逐 token 返回，实时显示"""
    try:
        contexts = retriever.retrieve(query)
    except Exception as e:
        logger.error("检索失败: %s", e)
        yield f"⚠️ 检索出错: {e}"
        return

    # 拼接来源信息
    sources = "\n".join(
        f"- {c['metadata']['source']} (score: {c['score']:.3f})"
        for c in contexts
    )
    source_block = f"\n\n---\n📎 参考来源:\n{sources}"

    # 流式生成回答
    answer = ""
    try:
        for token in generator.generate_stream(query, contexts):
            answer += token
            yield answer + source_block
    except Exception as e:
        logger.error("生成失败: %s", e)
        if answer:
            yield answer + f"\n\n⚠️ 生成中断: {e}" + source_block
        else:
            yield f"⚠️ 生成出错: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="📚 个人知识库助手",
    description="基于你的笔记回答问题（流式输出）",
    examples=["总结一下我关于 RAG 的笔记", "我之前记录的面试准备要点有哪些"],
)

if __name__ == "__main__":
    demo.launch()
