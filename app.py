import gradio as gr
from retriever import Retriever
from generator import Generator

retriever = Retriever()
generator = Generator()

def chat(query: str, history: list):
    contexts = retriever.retrieve(query)
    answer = generator.generate(query, contexts)
    
    # 展示来源
    sources = "\n".join(
        f"- {c['metadata']['source']} (score: {c['score']:.3f})"
        for c in contexts
    )
    full_answer = f"{answer}\n\n---\n📎 参考来源:\n{sources}"
    return full_answer

demo = gr.ChatInterface(
    fn=chat,
    title="个人知识库助手",
    description="基于你的笔记回答问题",
    examples=["总结一下我关于 RAG 的笔记", "我之前记录的面试准备要点有哪些"]
)

if __name__ == "__main__":
    demo.launch()
