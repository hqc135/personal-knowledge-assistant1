"""
RAG 管线编排层：将检索器和生成器组合为一个可独立测试的管线对象。

职责：
- 持有 Retriever 和 Generator 实例（依赖注入）
- 提供 Pipeline 对象供 UI 层使用，解耦业务逻辑与 Gradio
- 不包含任何 UI 渲染代码
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGPipeline:
    """
    RAG 管线，封装 retriever + generator，供 UI 层注入使用。

    UI 层通过 pipeline.retriever.retrieve_with_trace() 和
    pipeline.generator.generate_stream() 调用，
    避免直接依赖全局实例。
    """
    retriever: object  # Retriever instance
    generator: object  # Generator instance


def build_pipeline() -> RAGPipeline:
    """
    按默认配置构建并返回 RAGPipeline 实例。

    app.py 中调用此函数获得管线，避免在启动入口中混入初始化逻辑。
    """
    from retriever import Retriever
    from generator import Generator

    retriever = Retriever()
    generator = Generator()
    return RAGPipeline(retriever=retriever, generator=generator)
