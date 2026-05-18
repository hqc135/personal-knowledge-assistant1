import logging
import time
from typing import Generator as GenType

from openai import OpenAI
import config

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model = config.LLM_MODEL

    def _build_messages(self, query: str, contexts: list[dict]) -> list[dict]:
        """构建 LLM 消息列表"""
        context_text = "\n\n---\n\n".join(
            f"[来源: {c['metadata']['source']}]\n{c['text']}"
            for c in contexts
        )

        return [
            {
                "role": "system",
                "content": (
                    "你是一个知识库助手。根据提供的上下文回答问题。"
                    "如果上下文中没有相关信息，明确说明你不知道。"
                    "回答时引用来源文件名。"
                ),
            },
            {
                "role": "user",
                "content": f"上下文:\n{context_text}\n\n问题: {query}",
            },
        ]

    def generate(self, query: str, contexts: list[dict]) -> str:
        """阻塞式生成完整回答"""
        messages = self._build_messages(query, contexts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM 生成失败: %s", e)
            raise

    def generate_stream(self, query: str, contexts: list[dict]) -> GenType[str, None, None]:
        """流式生成回答，逐 token yield"""
        messages = self._build_messages(query, contexts)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("LLM 流式生成失败: %s", e)
            raise
