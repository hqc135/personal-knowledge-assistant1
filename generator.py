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

    @staticmethod
    def _format_evidence_chain(context: dict) -> str:
        metadata = context.get("metadata") or {}
        source = metadata.get("source", "unknown")
        channels = context.get("channels") or []
        raw_scores = context.get("channel_scores") or {}
        merged_score = context.get("score")

        lines = [f"[来源: {source}]"]
        if channels:
            lines.append(f"[通道: {', '.join(channels)}]")
        if raw_scores:
            score_chain = ", ".join(
                f"{channel}={score:.4f}"
                for channel, score in sorted(raw_scores.items())
            )
            lines.append(f"[raw_scores: {score_chain}]")
        if merged_score is not None:
            lines.append(f"[fused_score: {float(merged_score):.4f}]")

        if metadata:
            evidence_meta = {
                key: value
                for key, value in metadata.items()
                if key not in {"source", "channel_metadata"}
            }
            if evidence_meta:
                lines.append(f"[metadata: {evidence_meta}]")

        lines.append(context.get("text", ""))
        return "\n".join(lines)

    def _build_messages(self, query: str, contexts: list[dict]) -> list[dict]:
        """构建 LLM 消息列表"""
        context_text = "\n\n---\n\n".join(
            self._format_evidence_chain(c)
            for c in contexts
        )

        return [
            {
                "role": "system",
                "content": (
                    "你是一个知识库助手。根据提供的上下文回答问题。"
                    "优先参考每条证据中的 raw_scores、通道与来源信息。"
                    "如果上下文中没有相关信息，明确说明你不知道。"
                    "回答时引用来源文件名，并尽量保留证据链中的关键分值线索。"
                ),
            },
            {
                "role": "user",
                "content": f"上下文证据链:\n{context_text}\n\n问题: {query}",
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
