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
        
        # 💡 极简对齐：无条件信任上游契约，直接从 root 层取数据，毫无臃肿逻辑
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
            
        # 💡 动态标签：仅用一行代码替换原本硬编码的 fused_score 即可
        score_type = metadata.get("score_type", "fused_score")
        if merged_score is not None:
            lines.append(f"[{score_type}: {float(merged_score):.4f}]")
            
        # 裸露未压缩的精排原始分（如果有）
        if "raw_rerank_logit" in metadata:
            lines.append(f"[raw_rerank_logit: {float(metadata['raw_rerank_logit']):.4f}]")

        if metadata:
            # 清理已经被格式化暴露出来的内部键，降低 prompt 杂质
            evidence_meta = {
                key: value
                for key, value in metadata.items()
                if key not in {"source", "channel_metadata", "score_type", "raw_rerank_logit", "scope", "is_global_summary_block"}
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

        # 💡 提示词也做了深度精简，仅用短小精悍的说明教大模型认清量纲即可，完全不占 token
        system_prompt = (
            "你是一个知识库助手。根据提供的上下文回答问题。\n"
            "【分数语义说明】：\n"
            "- raw_scores: 各通道原始分数，仅用于调试与审计\n"
            "- rerank_compressed_score: 二阶段精排置信度(0~1，越接近1越相关)\n"
            "- rrf_position_score: 多通道粗排融合位置分\n"
            "- vector_similarity: 纯向量空间几何相似度\n"
            "- neighbor_boost_score: 邻居切块拓扑扩展分\n\n"
            "优先参考包含强精排证据（如含有 raw_rerank_logit 或高 rerank_compressed_score）的上下文。"
            "回答时引用来源文件名，并尽量在正文关键要点后保留分值线索。如果上下文中没有相关信息，明确说明你不知道。"
        )

        return [
            {
                "role": "system",
                "content": system_prompt,
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