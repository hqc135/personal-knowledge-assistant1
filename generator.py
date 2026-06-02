import logging
import re
from typing import Generator as GenType

from openai import OpenAI
import config

logger = logging.getLogger(__name__)

# 匹配参考来源块（"---\n📎 **参考来源**" 及其后内容）和 Trace 块（<details>...</details>）
_RE_SOURCE_BLOCK = re.compile(r"\n*---\n📎 \*\*参考来源\*\*.*", re.DOTALL)
_RE_TRACE_BLOCK = re.compile(r"\n*<details>.*?</details>", re.DOTALL)


class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model = config.LLM_MODEL

    @staticmethod
    def _extract_text(raw_content) -> str:
        """从可能带有 Gradio 多模态封装的格式中安全提取纯文本内容"""
        if isinstance(raw_content, str):
            return raw_content
        elif isinstance(raw_content, list):
            # 处理 [{"text": "...", "type": "text"}] 等格式
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        elif isinstance(raw_content, tuple):
            return str(raw_content[0])
        return str(raw_content or "")

    @staticmethod
    def _clean_assistant_content(content: str) -> str:
        """
        从 assistant 历史消息中剥离 UI 专用装饰块：
        - 参考来源块（"---\\n📎 **参考来源**" 起至末尾）
        - Trace 折叠块（<details>…</details>）
        保留纯回答文本，避免把调试信息污染后续对话上下文。
        """
        text = _RE_SOURCE_BLOCK.sub("", content)
        text = _RE_TRACE_BLOCK.sub("", text)
        return text.rstrip()

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

    def _build_messages(
        self,
        query: str,
        contexts: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """
        构建 LLM 消息列表。

        history 格式：Gradio messages 格式，每条形如
          {"role": "user"|"assistant", "content": "..."}
        取最近 config.LLM_HISTORY_TURNS 轮（每轮 = 1 user + 1 assistant），
        assistant 消息会先经 _clean_assistant_content 剥离 UI 装饰块。
        """
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

        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # ── 注入历史轮次 ──────────────────────────────────────
        max_turns = config.LLM_HISTORY_TURNS
        if max_turns > 0 and history:
            # 只保留 role in {user, assistant} 的消息，跳过其他（如 system）
            valid = [
                m for m in history
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ]
            # 取最近 max_turns 轮（= 2*max_turns 条消息）
            recent = valid[-(max_turns * 2):]
            for msg in recent:
                role = msg["role"]
                content = self._extract_text(msg.get("content"))
                if role == "assistant":
                    content = self._clean_assistant_content(content)
                if content.strip():
                    messages.append({"role": role, "content": content})

        # ── 当前 user 消息（附带检索上下文） ──────────────────
        messages.append(
            {
                "role": "user",
                "content": f"上下文证据链:\n{context_text}\n\n问题: {query}",
            }
        )
        return messages

    def generate(
        self,
        query: str,
        contexts: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """阻塞式生成完整回答"""
        messages = self._build_messages(query, contexts, history)

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

    def generate_stream(
        self,
        query: str,
        contexts: list[dict],
        history: list[dict] | None = None,
    ) -> GenType[str, None, None]:
        """流式生成回答，逐 token yield"""
        messages = self._build_messages(query, contexts, history)

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