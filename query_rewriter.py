import logging
import re
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

_RE_SOURCE_BLOCK = re.compile(r"\n*---\n📎 \*\*参考来源\*\*.*", re.DOTALL)
_RE_TRACE_BLOCK = re.compile(r"\n*<details>.*?</details>", re.DOTALL)

class QueryRewriter:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model = config.QUERY_REWRITER_MODEL

    def _clean_content(self, content: str) -> str:
        text = _RE_SOURCE_BLOCK.sub("", content)
        text = _RE_TRACE_BLOCK.sub("", text)
        return text.rstrip()

    def rewrite(self, query: str, history: list[dict]) -> str:
        """
        基于多轮对话历史重写当前查询，进行指代消解和上下文补全。
        如果不满足重写条件或重写失败，原样返回原 query。
        """
        if not history:
            return query
            
        # 提取有效历史
        valid_history = [
            m for m in history
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]
        if not valid_history:
            return query
            
        # 限制历史轮数
        turns = min(len(valid_history) // 2, config.LLM_HISTORY_TURNS)
        if turns == 0:
            return query
            
        recent = valid_history[-(turns * 2):]
        
        history_text = ""
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            content = str(msg.get("content") or "")
            if msg["role"] == "assistant":
                content = self._clean_content(content)
            if content.strip():
                history_text += f"{role}: {content}\n"
                
        prompt = (
            "你是一个查询重写助手。你的任务是基于用户的历史对话，"
            "将用户最新提出的问题重写为一个完整、独立且表意清晰的句子，"
            "以解决指代不明或主语缺失的问题（例如将“它有什么缺点？”改写为“RAG有什么缺点？”）。\n\n"
            "【规则】：\n"
            "1. 如果当前问题已经很完整，没有指代不明的情况，请直接原样输出当前问题。\n"
            "2. 你的输出只能包含重写后的查询语句，不能包含任何解释、寒暄或其他废话。\n"
            "3. 不要尝试回答问题，只需重写查询。\n\n"
            f"【历史对话】：\n{history_text}\n"
            f"【当前问题】：{query}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = (response.choices[0].message.content or "").strip()
            if not rewritten:
                return query
            
            # 清理可能的常见冗余前缀
            for prefix in ["重写后查询：", "重写后的查询：", "重写后：", "重写：", "重写后查询:", "重写后的查询:"]:
                if rewritten.startswith(prefix):
                    rewritten = rewritten[len(prefix):].strip()
                    break
                    
            # 简单校验，如果模型生成了异常长的文本（可能是解答了问题），则丢弃
            if len(rewritten) > max(len(query) * 3, 50) and "？" not in rewritten and "?" not in rewritten:
                 logger.debug("重写结果可能异常(过长)，回退: %s", rewritten)
                 return query
                 
            return rewritten
        except Exception as e:
            logger.warning("查询重写失败，回退到原查询: %s", e)
            return query
