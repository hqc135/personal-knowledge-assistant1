from openai import OpenAI
import config

class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model = config.LLM_MODEL

    def generate(self, query: str, contexts: list[dict]) -> str:
        context_text = "\n\n---\n\n".join(
            f"[来源: {c['metadata']['source']}]\n{c['text']}" 
            for c in contexts
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个知识库助手。根据提供的上下文回答问题。"
                    "如果上下文中没有相关信息，明确说明你不知道。"
                    "回答时引用来源文件名。"
                )
            },
            {
                "role": "user",
                "content": f"上下文:\n{context_text}\n\n问题: {query}"
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content
