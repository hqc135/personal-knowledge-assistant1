"""
集中配置管理：从 .env 文件加载，统一管理所有配置项
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


def _require_env(key: str) -> str:
    """获取必需的环境变量，未设置时给出清晰提示"""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"环境变量 {key} 未设置。\n"
            f"请复制 .env.example 为 .env 并填入你的 API Key：\n"
            f"  cp .env.example .env"
        )
    return value


# ── API Keys ──────────────────────────────────────────────
ZHIPUAI_API_KEY = _require_env("ZHIPUAI_API_KEY")
DEEPSEEK_API_KEY = _require_env("DEEPSEEK_API_KEY")

# ── Embedding 配置 ────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# ── LLM 配置 ─────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Retriever 配置 ────────────────────────────────────────
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_FINAL_K = int(os.getenv("RETRIEVER_FINAL_K", "3"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# ── Data Pipeline 配置 ────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_knowledge_base")
NOTES_DIR = os.getenv("NOTES_DIR", "./notes")
