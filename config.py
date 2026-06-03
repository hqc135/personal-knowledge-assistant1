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
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
# 多轮对话：注入给 LLM 的历史轮数上限（每轮 = 1 user + 1 assistant）；设为 0 禁用
LLM_HISTORY_TURNS = int(os.getenv("LLM_HISTORY_TURNS", "3"))

# ── Retriever 配置 ────────────────────────────────────────
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_FINAL_K = int(os.getenv("RETRIEVER_FINAL_K", "3"))
RETRIEVER_RECALL_TOP_K = int(os.getenv("RETRIEVER_RECALL_TOP_K", "10"))
RETRIEVER_NEIGHBOR_WINDOW = int(os.getenv("RETRIEVER_NEIGHBOR_WINDOW", "1"))
RETRIEVER_NEIGHBOR_BUDGET = int(os.getenv("RETRIEVER_NEIGHBOR_BUDGET", "2"))
RETRIEVER_WEIGHT_VECTOR = float(os.getenv("RETRIEVER_WEIGHT_VECTOR", "0.45"))
RETRIEVER_WEIGHT_BM25 = float(os.getenv("RETRIEVER_WEIGHT_BM25", "0.35"))
RETRIEVER_WEIGHT_KG = float(os.getenv("RETRIEVER_WEIGHT_KG", "0.20"))
RETRIEVER_WEIGHT_NEIGHBOR = float(os.getenv("RETRIEVER_WEIGHT_NEIGHBOR", "0.10"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# ── Intent Router 配置 ────────────────────────────────────
USE_INTENT_ROUTER = os.getenv("USE_INTENT_ROUTER", "true").lower() == "true"
INTENT_ROUTER_PROTOTYPES_PATH = os.getenv(
    "INTENT_ROUTER_PROTOTYPES_PATH", "./intent_prototypes.json"
)
INTENT_ROUTER_MIN_SCORE = float(os.getenv("INTENT_ROUTER_MIN_SCORE", "0.35"))
INTENT_ROUTER_MIN_MARGIN = float(os.getenv("INTENT_ROUTER_MIN_MARGIN", "0.05"))
INTENT_ROUTER_AGG = os.getenv("INTENT_ROUTER_AGG", "max")
INTENT_ROUTER_FALLBACK = os.getenv("INTENT_ROUTER_FALLBACK", "local")
INTENT_ROUTER_GLOBAL_TOP_K = int(os.getenv("INTENT_ROUTER_GLOBAL_TOP_K", "12"))
INTENT_ROUTER_GLOBAL_MAX_SOURCES = int(
    os.getenv("INTENT_ROUTER_GLOBAL_MAX_SOURCES", "3")
)
INTENT_ROUTER_GLOBAL_MAX_CHARS = int(
    os.getenv("INTENT_ROUTER_GLOBAL_MAX_CHARS", "4000")
)
INTENT_ROUTER_GLOBAL_MODE = os.getenv("INTENT_ROUTER_GLOBAL_MODE", "full")
INTENT_ROUTER_GLOBAL_SUMMARY_MODEL = os.getenv(
    "INTENT_ROUTER_GLOBAL_SUMMARY_MODEL", LLM_MODEL
)
INTENT_ROUTER_GLOBAL_SUMMARY_MAX_TOKENS = int(
    os.getenv("INTENT_ROUTER_GLOBAL_SUMMARY_MAX_TOKENS", "800")
)

# ── Query Aligner 配置 ────────────────────────────────────
USE_QUERY_ALIGNER = os.getenv("USE_QUERY_ALIGNER", "true").lower() == "true"
QUERY_ALIGNER_MODEL = os.getenv("QUERY_ALIGNER_MODEL", LLM_MODEL)
QUERY_ALIGNER_MIN_QUERY_LENGTH = int(os.getenv("QUERY_ALIGNER_MIN_QUERY_LENGTH", "4"))
QUERY_ALIGNER_MIN_CONFIDENCE = float(os.getenv("QUERY_ALIGNER_MIN_CONFIDENCE", "0.55"))
QUERY_ALIGNER_MAX_EXPANSIONS = int(os.getenv("QUERY_ALIGNER_MAX_EXPANSIONS", "4"))
QUERY_ALIGNER_MAX_TERMS = int(os.getenv("QUERY_ALIGNER_MAX_TERMS", "6"))

# ── Query Rewriter 配置 ───────────────────────────────────
USE_QUERY_REWRITER = os.getenv("USE_QUERY_REWRITER", "true").lower() == "true"
QUERY_REWRITER_MODEL = os.getenv("QUERY_REWRITER_MODEL", LLM_MODEL)

# ── Data Pipeline 配置 ────────────────────────────────────
# fallback 字符切块（仅在语义分块关闭或异常降级时使用）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_knowledge_base")
NOTES_DIR = os.getenv("NOTES_DIR", "./notes")

# ── Semantic Chunker 配置 ─────────────────────────────────
# 总开关；设为 false 则回退至旧的 RecursiveCharacterTextSplitter
SEMANTIC_CHUNKER_ENABLED = os.getenv("SEMANTIC_CHUNKER_ENABLED", "true").lower() == "true"
# 百分位阈值：相似度低于第 N 百分位的相邻句对被视为语义断点（越高 → 切点越少）
SEMANTIC_BREAKPOINT_PERCENTILE = float(os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", "85"))
# 滑动平均窗口大小：对句向量做局部平滑，降低单句噪声影响
SEMANTIC_WINDOW_SIZE = int(os.getenv("SEMANTIC_WINDOW_SIZE", "2"))
# chunk 字符下限：产出的 chunk 若小于此值，向前合并以避免碎片
SEMANTIC_MIN_CHUNK_SIZE = int(os.getenv("SEMANTIC_MIN_CHUNK_SIZE", "80"))
# chunk 字符上限：超过此值时对该 chunk 按句子二分强制硬切
SEMANTIC_MAX_CHUNK_SIZE = int(os.getenv("SEMANTIC_MAX_CHUNK_SIZE", "1200"))

# ── 混合检索 配置 ─────────────────────────────────────────
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
RRF_K = int(os.getenv("RRF_K", "60"))

# ── 知识图谱 配置 ─────────────────────────────────────────
KG_TRIPLES_PATH = os.getenv("KG_TRIPLES_PATH", "./kg_triples.json")
USE_KG_EXTRACTION = os.getenv("USE_KG_EXTRACTION", "true").lower() == "true"
USE_KG_RETRIEVAL = os.getenv("USE_KG_RETRIEVAL", "true").lower() == "true"
KG_TOP_K = int(os.getenv("KG_TOP_K", "5"))
KG_SCORE_BASE = float(os.getenv("KG_SCORE_BASE", "0.4"))
KG_MAX_TRIPLES_PER_CHUNK = int(os.getenv("KG_MAX_TRIPLES_PER_CHUNK", "8"))
KG_LLM_MODEL = os.getenv("KG_LLM_MODEL", "deepseek-chat")
KG_FALLBACK_MODEL = os.getenv("KG_FALLBACK_MODEL", "deepseek-chat")
KG_EXTRACTION_WORKERS = int(os.getenv("KG_EXTRACTION_WORKERS", "4"))  # 三元组并发抽取线程数
KG_ENTITY_VECTOR_TOP_K = int(os.getenv("KG_ENTITY_VECTOR_TOP_K", "5"))
KG_ENTITY_VECTOR_MIN_SIM = float(os.getenv("KG_ENTITY_VECTOR_MIN_SIM", "0.35"))

# ── 可观测性 配置 ─────────────────────────────────────────
METRICS_MAX_HISTORY = int(os.getenv("METRICS_MAX_HISTORY", "200"))

