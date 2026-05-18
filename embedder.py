"""
统一 Embedding 模块：调用智谱 embedding-3 API
"""
import numpy as np
from zhipuai import ZhipuAI
import config


class ZhipuEmbedder:
    def __init__(self):
        self.client = ZhipuAI(api_key=config.ZHIPUAI_API_KEY)
        self.model = config.EMBEDDING_MODEL
        self.dimensions = config.EMBEDDING_DIMENSIONS

    def encode(self, texts, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """
        对单条或多条文本进行 Embedding，返回 numpy 数组。
        兼容 sentence-transformers 的 encode 接口风格。
        """
        if isinstance(texts, str):
            texts = [texts]

        # 智谱 API 单次最多支持 64 条，需分批
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        result = np.array(all_embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            result = result / norms

        return result
