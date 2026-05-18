"""
统一 Embedding 模块：调用智谱 embedding-3 API
"""
import logging
import time
from typing import Union

import numpy as np
from zhipuai import ZhipuAI
import config

logger = logging.getLogger(__name__)


class ZhipuEmbedder:
    def __init__(self):
        self.client = ZhipuAI(api_key=config.ZHIPUAI_API_KEY)
        self.model = config.EMBEDDING_MODEL
        self.dimensions = config.EMBEDDING_DIMENSIONS

    def encode(
        self,
        texts: Union[str, list[str]],
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        对文本进行 Embedding，返回 numpy 数组。
        兼容 sentence-transformers 的 encode 接口：
          - 输入 str  → 返回 shape (dim,) 的 1D 数组
          - 输入 list → 返回 shape (N, dim) 的 2D 数组
        """
        single_input = isinstance(texts, str)
        text_list: list[str] = [texts] if isinstance(texts, str) else texts

        # 智谱 API 单次最多支持 64 条，需分批
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            batch_embeddings = self._call_api_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        result = np.array(all_embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            result = result / norms

        # 单条输入返回 1D，匹配 sentence-transformers 行为
        if single_input:
            result = result.squeeze(0)

        return result

    def _call_api_with_retry(
        self, texts: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        """调用智谱 Embedding API，带指数退避重试"""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries:
                    logger.error("Embedding API 调用失败（已重试 %d 次）: %s", max_retries, e)
                    raise
                wait = 2**attempt
                logger.warning(
                    "Embedding API 调用失败（第 %d 次），%ds 后重试: %s",
                    attempt, wait, e,
                )
                time.sleep(wait)
        return []
