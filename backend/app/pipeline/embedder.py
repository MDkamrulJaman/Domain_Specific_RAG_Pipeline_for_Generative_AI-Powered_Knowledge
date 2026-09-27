from functools import lru_cache
import logging
import time

import numpy as np
from huggingface_hub import InferenceClient
from app.core.config import RetrievalSettings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """One reusable HTTP client for externally embedded dense indexes."""
    def __init__(self, settings=None, client=None):
        settings = settings if settings is not None else RetrievalSettings()
        self.model_name = settings.EMBEDDING_MODEL
        self.client = client if client is not None else InferenceClient(provider="hf-inference", api_key=settings.HF_TOKEN, timeout=30)

    def embed_chunks(self, chunks):
        if not chunks:
            return np.empty((0, 0), dtype="float32")
        start = time.perf_counter()
        vectors = self.client.feature_extraction([chunk.page_content for chunk in chunks], model=self.model_name)
        logger.info("Document embedding completed in %.3fs", time.perf_counter() - start)
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, query):
        start = time.perf_counter()
        vector = self.client.feature_extraction(query, model=self.model_name)
        logger.info("Query embedding completed in %.3fs", time.perf_counter() - start)
        return np.asarray([vector], dtype="float32")


@lru_cache(maxsize=1)
def get_embedding_service():
    return EmbeddingService()
