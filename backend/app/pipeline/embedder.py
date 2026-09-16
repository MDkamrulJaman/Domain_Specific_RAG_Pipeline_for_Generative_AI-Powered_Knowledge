import os
import time
import numpy as np
from typing import List, Any
from huggingface_hub import InferenceClient
from app.core.config import Settings

class EmbeddingService:
    def __init__(self):

        settings = Settings()
        self.model_name = settings.EMBEDDING_MODEL  # Use the model name from settings
        """
        Initializes the Hugging Face inference client safely.
        """
        try:
            self.model_name = settings.EMBEDDING_MODEL
            self.client = InferenceClient(
                provider="hf-inference",
                api_key=settings.HF_TOKEN,
            )
        except Exception as e:
            # In production, you'd use a proper logger here
            print(f"CRITICAL: Failed to initialize Hugging Face client: {e}")
            self.client = None

    def _ensure_model_loaded(self):
        """Internal helper to verify the model exists before encoding."""
        if self.client is None:
            raise RuntimeError(
                "Hugging Face inference client is not initialized. Check your HF_TOKEN."
            )

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Extracts text from LangChain-style chunks and returns a float32 NumPy array of vectors.
        """
        self._ensure_model_loaded()
        
        # Extract page content from the chunk objects
        texts = [c.page_content for c in chunks]
        
        if not texts:
            return np.empty((0, 0), dtype="float32")

        start_time = time.perf_counter()
        embeddings = self.client.feature_extraction(
            texts,
            model=self.model_name,
        )
        print(
            f"Embedding API completed in {time.perf_counter() - start_time:.3f} seconds "
            f"for {len(texts)} chunks"
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Converts a single string query into a 2D float32 NumPy vector array.
        """
        self._ensure_model_loaded()
        
        start_time = time.perf_counter()
        embedding = self.client.feature_extraction(
            query,
            model=self.model_name,
        )
        print(f"Query embedding API completed in {time.perf_counter() - start_time:.3f} seconds")
        # Keeps the 2D array structure matching your original code: shape (1, dimensions)
        return np.asarray([embedding], dtype="float32")
    



