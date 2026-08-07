import faiss
import numpy as np
import os
import pickle
import logging
from fastapi import HTTPException, status  # Import standard FastAPI exceptions

from app.pipeline.embedder import EmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

        self.index_path = os.path.join(path, "index.faiss")
        self.meta_path = os.path.join(path, "meta.pkl")

        self.index = None
        self.metadata = []

        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info("Successfully loaded existing FAISS index and metadata.")
            except Exception as e:
                logger.error(f"Failed to read existing vector store files: {str(e)}")
                self.index = None
                self.metadata = []

    def add(self, embeddings, chunks):
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings)

        for c in chunks:
            # Handle both LangChain Document objects and raw text strings safely
            text_content = getattr(c, 'page_content', str(c))
            self.metadata.append({"text": text_content})

    def search(self, query: str, top_k: int):
        # 1. Guard against empty vector index
        if self.index is None or self.index.ntotal == 0:
            error_msg = "Search failed: The knowledge base is currently empty. Please upload documents first."
            logger.warning(error_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )

        if not query or not query.strip():
            error_msg = "Search failed: The query string cannot be empty."
            logger.warning(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )

        try:
            # 2. Extract embedding vector
            load_embedder = EmbeddingService()
            q = load_embedder.embed_query(query)
            
            # 3. Perform search
            distances, indices = self.index.search(q, top_k)

            # 4. Filter out any missing or invalid FAISS indices (-1 indicates no match)
            valid_results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.metadata):
                    valid_results.append(self.metadata[idx])

            if not valid_results:
                logger.info(f"Search completed for query '{query}' but yielded zero matching chunks.")
            
            return valid_results

        except Exception as e:
            # Catch internal model parsing or FAISS mismatch errors cleanly
            logger.error(f"Unexpected error during FAISS index searching: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while processing your search query: {str(e)}"
            )

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info("VectorStore saved successfully.")