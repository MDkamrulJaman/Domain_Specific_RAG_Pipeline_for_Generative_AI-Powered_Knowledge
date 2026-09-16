import logging
import hashlib
import time
import numpy as np
from fastapi import HTTPException, status
from pinecone import Pinecone

from app.core.config import Settings
from app.pipeline.embedder import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        settings = Settings()
        self.namespace = settings.PINECONE_NAMESPACE
        self.rerank_model = settings.PINECONE_MODEL
        self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        index_info = self.client.describe_index(self.index_name)
        if index_info.dimension != settings.PINECONE_DIMENSION:
            raise RuntimeError(
                f"Pinecone index '{self.index_name}' has dimension "
                f"{index_info.dimension}, expected {settings.PINECONE_DIMENSION}."
            )
        self.index = self.client.Index(self.index_name)
        logger.info(
            "Connected to cloud Pinecone index '%s' in namespace '%s'.",
            self.index_name,
            self.namespace,
        )

    def add(self, embeddings, chunks):
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("Embeddings and chunks must have matching 2D shapes.")

        records = []
        for position, (vector, chunk) in enumerate(zip(vectors, chunks)):
            text = getattr(chunk, "page_content", str(chunk))
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            records.append(
                {
                    "id": f"chunk-{position}-{digest}",
                    "values": vector.tolist(),
                    "metadata": {
                        "text": text,
                        "source": str(getattr(chunk, "metadata", {}).get("source", "")),
                    },
                }
            )

        if records:
            start = time.perf_counter()
            self.index.upsert(vectors=records, namespace=self.namespace)
            elapsed = time.perf_counter() - start
            logger.info(
                "Pinecone upsert completed in %.3f seconds for %d records.",
                elapsed,
                len(records),
            )

    def search(self, query: str, top_k: int):
        if not query or not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search failed: The query string cannot be empty.",
            )

        try:
            query_vector = EmbeddingService().embed_query(query)[0].tolist()
            start_query = time.perf_counter()
            response = self.index.query(
                vector=query_vector,
                top_k=max(top_k * 2, top_k),
                include_metadata=True,
                namespace=self.namespace,
            )
            query_elapsed = time.perf_counter() - start_query
            logger.info("Pinecone query completed in %.3f seconds.", query_elapsed)

            documents = [
                match.metadata.get("text", "")
                for match in response.matches
                if match.metadata and match.metadata.get("text")
            ]
            if not documents:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="The knowledge base is empty. Please upload documents first.",
                )

            start_rerank = time.perf_counter()
            reranked = self.client.inference.rerank(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                return_documents=True,
            )
            rerank_elapsed = time.perf_counter() - start_rerank
            logger.info("Pinecone rerank completed in %.3f seconds.", rerank_elapsed)

            results = []
            for result in reranked.data:
                document = result.document
                text = (
                    document.get("text", "")
                    if isinstance(document, dict)
                    else getattr(document, "text", "")
                )
                if text:
                    results.append({"text": text, "score": result.score})

            if results:
                logger.info(
                    "Pinecone total retrieval time: %.3f seconds (query: %.3f, rerank: %.3f)",
                    query_elapsed + rerank_elapsed,
                    query_elapsed,
                    rerank_elapsed,
                )
            return results

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error("Unexpected Pinecone search error: %s", e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while processing your search query.",
            ) from e

    def save(self):
        # Pinecone persists writes remotely; this method remains for API compatibility.
        return None