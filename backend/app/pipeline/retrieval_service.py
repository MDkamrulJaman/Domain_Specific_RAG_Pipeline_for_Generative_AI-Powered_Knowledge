import logging
import hashlib
import time
import numpy as np
from fastapi import HTTPException, status
from pinecone import Pinecone
from pinecone.models.indexes.schema import SemanticTextField
from app.pipeline.index_schema import _read_field, validate_index_dimension

from app.core.config import RetrievalSettings
from app.pipeline.embedder import get_embedding_service as EmbeddingService

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, settings=None, client=None, embedder_factory=None):
        settings = settings if settings is not None else RetrievalSettings()
        self.embedder_factory = embedder_factory if embedder_factory is not None else EmbeddingService
        self.namespace = settings.PINECONE_NAMESPACE
        self.rerank_model = settings.PINECONE_MODEL
        self.client = client if client is not None else Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        index_info = self.client.describe_index(self.index_name)
        fields = _read_field(_read_field(index_info, "schema"), "fields", {}) or {}
        text_field = fields.get("text")
        self.integrated_embedding = (
            isinstance(text_field, SemanticTextField)
            or _read_field(text_field, "type") == "semantic_text"
        )
        self.dimension = None
        if not self.integrated_embedding:
            self.dimension = validate_index_dimension(index_info, self.index_name, settings.PINECONE_DIMENSION)
            if fields and "_values" not in fields:
                raise HTTPException(503, "This index uses the documents API. Select a vector index "
                                    "with the reserved _values dense field for this upload pipeline.")
        self.index = self.client.Index(self.index_name)
        logger.info(
            "Connected to cloud Pinecone index '%s' in namespace '%s'.",
            self.index_name,
            self.namespace,
        )

    def add(self, embeddings, chunks):
        if getattr(self, "integrated_embedding", False):
            records = []
            for position, chunk in enumerate(chunks):
                text = getattr(chunk, "page_content", str(chunk))
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                records.append({"_id": f"chunk-{position}-{digest}", "text": text,
                                "source": str(getattr(chunk, "metadata", {}).get("source", ""))})
            for offset in range(0, len(records), 96):
                self.index.upsert_records(records=records[offset:offset + 96], namespace=self.namespace)
            return

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("Embeddings and chunks must have matching 2D shapes.")

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Embedding width {vectors.shape[1]} does not match index dimension {self.dimension}.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite numbers.")

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

    def search(self, query: str, top_k: int = 5):
        if not query or not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search failed: The query string cannot be empty.",
            )

        try:
            start_query = time.perf_counter()
            if getattr(self, "integrated_embedding", False):
                response = self.index.search(
                    namespace=self.namespace,
                    query={"inputs": {"text": query}, "top_k": max(top_k * 2, top_k)},
                    fields=["text", "source"],
                )
                hits = _read_field(_read_field(response, "result"), "hits", [])
                documents = [_read_field(hit, "fields", {}).get("text", "") for hit in hits
                             if _read_field(hit, "fields", {}).get("text")]
            else:
                query_vector = getattr(self, "embedder_factory", EmbeddingService)().embed_query(query)[0].tolist()
                response = self.index.query(
                    vector=query_vector,
                    top_k=max(top_k * 2, top_k),
                    include_metadata=True,
                    namespace=self.namespace,
                )
                documents = [
                    match.metadata.get("text", "")
                    for match in response.matches
                    if match.metadata and match.metadata.get("text")
                ]
            query_elapsed = time.perf_counter() - start_query
            logger.info("Pinecone query completed in %.3f seconds.", query_elapsed)

            if not documents:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="The knowledge base is empty. Please upload documents first.",
                )

            documents = list(dict.fromkeys(documents))
            if len(documents) == 1:
                return [{"text": documents[0], "score": None}]
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