from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.schema import (
    DenseVectorField, IndexSchema, SemanticTextField, SparseVectorField,
)

from app.pipeline.retrieval_service import validate_index_dimension
from app.services import provider_service


def index_with(fields):
    return IndexModel(
        name="rag", schema=IndexSchema(fields=fields), status=None,
        deployment=None, deletion_protection="disabled",
    )


def test_semantic_schema_reproduces_sdk_error_and_returns_actionable_error():
    index = index_with({"text": SemanticTextField(model="llama-text-embed-v2")})
    with pytest.raises(AttributeError, match="no dense or sparse"):
        _ = index.dimension
    with pytest.raises(HTTPException) as error:
        validate_index_dimension(index, "rag", 1024)
    assert error.value.status_code == 503
    assert "text: semantic_text" in error.value.detail
    assert "dimension 1024" in error.value.detail
    assert "PINECONE_INDEX_NAME" in error.value.detail


def test_typed_dense_schema_is_accepted():
    index = index_with({"vector": DenseVectorField(dimension=1024, metric="cosine")})
    assert validate_index_dimension(index, "rag", 1024) == 1024


def test_dictionary_schema_is_accepted():
    index = {"schema": {"fields": {"vector": {"type": "dense_vector", "dimension": 1024}}}}
    assert validate_index_dimension(index, "rag", 1024) == 1024


def test_legacy_dimension_is_accepted():
    assert validate_index_dimension(SimpleNamespace(dimension=1024), "rag", 1024) == 1024


@pytest.mark.parametrize("fields", [
    {},
    {"sparse": SparseVectorField()},
    {"first": DenseVectorField(dimension=1024, metric="cosine"),
     "second": DenseVectorField(dimension=1024, metric="cosine")},
])
def test_incompatible_schemas_are_rejected(fields):
    with pytest.raises(HTTPException) as error:
        validate_index_dimension(index_with(fields), "rag", 1024)
    assert error.value.status_code == 503
    assert "requires an index with one dense-vector field" in error.value.detail


def test_dimension_mismatch_is_rejected():
    index = index_with({"vector": DenseVectorField(dimension=768, metric="cosine")})
    with pytest.raises(HTTPException) as error:
        validate_index_dimension(index, "rag", 1024)
    assert "dimension 768" in error.value.detail
    assert "PINECONE_DIMENSION is 1024" in error.value.detail


def test_connection_check_preserves_configuration_error(monkeypatch):
    monkeypatch.setattr(provider_service, "provider_configuration", lambda _: {"configured": True})
    def fail(_):
        raise HTTPException(503, detail="Select a dense index of dimension 1024.")
    monkeypatch.setattr(provider_service, "get_chat_service", fail)
    result = provider_service.inspect_provider("nvidia")
    assert result["connected"] is False
    assert result["message"] == "Select a dense index of dimension 1024."


def test_document_index_rejected_before_vector_api_use(monkeypatch):
    from unittest.mock import Mock
    from app.pipeline import retrieval_service
    settings = SimpleNamespace(PINECONE_API_KEY="key", PINECONE_NAMESPACE="shared",
        PINECONE_MODEL="rerank", PINECONE_INDEX_NAME="index", PINECONE_DIMENSION=1024)
    client = Mock()
    client.describe_index.return_value = index_with({"embedding": DenseVectorField(dimension=1024, metric="cosine")})
    monkeypatch.setattr(retrieval_service, "RetrievalSettings", lambda: settings)
    monkeypatch.setattr(retrieval_service, "Pinecone", lambda **kw: client)
    with pytest.raises(HTTPException, match="documents API"):
        retrieval_service.VectorStore()
    client.Index.assert_not_called()


@pytest.mark.parametrize("values", [[[1, 2]], [[1, float("nan"), 3]]])
def test_invalid_vectors_never_reach_pinecone(values):
    from unittest.mock import Mock
    from app.pipeline.retrieval_service import VectorStore
    store = VectorStore.__new__(VectorStore)
    store.dimension = 3
    store.index = Mock()
    with pytest.raises(ValueError):
        store.add(values, [SimpleNamespace(page_content="text", metadata={})])
    store.index.upsert.assert_not_called()


def test_incompatible_storage_fails_before_embedding(monkeypatch):
    from unittest.mock import Mock
    from app.services import ingestion_service as ingest
    embedding = Mock()
    monkeypatch.setattr(ingest, "EmbeddingService", embedding)
    def fail():
        raise HTTPException(503, "Incompatible index")
    monkeypatch.setattr(ingest, "get_vectorstore", fail)
    with pytest.raises(HTTPException, match="Incompatible index"):
        ingest.process_document("test.txt", b"document text", "nvidia")
    embedding.assert_not_called()
