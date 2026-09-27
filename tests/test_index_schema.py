"""Pinecone SDK schema compatibility and dimension validation."""

from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.schema import DenseVectorField, IndexSchema, SemanticTextField, SparseVectorField
from app.pipeline.retrieval_service import validate_index_dimension


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
