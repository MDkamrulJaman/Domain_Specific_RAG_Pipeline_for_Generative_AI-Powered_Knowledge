"""Dense-vector and integrated-text index operations."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest


def test_shared_index_receives_huggingface_vectors(monkeypatch):
    import numpy as np
    from app.pipeline import retrieval_service
    settings = SimpleNamespace(PINECONE_API_KEY="vector-key", PINECONE_NAMESPACE="shared", PINECONE_MODEL="rerank", PINECONE_INDEX_NAME="index", PINECONE_DIMENSION=3)
    sdk = Mock()
    sdk.describe_index.return_value = SimpleNamespace(schema=None, dimension=3)
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(retrieval_service, "RetrievalSettings", lambda: settings)
    monkeypatch.setattr(retrieval_service, "Pinecone", factory)
    store = retrieval_service.VectorStore()
    store.add(np.array([[1, 2, 3]]), [SimpleNamespace(page_content="chunk", metadata={"source": "a.txt"})])
    factory.assert_called_once_with(api_key="vector-key")
    assert sdk.Index.return_value.upsert.call_args.kwargs["vectors"][0]["values"] == [1, 2, 3]
    assert sdk.Index.return_value.upsert.call_args.kwargs["namespace"] == "shared"
    sdk.Index.return_value.upsert_records.assert_not_called()


def test_shared_search_embeds_query_with_huggingface(monkeypatch):
    import numpy as np
    from app.pipeline import retrieval_service
    store = retrieval_service.VectorStore.__new__(retrieval_service.VectorStore)
    store.namespace = "shared"
    store.rerank_model = "rerank"
    store.index = Mock()
    store.index.query.return_value = SimpleNamespace(matches=[SimpleNamespace(metadata={"text": "source"})])
    store.client = Mock()
    store.client.inference.rerank.return_value = SimpleNamespace(data=[SimpleNamespace(document={"text": "source"}, score=0.9)])
    embeddings = Mock()
    embeddings.embed_query.return_value = np.array([[1, 2, 3]])
    monkeypatch.setattr(retrieval_service, "EmbeddingService", lambda: embeddings)
    assert store.search("question", 4) == [{"text": "source", "score": None}]
    embeddings.embed_query.assert_called_once_with("question")
    store.index.query.assert_called_once_with(vector=[1, 2, 3], top_k=8, include_metadata=True, namespace="shared")
    store.index.search.assert_not_called()


def test_integrated_rag_uses_text_records_and_text_search(monkeypatch):
    from app.pipeline import retrieval_service as rs
    settings = SimpleNamespace(PINECONE_API_KEY="key", PINECONE_NAMESPACE="ns", PINECONE_MODEL="rerank", PINECONE_INDEX_NAME="rag", PINECONE_DIMENSION=1024)
    sdk = Mock()
    sdk.describe_index.return_value = {"schema":{"fields":{"text":{"type":"semantic_text"}}}}
    monkeypatch.setattr(rs, "RetrievalSettings", lambda: settings)
    monkeypatch.setattr(rs, "Pinecone", lambda **kw: sdk)
    embedding = Mock(side_effect=AssertionError("HF should not be used"))
    monkeypatch.setattr(rs, "EmbeddingService", embedding)
    store = rs.VectorStore()
    store.add(None, [SimpleNamespace(page_content="source", metadata={})])
    sdk.Index.return_value.upsert_records.assert_called_once()
    sdk.Index.return_value.upsert.assert_not_called()
    sdk.Index.return_value.search.return_value = {"result":{"hits":[{"fields":{"text":"source"}}]}}
    sdk.inference.rerank.return_value = SimpleNamespace(data=[SimpleNamespace(document={"text":"source"},score=1)])
    assert store.search("q") == [{"text":"source","score":None}]
    sdk.Index.return_value.query.assert_not_called()
    embedding.assert_not_called()


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
