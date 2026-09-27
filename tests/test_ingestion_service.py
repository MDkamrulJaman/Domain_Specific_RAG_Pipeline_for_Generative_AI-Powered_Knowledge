"""Document ingestion routing, progress, and failure isolation."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi import HTTPException
from app.services import ingestion_service as ingest
from app.services import ingestion_service


def test_nvidia_ingestion_reports_stages(monkeypatch):
    import numpy as np
    embedding = Mock()
    embedding.embed_chunks.return_value = np.ones((1, 3))
    monkeypatch.setattr(ingest, "EmbeddingService", lambda: embedding)
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: Mock())
    stages = []
    result = ingest.process_document("manual.txt", b"hello", "nvidia", lambda fraction, text: stages.append((fraction, text)))
    assert result["chunks_created"] == 1
    assert [stage[0] for stage in stages] == [0.15, 0.3, 0.5, 0.8]


def test_default_upload_goes_only_to_assistant(monkeypatch):
    assistant=Mock()
    assistant.upload.return_value={"file_id":"f","name":"a.txt","status":"Processing"}
    monkeypatch.setattr(ingest,"get_chat_service",lambda _:assistant)
    index=Mock(side_effect=AssertionError("must not index"))
    monkeypatch.setattr(ingest,"index_document",index)
    result=ingest.process_document("a.txt",b"text")
    assert set(result["results"]) == {"pinecone"}
    index.assert_not_called()


def test_nvidia_failure_does_not_fall_back_to_assistant(monkeypatch):
    def fail(*a):
        raise HTTPException(503,"Index unavailable")
    monkeypatch.setattr(ingest,"index_document",fail)
    assistant=Mock(side_effect=AssertionError("must not upload"))
    monkeypatch.setattr(ingest,"get_chat_service",assistant)
    with pytest.raises(HTTPException):
        ingest.process_document("a.txt",b"text","nvidia")
    assistant.assert_not_called()


def test_integrated_upload_skips_huggingface(monkeypatch):
    store = Mock()
    store.integrated_embedding = True
    monkeypatch.setattr(ingest,"get_vectorstore",lambda:store)
    embedding = Mock(side_effect=AssertionError("HF should not be used"))
    monkeypatch.setattr(ingest,"EmbeddingService",embedding)
    result = ingest.process_document("a.txt",b"text","nvidia")
    assert result["results"]["nvidia"]["status"] == "Indexed"
    assert store.add.call_args.args[0] is None
    embedding.assert_not_called()


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


def test_shared_service_rejects_large_files_before_provider_call(monkeypatch):
    monkeypatch.setattr(ingestion_service, "AppSettings", lambda: SimpleNamespace(max_upload_bytes=4))
    factory = Mock()
    monkeypatch.setattr(ingestion_service, "get_chat_service", factory)
    with pytest.raises(HTTPException) as error:
        ingestion_service.process_document("a.txt", b"oversized")
    assert error.value.status_code == 413
    factory.assert_not_called()
