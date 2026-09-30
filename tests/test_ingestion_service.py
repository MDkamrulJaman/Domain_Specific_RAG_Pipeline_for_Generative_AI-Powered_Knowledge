"""Document ingestion routing, progress, and failure isolation."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi import HTTPException
from app.services import ingestion_service as ingest
from app.services import ingestion_service


def test_default_upload_goes_only_to_assistant(monkeypatch):
    assistant=Mock()
    assistant.upload.return_value={"file_id":"f","name":"a.txt","status":"Processing"}
    monkeypatch.setattr(ingest,"get_chat_service",lambda _:assistant)
    result=ingest.process_document("a.txt",b"text")
    assert set(result["results"]) == {"pinecone"}


def test_shared_service_rejects_large_files_before_provider_call(monkeypatch):
    monkeypatch.setattr(ingestion_service, "AppSettings", lambda: SimpleNamespace(max_upload_bytes=4))
    factory = Mock()
    monkeypatch.setattr(ingestion_service, "get_chat_service", factory)
    with pytest.raises(HTTPException) as error:
        ingestion_service.process_document("a.txt", b"oversized")
    assert error.value.status_code == 413
    factory.assert_not_called()
