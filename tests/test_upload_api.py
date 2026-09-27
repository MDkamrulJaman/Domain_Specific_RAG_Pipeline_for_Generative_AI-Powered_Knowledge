"""Upload endpoint provider selection, size limits, and file cleanup."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi import HTTPException
from app.api.routes import ingest as ingest_routes
from app.services import ingestion_service as ingest
import asyncio
from io import BytesIO
from fastapi import HTTPException, UploadFile


def test_assistant_upload_does_not_use_vector_index(client, monkeypatch):
    service = Mock()
    service.upload.return_value = {"file_id": "f", "name": "manual.txt", "status": "Processing"}
    monkeypatch.setattr(ingest, "get_chat_service", lambda _: service)
    vector = Mock(side_effect=AssertionError("must not use vector index"))
    monkeypatch.setattr(ingest, "get_vectorstore", vector)
    response = client.post("/ingest/upload", data={"provider":"pinecone"}, files={"file": ("manual.txt", b"hello")})
    assert response.status_code == 201
    assert response.json()["results"]["pinecone"]["status"] == "Processing"
    service.upload.assert_called_once_with("manual.txt", b"hello")
    vector.assert_not_called()


def test_nvidia_upload_still_indexes(client, monkeypatch):
    import numpy as np
    embeddings = Mock()
    embeddings.embed_chunks.return_value = np.ones((1, 3))
    store = Mock()
    monkeypatch.setattr(ingest, "EmbeddingService", lambda: embeddings)
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: store)
    response = client.post("/ingest/upload", data={"provider":"nvidia"}, files={"file": ("manual.txt", b"hello")})
    assert response.status_code == 201
    assert response.json()["results"]["nvidia"]["status"] == "Indexed"
    store.add.assert_called_once()


def test_upload_validation_and_cleanup(monkeypatch):
    import asyncio
    from io import BytesIO
    from fastapi import UploadFile
    file = UploadFile(file=BytesIO(b"hello"), filename="unsafe.exe")
    with pytest.raises(HTTPException) as error:
        asyncio.run(ingest_routes.upload_file(file, provider="nvidia"))
    assert error.value.status_code == 400
    assert file.file.closed


def test_api_upload_default_is_pinecone(client,monkeypatch):
    service=Mock()
    service.upload.return_value={"file_id":"f","name":"a.txt","status":"Processing"}
    monkeypatch.setattr(ingest,"get_chat_service",lambda _:service)
    index=Mock(side_effect=AssertionError("index touched"))
    monkeypatch.setattr(ingest,"get_vectorstore",index)
    response=client.post("/ingest/upload",files={"file":("a.txt",b"content")})
    assert response.status_code == 201
    assert response.json()["provider"] == "pinecone"
    index.assert_not_called()


def test_api_reads_at_most_limit_plus_one_and_closes_file(monkeypatch):
    monkeypatch.setattr(ingest_routes, "AppSettings", lambda: SimpleNamespace(max_upload_bytes=4))
    process = Mock()
    monkeypatch.setattr(ingest_routes, "process_document", process)
    file = UploadFile(file=BytesIO(b"oversized"), filename="a.txt")
    with pytest.raises(HTTPException) as error:
        asyncio.run(ingest_routes.upload_file(file, provider="pinecone"))
    assert error.value.status_code == 413
    assert file.file.closed
    process.assert_not_called()
