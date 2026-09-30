"""Chat endpoint routing, streaming errors, and per-request options."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi import HTTPException
from app.api.routes import chat


@pytest.mark.parametrize("provider", ["pinecone"])
def test_chat_routes_selected_provider(client, monkeypatch, provider):
    service = Mock()
    service.stream.return_value = iter(["Hello", " world"])
    factory = Mock(return_value=service)
    monkeypatch.setattr(chat, "get_chat_service", factory)
    response = client.post("/chat/stream", json={"query": "question", "provider": provider})
    assert response.status_code == 200
    assert response.text == "Hello world"
    factory.assert_called_once_with(provider)
    service.stream.assert_called_once_with("question", web_search=False, skill="general")


def test_invalid_provider_rejected(client):
    assert client.post("/chat/stream", json={"query": "q", "provider": "unknown"}).status_code == 422
    assert client.post("/ingest/upload", data={"provider": "unknown"}, files={"file": ("a.txt", b"abc")}).status_code == 422


def test_missing_assistant_configuration(client, monkeypatch):
    def missing(_):
        raise HTTPException(503, "Set PINECONE_ASSISTANT_NAME")
    monkeypatch.setattr(chat, "get_chat_service", missing)
    response = client.post("/chat/stream", json={"query": "q", "provider": "pinecone"})
    assert response.status_code == 503
    assert "PINECONE_ASSISTANT_NAME" in response.text


def test_midstream_failure_preserves_partial_answer(client, monkeypatch):
    def broken(*args, **kwargs):
        yield "Partial answer"
        raise RuntimeError("private provider details")
    monkeypatch.setattr(chat, "get_chat_service", lambda _: SimpleNamespace(stream=broken))
    response = client.post("/chat/stream", json={"query": "q"})
    assert response.text.startswith("Partial answer")
    assert "interrupted" in response.text
    assert "private provider details" not in response.text
