"""Pinecone Assistant chat requests and asynchronous file processing."""

from types import SimpleNamespace
from unittest.mock import Mock
from app.core.config import AssistantSettings
from app.services import assistant_service


def test_assistant_chat_has_no_exclusion_filter(monkeypatch):
    settings = AssistantSettings(_env_file=None, PINECONE_ASSISTANT_API_KEY="assistant-key", PINECONE_ASSISTANT_NAME="manuals", PINECONE_ASSISTANT_MODEL="gpt-4o", PINECONE_ASSISTANT_TIMEOUT_SECONDS=60)
    monkeypatch.setattr(assistant_service, "AssistantSettings", lambda: settings)
    sdk = Mock()
    sdk.assistants.chat.return_value.text.return_value = iter(["answer"])
    monkeypatch.setattr(assistant_service, "Pinecone", lambda **kw: sdk)
    service = assistant_service.AssistantService()
    assert list(service.stream("question")) == ["answer"]
    options = sdk.assistants.chat.call_args.kwargs
    assert options["messages"] == [{"role":"user", "content":"question"}]
    assert "filter" not in options


def test_assistant_upload_returns_processing_without_waiting():
    service = assistant_service.AssistantService.__new__(assistant_service.AssistantService)
    service.name = "assistant"
    service.client = Mock()
    def upload(**kw):
        assert kw["file_stream"].read() == b"content"
        assert kw["file_name"] == "a.txt"
        assert kw["timeout"] == -1
        return SimpleNamespace(id="f",name="a.txt",status="Processing")
    service.client.assistants.upload_file.side_effect = upload
    assert service.upload("a.txt",b"content")["status"] == "Processing"
