"""Regression boundaries for the single-provider application."""
import ast
from pathlib import Path
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from app.application import create_app
from app.schemas.chat import ChatRequest
from app.services import provider_service


def test_removed_provider_rejected_before_initialization(monkeypatch):
    factory = Mock(side_effect=AssertionError("No SDK should be initialized"))
    monkeypatch.setattr("app.api.routes.chat.get_chat_service", factory)
    with TestClient(create_app(include_ui=False)) as client:
        assert client.post("/chat/stream", json={"query": "q", "provider": "nvidia"}).status_code == 422
        assert client.post("/ingest/upload", data={"provider": "nvidia"}, files={"file": ("a.txt", b"text")}).status_code == 422
        assert client.get("/providers/nvidia").status_code == 422
    factory.assert_not_called()
    with pytest.raises(ValueError):
        provider_service.get_chat_service("nvidia")


def test_only_assistant_options_are_exposed():
    assert provider_service.PROVIDER_LABELS == {"pinecone": "Pinecone Assistant"}
    assert "top_k" not in ChatRequest.model_fields
    assert "enable_thinking" not in ChatRequest.model_fields


def test_retained_helper_has_no_active_importers():
    app = Path(__file__).resolve().parents[1] / "backend" / "app"
    assert (app / "services" / "llm_service.py").is_file()
    for source in app.rglob("*.py"):
        if source.name == "llm_service.py":
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert all("llm_service" not in alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert "llm_service" not in (node.module or "")
                assert all(alias.name != "llm_service" for alias in node.names)


def test_library_refresh_is_wired_after_load_and_upload():
    from app.ui.frontend import create_demo
    from app.ui.handlers import refresh_provider
    demo = create_demo()
    refreshes = [fn for fn in demo.fns.values() if fn.fn is refresh_provider]
    assert len(refreshes) == 2
    assert all(any(getattr(output, "elem_id", None) == "document-library" for output in fn.outputs) for fn in refreshes)
    labels = {getattr(block, "label", None) for block in demo.blocks.values()}
    assert "Answer provider" not in labels
    assert "Enable NVIDIA thinking" not in labels
    assert {"Pinecone task", "Allow web search", "Your documents"} <= labels
