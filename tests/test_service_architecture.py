"""Behavioral contracts for dependency injection and provider extension."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from app.application import create_app
from app.services.contracts import ignore_progress
from app.services.ingestion_service import process_document
from app.services.provider_registry import ProviderDefinition, ProviderRegistry


def test_custom_provider_upload_uses_same_workflow_without_builtin_branch():
    received = []
    def upload(filename, content, progress):
        received.append((filename, content))
        return {"status": "success", "provider": "custom"}
    registry = ProviderRegistry({"custom": ProviderDefinition(
        label="Custom", chat_factory=lambda: None, upload=upload,
        configuration=lambda: (None, (), "custom"), inspect=lambda _: {},
    )})
    result = process_document("a.txt", b"text", "custom", providers=registry)
    assert result["provider"] == "custom"
    assert received == [("a.txt", b"text")]


def test_unknown_provider_does_not_construct_any_adapter():
    registry = ProviderRegistry({})
    with pytest.raises(ValueError, match="Unsupported provider"):
        registry.resolve("missing")


def test_app_factory_builds_independent_api_instances_without_ui():
    first, second = create_app(include_ui=False), create_app(include_ui=False)
    assert first is not second
    with TestClient(first) as client:
        assert client.get("/health").json()["status"] == "healthy"
        assert "/chat/stream" in client.get("/openapi.json").json()["paths"]
