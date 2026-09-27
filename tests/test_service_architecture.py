"""Behavioral contracts for dependency injection and provider extension."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from app.application import create_app
from app.core.config import NvidiaSettings
from app.services.contracts import ignore_progress
from app.services.indexing_service import IndexingService
from app.services.ingestion_service import process_document
from app.services.provider_registry import ProviderDefinition, ProviderRegistry
from app.services.rag_service import RAGService


class MemoryRetriever:
    def search(self, query, top_k=5):
        return [{"text": "Document evidence"}]


class EchoGenerator:
    def stream(self, prompt, enable_thinking=None):
        yield prompt


def test_rag_accepts_substitute_retriever_and_generator_without_sdk_setup():
    service = RAGService(MemoryRetriever(), EchoGenerator())
    response = "".join(service.stream("What does the document say?"))
    assert "Document evidence" in response
    assert "What does the document say?" in response


def test_custom_provider_upload_uses_same_workflow_without_builtin_branch():
    received = []
    def upload(filename, content, progress):
        received.append((filename, content))
        return {"status": "success", "provider": "custom"}
    registry = ProviderRegistry({"custom": ProviderDefinition(
        label="Custom", chat_factory=lambda: None, upload=upload,
        configuration=lambda: (None, (), "custom", False), inspect=lambda _: {},
    )})
    result = process_document("a.txt", b"text", "custom", providers=registry)
    assert result["provider"] == "custom"
    assert received == [("a.txt", b"text")]


def test_indexing_only_requests_embedder_for_dense_writer():
    writer = Mock()
    writer.integrated_embedding = True
    loader = Mock()
    loader.load_bytes.return_value = ["document"]
    chunker = Mock()
    chunker.chunk_documents.return_value = ["chunk"]
    embedder = Mock(side_effect=AssertionError("Integrated writer does not need embeddings"))
    service = IndexingService(writer, loader, chunker, embedder)
    assert service.index("a.txt", b"text")["chunks_created"] == 1
    writer.add.assert_called_once_with(None, ["chunk"])
    embedder.assert_not_called()


def test_unknown_provider_does_not_construct_any_adapter():
    registry = ProviderRegistry({})
    with pytest.raises(ValueError, match="Unsupported provider"):
        registry.resolve("missing")


def test_nvidia_settings_do_not_require_vector_database_credentials():
    settings = NvidiaSettings(_env_file=None, MODEL_BASE_URL="https://example.invalid",
        MODEL_NAME="test", MODEL_API_KEY="test", MODEL_TIMEOUT_SECONDS=30,
        MODEL_TEMPERATURE=0.2, MODEL_TOP_P=0.95, MODEL_MAX_TOKENS=1024)
    assert settings.MODEL_NAME == "test"
    assert "PINECONE_API_KEY" not in NvidiaSettings.model_fields


def test_app_factory_builds_independent_api_instances_without_ui():
    first, second = create_app(include_ui=False), create_app(include_ui=False)
    assert first is not second
    with TestClient(first) as client:
        assert client.get("/health").json()["status"] == "healthy"
        assert "/chat/stream" in client.get("/openapi.json").json()["paths"]
