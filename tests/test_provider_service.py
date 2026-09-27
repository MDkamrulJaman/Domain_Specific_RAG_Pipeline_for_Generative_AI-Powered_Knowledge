"""Provider factories, configuration isolation, and readiness checks."""

from types import SimpleNamespace
from unittest.mock import Mock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from app.services import assistant_service, provider_service
from fastapi import HTTPException
from app.services import provider_service


def test_assistant_factory_never_initializes_vector_store(monkeypatch):
    provider_service.get_chat_service.cache_clear()
    vector = Mock(side_effect=AssertionError("Assistant must not retrieve from rag"))
    monkeypatch.setattr(provider_service, "get_vectorstore", vector)
    service = Mock()
    monkeypatch.setattr(assistant_service, "AssistantService", lambda: service)
    assert provider_service.get_chat_service("pinecone") is service
    vector.assert_not_called()
    provider_service.get_chat_service.cache_clear()


def test_provider_configuration_never_exposes_keys(monkeypatch):
    from app.core import config
    settings = SimpleNamespace(PINECONE_ASSISTANT_API_KEY="private-key", PINECONE_ASSISTANT_NAME="", PINECONE_ASSISTANT_MODEL="model")
    monkeypatch.setattr(config, "AssistantSettings", lambda: settings)
    monkeypatch.setattr(config, "RetrievalSettings", lambda: SimpleNamespace(PINECONE_API_KEY="vector-secret", PINECONE_INDEX_NAME="index", PINECONE_MODEL="rerank", HF_TOKEN="hf-secret", EMBEDDING_MODEL="embeddings"))
    result = provider_service.provider_configuration("pinecone")
    assert not result["configured"]
    assert "PINECONE_ASSISTANT_NAME" in result["message"]
    assert "private-key" not in str(result)


def test_provider_status_api(monkeypatch):
    from app.api.routes import providers
    app = FastAPI()
    app.include_router(providers.providers_router)
    monkeypatch.setattr(providers, "provider_configuration", lambda p: {"provider": p, "configured": False})
    check = Mock(return_value={"provider": "pinecone", "connected": True})
    monkeypatch.setattr(providers, "inspect_provider", check)
    with TestClient(app) as client:
        assert client.get("/providers/pinecone").json()["configured"] is False
        check.assert_not_called()
        assert client.get("/providers/pinecone?check_connection=true").json()["connected"] is True
        assert client.get("/providers/other").status_code == 422


def test_assistant_refresh_checks_own_files(monkeypatch):
    monkeypatch.setattr(provider_service, "provider_configuration", lambda p: {"configured":True,"provider":p})
    service = Mock()
    service.inspect_status.return_value = "Ready"
    service.list_files.return_value = [{"file_id":"f","name":"a.txt","status":"Processing"}]
    monkeypatch.setattr(provider_service, "get_chat_service", lambda _: service)
    result = provider_service.inspect_provider("pinecone")
    assert result["connected"]
    assert result["files"][0]["status"] == "Processing"
    assert "0/1" in result["message"]
    service.vectorstore.index.describe_index_stats.assert_not_called()


def test_shared_settings_do_not_require_generation_credentials():
    from app.core.config import RetrievalSettings
    settings = RetrievalSettings(_env_file=None, EMBEDDING_MODEL="embedding", HF_TOKEN="hf", PINECONE_API_KEY="vector", PINECONE_INDEX_NAME="index", PINECONE_DIMENSION=3, PINECONE_NAMESPACE="shared", PINECONE_MODEL="rerank")
    assert settings.PINECONE_API_KEY == "vector"
    assert "MODEL_API_KEY" not in RetrievalSettings.model_fields
    assert "PINECONE_ASSISTANT_API_KEY" not in RetrievalSettings.model_fields


def test_assistant_settings_do_not_load_vector_settings(monkeypatch):
    from app.core import config
    monkeypatch.setattr(config,"RetrievalSettings",Mock(side_effect=AssertionError("No index required")))
    monkeypatch.setattr(config,"AssistantSettings",lambda:SimpleNamespace(PINECONE_ASSISTANT_API_KEY="key",PINECONE_ASSISTANT_NAME="name",PINECONE_ASSISTANT_MODEL="model"))
    assert provider_service.provider_configuration("pinecone")["configured"]


def test_connection_check_preserves_configuration_error(monkeypatch):
    monkeypatch.setattr(provider_service, "provider_configuration", lambda _: {"configured": True})
    def fail(_):
        raise HTTPException(503, detail="Select a dense index of dimension 1024.")
    monkeypatch.setattr(provider_service, "get_chat_service", fail)
    result = provider_service.inspect_provider("nvidia")
    assert result["connected"] is False
    assert result["message"] == "Select a dense index of dimension 1024."
