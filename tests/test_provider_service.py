"""Provider factories, configuration isolation, and readiness checks."""

from types import SimpleNamespace
from unittest.mock import Mock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from app.services import assistant_service, provider_service
from fastapi import HTTPException
from app.services import provider_service


def test_assistant_factory_reuses_cached_service(monkeypatch):
    provider_service.get_chat_service.cache_clear()
    service = Mock()
    monkeypatch.setattr(assistant_service, "AssistantService", lambda **kwargs: service)
    assert provider_service.get_chat_service("pinecone") is service
    assert provider_service.get_chat_service("pinecone") is service
    provider_service.get_chat_service.cache_clear()


def test_provider_configuration_never_exposes_keys(monkeypatch):
    from app.core import config
    settings = SimpleNamespace(PINECONE_ASSISTANT_API_KEY="private-key", PINECONE_ASSISTANT_NAME="", PINECONE_ASSISTANT_MODEL="model")
    monkeypatch.setattr(config, "AssistantSettings", lambda: settings)
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


def test_assistant_settings_do_not_load_vector_settings(monkeypatch):
    from app.core import config
    monkeypatch.setattr(config,"AssistantSettings",lambda:SimpleNamespace(PINECONE_ASSISTANT_API_KEY="key",PINECONE_ASSISTANT_NAME="name",PINECONE_ASSISTANT_MODEL="model"))
    assert provider_service.provider_configuration("pinecone")["configured"]


def test_connection_check_preserves_configuration_error(monkeypatch):
    monkeypatch.setattr(provider_service, "provider_configuration", lambda _: {"configured": True})
    def fail(_):
        raise HTTPException(503, detail="Assistant unavailable.")
    monkeypatch.setattr(provider_service, "get_chat_service", fail)
    result = provider_service.inspect_provider("pinecone")
    assert result["connected"] is False
    assert result["message"] == "Assistant unavailable."
