"""Composition root: wire concrete adapters and expose provider use cases.

SDK construction belongs here or in adapters, never inside application workflows.
"""
import logging
from functools import lru_cache
from fastapi import HTTPException
from pydantic import ValidationError
from app.schemas.chat import Provider
from app.services.contracts import ChatService
from app.services.provider_registry import ProviderDefinition, ProviderRegistry


def _assistant_chat():
    from app.services.assistant_service import AssistantService
    from app.services.web_search import TavilySearchAdapter
    return AssistantService(web_search_tool=TavilySearchAdapter())


def _assistant_upload(filename, content, progress):
    from app.services.ingestion_service import upload_to_assistant
    return upload_to_assistant(filename, content, progress)


def _assistant_configuration():
    from app.core.config import AssistantSettings
    settings = AssistantSettings()
    fields = ("PINECONE_ASSISTANT_API_KEY", "PINECONE_ASSISTANT_NAME", "PINECONE_ASSISTANT_MODEL")
    return settings, fields, settings.PINECONE_ASSISTANT_MODEL


def _assistant_status(service):
    state = service.inspect_status()
    files = service.list_files()
    available = sum(file["status"].lower() == "available" for file in files)
    message = f"Assistant: {state}. {available}/{len(files)} files available. "
    if not available:
        message += "Upload files and wait for processing before chatting."
    return {"connected": True, "assistant_status": state, "files": files, "message": message}


registry = ProviderRegistry({
    "pinecone": ProviderDefinition("Pinecone Assistant", _assistant_chat, _assistant_upload,
                                   _assistant_configuration, _assistant_status),
})
PROVIDER_LABELS = registry.labels


# Cached factory: process-local reuse, not a strict Singleton. Concurrent first
# calls may construct multiple clients; replicas never share this cache.
@lru_cache(maxsize=1)
def get_chat_service(provider: Provider = "pinecone") -> ChatService:
    return registry.resolve(provider).chat_factory()


def provider_configuration(provider: Provider):
    """Return a credential-free summary without opening remote connections."""
    definition = registry.resolve(provider)
    result = {"provider": provider, "label": definition.label, "configured": False,
              "model": ""}
    try:
        settings, fields, model = definition.configuration()
    except ValidationError as exc:
        missing = sorted({str(error["loc"][0]) for error in exc.errors()})
        result["message"] = "Setup required. Check these backend environment settings: " + ", ".join(missing)
        return result
    missing = [name for name in fields if not str(getattr(settings, name)).strip()]
    result.update(model=model, configured=not missing)
    result["message"] = (
        "Setup required. Set " + ", ".join(missing) + " in the backend environment, then restart the app."
        if missing else "Configuration loaded. Connection has not been checked."
    )
    return result


# Facade: expose one readiness operation over the provider-specific SDK calls.
def inspect_provider(provider: Provider):
    result = provider_configuration(provider)
    result.update(connected=False, files=[])
    if not result["configured"]:
        return result
    try:
        result.update(registry.resolve(provider).inspect(get_chat_service(provider)))
    except HTTPException as exc:
        result.update(connected=False, message=str(exc.detail))
    except Exception:
        logging.getLogger(__name__).exception("Provider readiness check failed for %s", provider)
        result.update(connected=False, message="Connection check failed. Check provider credentials "
                      "and network access. See server logs for details.")
    return result
