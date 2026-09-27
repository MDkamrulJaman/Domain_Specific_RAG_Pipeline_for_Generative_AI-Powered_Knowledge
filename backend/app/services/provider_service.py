"""Composition root: wire concrete adapters and expose provider use cases.

SDK construction belongs here or in adapters, never inside RAG/indexing workflows.
"""
import logging
from functools import lru_cache
from fastapi import HTTPException
from pydantic import ValidationError
from app.schemas.chat import Provider
from app.services.contracts import ChatService
from app.services.provider_registry import ProviderDefinition, ProviderRegistry


@lru_cache(maxsize=1)
def get_vectorstore():
    from app.pipeline.retrieval_service import VectorStore
    return VectorStore()


def _nvidia_chat():
    from app.services.llm_service import LLMService
    from app.services.rag_service import RAGService
    return RAGService(vectorstore=get_vectorstore(), llm=LLMService())


def _assistant_chat():
    from app.services.assistant_service import AssistantService
    return AssistantService()


def _nvidia_upload(filename, content, progress):
    from app.services.ingestion_service import upload_to_nvidia
    return upload_to_nvidia(filename, content, progress)


def _assistant_upload(filename, content, progress):
    from app.services.ingestion_service import upload_to_assistant
    return upload_to_assistant(filename, content, progress)


def _nvidia_configuration():
    from app.core.config import Settings
    settings = Settings()
    fields = ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_MODEL", "HF_TOKEN",
              "EMBEDDING_MODEL", "MODEL_BASE_URL", "MODEL_NAME", "MODEL_API_KEY")
    return settings, fields, settings.MODEL_NAME, settings.MODEL_ENABLE_THINKING


def _assistant_configuration():
    from app.core.config import AssistantSettings
    settings = AssistantSettings()
    fields = ("PINECONE_ASSISTANT_API_KEY", "PINECONE_ASSISTANT_NAME", "PINECONE_ASSISTANT_MODEL")
    return settings, fields, settings.PINECONE_ASSISTANT_MODEL, False


def _assistant_status(service):
    state = service.inspect_status()
    files = service.list_files()
    available = sum(file["status"].lower() == "available" for file in files)
    message = f"Assistant: {state}. {available}/{len(files)} files available. "
    if not available:
        message += "Upload files and wait for processing before chatting."
    return {"connected": True, "assistant_status": state, "files": files, "message": message}


def _nvidia_status(service):
    store = service.vectorstore
    namespace = store.index.describe_index_stats().namespaces.get(store.namespace)
    count = namespace.vector_count if namespace else 0
    mode = "Pinecone integrated embeddings" if store.integrated_embedding else "Hugging Face embeddings"
    return {"connected": True, "vector_count": count,
            "message": f"Pinecone index '{store.index_name}' connected: {count} records. "
                       f"Retrieval uses {mode}; NVIDIA generates answers."}


registry = ProviderRegistry({
    "pinecone": ProviderDefinition("Pinecone Assistant", _assistant_chat, _assistant_upload,
                                   _assistant_configuration, _assistant_status),
    "nvidia": ProviderDefinition("NVIDIA model", _nvidia_chat, _nvidia_upload,
                                 _nvidia_configuration, _nvidia_status),
})
PROVIDER_LABELS = registry.labels


@lru_cache(maxsize=2)
def get_chat_service(provider: Provider = "pinecone") -> ChatService:
    return registry.resolve(provider).chat_factory()


def provider_configuration(provider: Provider):
    """Return a credential-free summary without opening remote connections."""
    definition = registry.resolve(provider)
    result = {"provider": provider, "label": definition.label, "configured": False,
              "model": "", "enable_thinking": False}
    try:
        settings, fields, model, thinking = definition.configuration()
    except ValidationError as exc:
        missing = sorted({str(error["loc"][0]) for error in exc.errors()})
        result["message"] = "Setup required. Check these backend environment settings: " + ", ".join(missing)
        return result
    missing = [name for name in fields if not str(getattr(settings, name)).strip()]
    result.update(model=model, enable_thinking=thinking, configured=not missing)
    result["message"] = (
        "Setup required. Set " + ", ".join(missing) + " in the backend environment, then restart the app."
        if missing else "Configuration loaded. Connection has not been checked."
    )
    return result


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
