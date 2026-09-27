from functools import lru_cache
from fastapi import HTTPException

from app.schemas.chat import Provider


@lru_cache(maxsize=1)
def get_vectorstore():
    from app.pipeline.retrieval_service import VectorStore
    return VectorStore()


@lru_cache(maxsize=2)
def get_chat_service(provider: Provider = "pinecone"):
    if provider not in ("nvidia", "pinecone"):
        raise ValueError("Unsupported chat provider.")
    if provider == "pinecone":
        from app.services.assistant_service import AssistantService
        return AssistantService()
    from app.services.rag_service import RAGService
    return RAGService(provider=provider, vectorstore=get_vectorstore())


PROVIDER_LABELS = {"nvidia": "NVIDIA model", "pinecone": "Pinecone Assistant"}


def provider_configuration(provider: Provider):
    """Public configuration summary. Never return credentials or validation inputs."""
    from pydantic import ValidationError
    from app.core.config import Settings, AssistantSettings, RetrievalSettings
    if provider not in PROVIDER_LABELS:
        raise ValueError("Unsupported chat provider.")
    result = {
        "provider": provider, "label": PROVIDER_LABELS[provider],
        "configured": False, "model": "", "enable_thinking": False,
    }
    try:
        retrieval = RetrievalSettings() if provider == "nvidia" else None
        settings = AssistantSettings() if provider == "pinecone" else Settings()
    except ValidationError as exc:
        missing = sorted({str(error["loc"][0]) for error in exc.errors()})
        result["message"] = "Setup required. Check these backend environment settings: " + ", ".join(missing)
        return result
    shared_fields = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_MODEL", "HF_TOKEN", "EMBEDDING_MODEL"]
    fields = (["PINECONE_ASSISTANT_API_KEY", "PINECONE_ASSISTANT_NAME", "PINECONE_ASSISTANT_MODEL"]
              if provider == "pinecone" else ["MODEL_BASE_URL", "MODEL_NAME", "MODEL_API_KEY"])
    missing = [name for name in shared_fields if not str(getattr(retrieval, name)).strip()] if retrieval else []
    missing += [name for name in fields if not str(getattr(settings, name)).strip()]
    result["model"] = settings.PINECONE_ASSISTANT_MODEL if provider == "pinecone" else settings.MODEL_NAME
    result["enable_thinking"] = False if provider == "pinecone" else settings.MODEL_ENABLE_THINKING
    result["configured"] = not missing
    result["message"] = (
        "Setup required. Set " + ", ".join(missing) + " in the backend environment, then restart the app."
        if missing else "Configuration loaded. Connection has not been checked."
    )
    return result


def inspect_provider(provider: Provider):
    """Read remote readiness on demand without generating a model response."""
    import logging
    result = provider_configuration(provider)
    result.update(connected=False, files=[])
    if not result["configured"]:
        return result
    try:
        service = get_chat_service(provider)
        if provider == "pinecone":
            state = service.inspect_status()
            files = service.list_files()
            available = sum(file["status"].lower() == "available" for file in files)
            result.update(connected=True, assistant_status=state, files=files)
            result["message"] = f"Assistant: {state}. {available}/{len(files)} files available. "
            if not available:
                result["message"] += "Upload files and wait for processing before chatting."
        else:
            stats = service.vectorstore.index.describe_index_stats()
            namespace = stats.namespaces.get(service.vectorstore.namespace)
            count = namespace.vector_count if namespace else 0
            result.update(connected=True, vector_count=count)
            result["message"] = f"Pinecone index '{service.vectorstore.index_name}' connected: {count} records. "
            mode = "Pinecone integrated embeddings" if service.vectorstore.integrated_embedding else "Hugging Face embeddings"
            result["message"] += f"Retrieval uses {mode}; NVIDIA generates answers."
    except HTTPException as exc:
        result["connected"] = False
        result["message"] = str(exc.detail)
    except Exception:
        result["connected"] = False
        logging.getLogger(__name__).exception("Provider readiness check failed for %s", provider)
        result["message"] = (
            "Connection check failed. Check the provider credentials, assistant or index name, "
            "and network connection. See server logs for details."
        )
    return result
