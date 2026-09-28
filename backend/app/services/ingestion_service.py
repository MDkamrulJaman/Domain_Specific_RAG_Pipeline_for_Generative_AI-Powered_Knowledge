"""Document validation and provider-specific ingestion shared by API and UI."""
import logging
from pathlib import Path
from fastapi import HTTPException
from app.core.config import AppSettings
from app.services.provider_service import get_vectorstore, get_chat_service, registry
from app.services.contracts import Progress, ignore_progress

logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def EmbeddingService():
    from app.pipeline.embedder import get_embedding_service
    return get_embedding_service()


def process_document(filename, content, provider="pinecone", progress: Progress = ignore_progress, *, providers=None):
    providers = providers if providers is not None else registry
    try:
        # Strategy: resolve behavior once; validation is shared across providers.
        uploader = providers.resolve(provider).upload
    except ValueError as exc:
        raise HTTPException(422, "Choose a supported upload provider.") from exc
    if not filename:
        raise HTTPException(422, "File must have a valid filename.")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only PDF and TXT documents are supported.")
    if len(content) > AppSettings().max_upload_bytes:
        raise HTTPException(413, "Document exceeds the configured upload size limit.")
    if not content:
        raise HTTPException(400, "The selected document is empty.")
    try:
        return uploader(filename, content, progress)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, "The document could not be indexed. Check its text and the embedding configuration.") from exc
    except Exception as exc:
        logger.exception("Document ingestion failed for %s", provider)
        raise HTTPException(502, "Upload could not complete. Check the provider connection and refresh the document library before retrying.") from exc


def upload_to_assistant(filename, content, progress):
    progress(0.2, "Uploading to Pinecone Assistant")
    uploaded = get_chat_service("pinecone").upload(Path(filename).name, content)
    return {"status": "success", "storage": "pinecone_assistant", "provider": "pinecone",
            "results": {"pinecone": uploaded}, "message": f"Assistant: {uploaded['status']}"}


def upload_to_nvidia(filename, content, progress):
    result = index_document(filename, content, progress)
    result.update(provider="nvidia", results={"nvidia": {"status": "Indexed", "chunks_created": result["chunks_created"]}})
    return result


def index_document(filename: str, content: bytes, progress=lambda *args: None):
    from app.pipeline.loader import UniversalDocumentLoader
    from app.pipeline.chunker import DocumentSplitter
    from app.services.indexing_service import IndexingService
    store = get_vectorstore()
    service = IndexingService(store, UniversalDocumentLoader(), DocumentSplitter(), EmbeddingService)
    return service.index(filename, content, progress)
