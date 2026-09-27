"""Document validation and provider-specific ingestion shared by API and UI."""
import logging
from pathlib import Path
from fastapi import HTTPException
from app.core.config import AppSettings
from app.services.provider_service import get_vectorstore, get_chat_service

logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def EmbeddingService():
    from app.pipeline.embedder import get_embedding_service
    return get_embedding_service()


def process_document(filename, content, provider="pinecone", progress=lambda *args: None):
    if provider not in ("nvidia", "pinecone"):
        raise HTTPException(422, "Choose NVIDIA or Pinecone Assistant.")
    if not filename:
        raise HTTPException(422, "File must have a valid filename.")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only PDF and TXT documents are supported.")
    if len(content) > AppSettings().max_upload_bytes:
        raise HTTPException(413, "Document exceeds the configured upload size limit.")
    if not content:
        raise HTTPException(400, "The selected document is empty.")
    try:
        if provider == "pinecone":
            progress(0.2, "Uploading to Pinecone Assistant")
            uploaded = get_chat_service("pinecone").upload(Path(filename).name, content)
            return {"status": "success", "storage": "pinecone_assistant", "provider": provider,
                    "results": {provider: uploaded}, "message": f"Assistant: {uploaded['status']}"}
        result = index_document(filename, content, progress)
        result.update(provider=provider, results={provider: {"status": "Indexed", "chunks_created": result["chunks_created"]}})
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, "The document could not be indexed. Check its text and the embedding configuration.") from exc
    except Exception as exc:
        logger.exception("Document ingestion failed for %s", provider)
        raise HTTPException(502, "Upload could not complete. Check the provider connection and refresh the document library before retrying.") from exc


def index_document(filename: str, content: bytes, progress=lambda *args: None):
    from app.pipeline.loader import UniversalDocumentLoader
    from app.pipeline.chunker import DocumentSplitter

    # Validate storage before spending time or API calls on embedding.
    store = get_vectorstore()
    progress(0.15, "Extracting document text")
    loader_instance = UniversalDocumentLoader()
    docs = loader_instance.load_bytes(filename, content)
    logger.info("Loaded %d document(s) from '%s'", len(docs), filename)
    if not docs:
        raise ValueError("Document loader returned empty data.")

    progress(0.3, "Splitting document into chunks")
    chunks_instance = DocumentSplitter()
    chunks = chunks_instance.chunk_documents(docs)
    logger.info("Created %d chunk(s) for '%s'", len(chunks), filename)
    if not chunks:
        raise ValueError("Splitting resulted in 0 chunks.")

    embeddings = None
    if getattr(store, "integrated_embedding", False) is True:
        progress(0.5, "Pinecone will embed the document text")
    else:
        progress(0.5, "Generating Hugging Face embeddings")
        embeddings = EmbeddingService().embed_chunks(chunks)
        if embeddings.size == 0:
            raise ValueError("Embedding service returned no embeddings.")
        logger.info("Generated %d embedding(s) for '%s'", len(embeddings), filename)

    progress(0.8, "Saving to the shared Pinecone index")
    store.add(embeddings, chunks)
    store.save()
    logger.info("Finished ingestion for '%s'", filename)

    return {
        "status": "success",
        "storage": "shared_pinecone",
        "message": f"Successfully indexed '{filename}'",
        "chunks_created": len(chunks),
    }
