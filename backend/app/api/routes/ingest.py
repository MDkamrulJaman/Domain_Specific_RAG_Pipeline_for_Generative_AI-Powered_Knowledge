import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.pipeline.loader import UniversalDocumentLoader
from app.pipeline.chunker import DocumentSplitter
from app.pipeline.embedder import EmbeddingService
from app.pipeline.retrieval_service import VectorStore

logger = logging.getLogger(__name__)

ingest_router = APIRouter(prefix="/ingest", tags=["Ingest"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

vectorstore = None


def get_vectorstore() -> VectorStore:
    """Create the vector store only when an upload actually needs it."""
    global vectorstore
    if vectorstore is None:
        try:
            vectorstore = VectorStore()
        except Exception as error:
            logger.exception("Failed to initialize VectorStore")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store is currently unavailable.",
            ) from error
    return vectorstore


@ingest_router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document without saving it locally"
)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File must have a valid filename."
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File upload blocked: Unsupported extension '{file_ext}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        await file.seek(0)
        content = await file.read()
        logger.info("Starting ingestion for '%s' (%d bytes)", file.filename, len(content))
        if not content:
            raise ValueError("Uploaded file is empty.")

        loader_instance = UniversalDocumentLoader()
        docs = loader_instance.load_bytes(file.filename, content)
        logger.info("Loaded %d document(s) from '%s'", len(docs), file.filename)
        if not docs:
            raise ValueError("Document loader returned empty data.")

        chunks_instance = DocumentSplitter()
        chunks = chunks_instance.chunk_documents(docs)
        logger.info("Created %d chunk(s) for '%s'", len(chunks), file.filename)
        if not chunks:
            raise ValueError("Splitting resulted in 0 chunks.")

        embeddings = EmbeddingService().embed_chunks(chunks)
        if embeddings.size == 0:
            raise ValueError("Embedding service returned no embeddings.")
        logger.info("Generated %d embedding(s) for '%s'", len(embeddings), file.filename)

        store = get_vectorstore()
        store.add(embeddings, chunks)
        store.save()
        logger.info("Finished ingestion for '%s'", file.filename)

        return {
            "status": "success",
            "message": f"Successfully indexed '{file.filename}'",
            "chunks_created": len(chunks),
        }

    except HTTPException:
        raise
    except Exception as general_error:
        logger.exception(f"Unexpected system error processing file {file.filename}: {general_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing your document."
        ) from general_error
    finally:
        await file.close()