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

try:
    vectorstore = VectorStore()
except Exception as e:
    logger.critical(f"Failed to initialize VectorStore: {str(e)}")
    raise RuntimeError("VectorStore initialization failed.") from e


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
        if not content:
            raise ValueError("Uploaded file is empty.")

        loader_instance = UniversalDocumentLoader()
        docs = loader_instance.load_bytes(file.filename, content)
        if not docs:
            raise ValueError("Document loader returned empty data.")

        chunks_instance = DocumentSplitter()
        chunks = chunks_instance.chunk_documents(docs)
        if not chunks:
            raise ValueError("Splitting resulted in 0 chunks.")

        embeddings = EmbeddingService().embed_chunks(chunks)
        vectorstore.add(embeddings, chunks)
        vectorstore.save()

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