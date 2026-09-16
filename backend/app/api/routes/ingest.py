import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status

# Pipeline imports (Keep these exactly as they are configured in your app)
from app.pipeline.loader import UniversalDocumentLoader
from app.pipeline.chunker import DocumentSplitter
from app.pipeline.embedder import EmbeddingService
from app.pipeline.retrieval_service import VectorStore

# Setup logging
logger = logging.getLogger(__name__)

ingest_router = APIRouter(prefix="/ingest", tags=["Ingest"])

# Test: Ensure directory exists safely
upload_dir = Path("data/raw")
upload_dir.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Initialize VectorStore with clean fallback error handling
try:
    vectorstore = VectorStore()
except Exception as e:
    logger.critical(f"Failed to initialize VectorStore: {str(e)}")
    raise RuntimeError("VectorStore initialization failed.") from e

#insest routes
@ingest_router.post(
    "/upload", 
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document"
)
async def upload_file(file: UploadFile = File(...) ):
    # 1. Validate Filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="File must have a valid filename."
        )

    # Secure the file path configuration
    file_path = upload_dir / Path(file.filename).name
    file_ext = file_path.suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File upload blocked: Unsupported extension '{file_ext}'")
        raise HTTPException(  
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # 2. Save file to disk using FastAPI
        try:
            # Ensure we are at the start of the file stream before reading
            await file.seek(0)
            content = await file.read()
            if content:
                  with open(file_path, "wb") as buffer:
                      buffer.write(content)
  
                
        except Exception as e:
            logger.error(f"File system write failed for {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save the uploaded file to disk securely."
            )

        # 3. Load Document
        try:
            loader_instance = UniversalDocumentLoader(str(upload_dir))
            docs = loader_instance.load_file(file_path)
            
            if not docs:
                raise ValueError("Document loader returned empty data.")
       
                
        except Exception as e:
            logger.error(f"Document parsing failed for {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not parse document content: {str(e)}"
            )

        # 4. Chunk Document
        try:
            chunks_instance = DocumentSplitter()
            chunks = chunks_instance.chunk_documents(docs)
            if not chunks:
                raise ValueError("Splitting resulted in 0 chunks.")
        except Exception as e:
            logger.error(f"Document chunking failed for {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to split the document into digestible text chunks."
            )

        # 5. Create Hugging Face vectors, then store them in Pinecone.
        try:
            embeddings = EmbeddingService().embed_chunks(chunks)
            vectorstore.add(embeddings, chunks)
            vectorstore.save()
        except Exception as e:
            logger.critical(f"VectorStore write operation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error: Failed to commit embeddings to the vector store."
            )

        return {
            "status": "success",
            "message": f"Successfully indexed '{file.filename}'",
            "chunks_created": len(chunks)
        }

    except HTTPException:
        # If an explicit HTTP exception happened, clean up the file and re-raise it
        if file_path.exists():
            file_path.unlink()
        raise

    except Exception as general_error:
        # Fallback catch-all for completely unexpected errors
        logger.exception(f"Unexpected system error processing file {file.filename}: {str(general_error)}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing your document."
        )
    finally:
        # Always close the file stream cleanly
        await file.close()