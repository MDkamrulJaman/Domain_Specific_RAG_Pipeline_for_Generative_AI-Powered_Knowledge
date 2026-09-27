"""HTTP upload boundary; ingestion logic lives in the service layer."""
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.core.config import AppSettings
from app.schemas.chat import Provider
from app.services.ingestion_service import process_document

ingest_router = APIRouter(prefix="/ingest", tags=["Ingest"])


@ingest_router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document without saving it locally"
)
async def upload_file(file: UploadFile = File(...), provider: Provider = Form("pinecone")):
    try:
        await file.seek(0)
        limit = AppSettings().max_upload_bytes
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(413, "Document exceeds the configured upload size limit.")
        return await asyncio.to_thread(process_document, file.filename, content, provider)
    finally:
        await file.close()
