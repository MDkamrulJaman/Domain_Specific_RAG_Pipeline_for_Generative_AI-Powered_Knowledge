import asyncio
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest
from app.services.rag_service import RAGService

# Setup logging
logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["Chat"])
_rag_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@chat_router.post("/stream", summary="Streaming-compatible chat")
async def stream_chat(req: ChatRequest):
    """Return the answer through a streaming response for Gradio-style clients."""
    try:
        answer = await asyncio.to_thread(
            get_rag_service().ask, req.query, req.top_k
        )
    except Exception as exc:
        logger.error("Error in /chat/stream endpoint: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request.",
        ) from exc

    async def answer_stream() -> AsyncGenerator[str, None]:
        yield answer

    return StreamingResponse(answer_stream(), media_type="text/plain")


