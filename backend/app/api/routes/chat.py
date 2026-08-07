import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest
from app.services.rag_service import RAGService

# Setup logging
logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["Chat"])
rag_service = RAGService()


@chat_router.post(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Standard Chat",
    description="Processes user query via RAG and returns the complete generated response."
)
async def chat(req: ChatRequest):
    try:
        # If rag_service.ask is already async, simply use: await rag_service.ask(...)
        answer = await asyncio.to_thread(rag_service.ask, req.query, req.top_k)
        
        if not answer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Could not generate an answer based on the provided context."
            )

        return {
            "query": req.query,
            "answer": answer
        }

    except HTTPException as http_exc:
        # Re-raise managed HTTP exceptions
        raise http_exc
    except Exception as e:
        # Log the actual error securely on the server side
        logger.error(f"Error in /chat endpoint: {str(e)}", exc_info=True)
        # Return a clean, non-revealing error to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request."
        )


