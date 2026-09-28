import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest
from app.services.chat_command import AnswerCommand
from app.services.provider_service import get_chat_service

logger = logging.getLogger(__name__)
chat_router = APIRouter(prefix="/chat", tags=["Chat"])


def response_stream(service, req):
    try:
        # Command invoker: HTTP error formatting stays at the transport boundary.
        yield from AnswerCommand.from_request(req).execute(service)
    except HTTPException as exc:
        yield "\n\n" + str(exc.detail)
    except Exception:
        logger.exception("Chat stream failed for %s", req.provider)
        yield "\n\n[Response interrupted. Check the provider configuration and try again.]"


@chat_router.post("/stream", summary="Stream an answer from NVIDIA or Pinecone Assistant")
def stream_chat(req: ChatRequest):
    try:
        service = get_chat_service(req.provider)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat provider initialization failed")
        raise HTTPException(503, "The selected chat provider is unavailable.") from exc
    return StreamingResponse(
        response_stream(service, req),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
