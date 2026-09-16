from fastapi import APIRouter

from app.api.routes.chat import chat_router
from app.api.routes.ingest import ingest_router


def build_backend_router(*extra_routers: APIRouter) -> APIRouter:
    router = APIRouter()
    router.include_router(chat_router)
    router.include_router(ingest_router)

    for extra_router in extra_routers:
        router.include_router(extra_router)

    return router


backend_router = build_backend_router()

# Backwards-compatible names used by the existing route tests and integrations.
api_router = backend_router
build_api_router = build_backend_router

__all__ = [
    "api_router",
    "backend_router",
    "build_api_router",
    "build_backend_router",
    "chat_router",
    "ingest_router",
]
