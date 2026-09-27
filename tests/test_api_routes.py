from app.api.app_router import api_router, build_api_router
from fastapi import APIRouter, FastAPI


def test_shared_api_router_exposes_expected_routes():
    app = FastAPI()
    app.include_router(api_router)
    paths = app.openapi()["paths"]

    assert "/chat/stream" in paths
    assert "/ingest/upload" in paths


def test_build_api_router_composes_custom_routers():
    extra_router = APIRouter(prefix="/health")

    @extra_router.get("/")
    def health():
        return {"status": "ok"}

    composed_router = build_api_router(extra_router)
    app = FastAPI()
    app.include_router(composed_router)
    paths = app.openapi()["paths"]

    assert "/health/" in paths
