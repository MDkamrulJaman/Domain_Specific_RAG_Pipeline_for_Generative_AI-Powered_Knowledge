from app.api.app_router import api_router, build_api_router
from fastapi import APIRouter


def test_shared_api_router_exposes_expected_routes():
    paths = {route.path for route in api_router.routes if hasattr(route, "path")}

    assert "/chat/" in paths
    assert "/chat/stream" in paths
    assert "/ingest/upload" in paths


def test_build_api_router_composes_custom_routers():
    extra_router = APIRouter(prefix="/health")

    @extra_router.get("/")
    def health():
        return {"status": "ok"}

    composed_router = build_api_router(extra_router)
    paths = {route.path for route in composed_router.routes if hasattr(route, "path")}

    assert "/health/" in paths
