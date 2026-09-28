"""Application factory; transport assembly is separate from service logic."""
from fastapi import FastAPI
from app.api.app_router import backend_router


# Simple application factory: assembles dependencies without a global Singleton.
def create_app(*, include_ui: bool = True) -> FastAPI:
    app = FastAPI(title="RAG Technical Document Assistant", description="Upload Files and Manage RAG Workflows", version="1.0.0")
    app.include_router(backend_router)

    @app.get("/api", tags=["Root"])
    async def root():
        return {"message": "Welcome to the RAG Technical Document Assistant"}

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "healthy", "version": app.version}

    if include_ui:
        from app.ui.frontend import create_demo, mount_demo
        app = mount_demo(app, create_demo())
    return app
