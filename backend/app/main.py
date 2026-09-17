import logging

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.api.app_router import backend_router
from app.ui.frontend import create_demo, mount_demo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_system")

app = FastAPI(
    title="RAG SYSTEM",
    description="Upload Files and Manage RAG Workflows",
    version="1.0.0",
)

app.include_router(backend_router)


@app.get("/api", tags=["Root"])
async def root():
    return {"message": "Welcome to the RAG SYSTEM API"}


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "version": app.version,
    }


gradio_demo = create_demo()
mount_demo(app, gradio_demo)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)