import logging
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile

load_dotenv()

from app.api.app_router import backend_router
from app.api.routes.ingest import upload_file
from app.services.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_system")

_rag_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

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


def rag_answer(message, _history):
    if not message.strip():
        return "Please enter a question about your knowledge base."
    try:
        return get_rag_service().ask(message)
    except Exception:
        logger.exception("Gradio RAG request failed")
        return "The RAG pipeline could not process your question. Check the backend logs."


def vote(data: gr.LikeData):
    value = (
        data.value.get("value", "")
        if isinstance(data.value, dict)
        else str(data.value)
    )
    logger.info(
        "RAG response %s: %s",
        "liked" if data.liked else "disliked",
        value[:200],
    )


async def ingest_file(file_path: str | None):
    if not file_path:
        return "Select a PDF or TXT file first."

    path = Path(file_path)
    if not path.is_file():
        return "The selected file could not be found."

    try:
        with path.open("rb") as file:
            upload = UploadFile(file=file, filename=path.name)
            result = await upload_file(upload)
        return result["message"]
    except Exception:
        logger.exception("Document ingestion failed for %s", path.name)
        return "Document ingestion failed. Check the backend logs."


with gr.Blocks(
    title="RAG System",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
) as demo:
    gr.HTML(
        """
        <nav class="navbar">
            <div class="navbar-brand">RAG System</div>
            <div class="navbar-links">
                <a href="/">Pinecone Assistant</a>
                <a href="/docs">API Docs</a>
                <a href="/health">System Health</a>
            </div>
        </nav>
        """
    )
    with gr.Row(equal_height=False):
        with gr.Sidebar(open=True, width=260, elem_classes=["sidebar"]):
            gr.Markdown("### Navigation", elem_classes=["sidebar-title"])
            upload = gr.File(
                label="Upload document",
                file_types=[".pdf", ".txt"],
                type="filepath",
            )
            upload_button = gr.Button("Upload and index", variant="primary")
            upload_status = gr.Markdown()
            upload_button.click(
                ingest_file,
                inputs=upload,
                outputs=upload_status,
            )
            gr.Markdown(
                "Use the assistant to search your indexed engineering knowledge base.\n\n",
                elem_classes=["sidebar-copy"],
            )
            gr.Markdown(
                "---\n_RAG SYSTEM v1.0.0_",
                elem_classes=["sidebar-copy"],
            )
        with gr.Column():
            gr.HTML(
                """
                <div class="hero">
                    <h1>RAG System</h1>
                    <p>Ask questions across your indexed engineering knowledge base.</p>
                </div>
                """
            )
            chatbot = gr.Chatbot(
                label="Knowledge assistant",
                height=520,
                placeholder=(
                    "<strong>Start a conversation</strong><br>"
                    "Ask about your indexed documents..."
                ),
            )
            chatbot.like(vote, None, None)
            gr.ChatInterface(
                fn=rag_answer,
                chatbot=chatbot,
                title="",
                description=(
                    "Answers are generated from your connected RAG pipeline."
                ),
                examples=[
                    "Summarize the latest uploaded document",
                    "What are the key system requirements?",
                ],
                submit_btn="Ask",
            )

gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)