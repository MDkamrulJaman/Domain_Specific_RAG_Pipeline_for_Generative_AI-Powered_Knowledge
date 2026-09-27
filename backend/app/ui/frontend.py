import asyncio
import copy
import html
import logging
import time
from pathlib import Path

import gradio as gr
from fastapi import HTTPException

from app.api.routes.ingest import process_document
from app.services.provider_service import (
    PROVIDER_LABELS, get_chat_service, inspect_provider, provider_configuration,
)

logger = logging.getLogger("rag_system")


def status_markdown(config):
    label = html.escape(config["label"])
    state = "Configured" if config["configured"] else "Setup required"
    if "connected" in config and config["configured"]:
        state = "Connected" if config["connected"] else "Connection failed"
    model = html.escape(config.get("model", ""))
    return f"**{label} · {state}**\n\n{html.escape(config['message'])}\n\nModel: `{model or 'not configured'}`"


def library_rows(provider, receipts):
    rows = []
    for item in (receipts or {}).get("shared", []):
        results = item.get("results", {})
        if provider not in results:
            continue
        result = results[provider]
        state = result.get("status", "Unknown")
        if result.get("message"):
            state += ": " + result["message"]
        rows.append([item["name"], state])
    return rows


def provider_panel(provider, receipts):
    config = provider_configuration(provider)
    is_nvidia = provider == "nvidia"
    note = ("Your NVIDIA uploads from this session. Existing indexed documents remain searchable."
            if is_nvidia else "Assistant searches its separate file library. Refresh to check processing status.")
    return (
        status_markdown(config), gr.update(visible=is_nvidia),
        gr.update(value=config["enable_thinking"]),
        gr.update(value=f"Upload to {PROVIDER_LABELS[provider]}"),
        library_rows(provider, receipts), note,
        f"Files upload only to {PROVIDER_LABELS[provider]}.",
        f"{PROVIDER_LABELS[provider]} - Ready" if config["configured"] else f"{PROVIDER_LABELS[provider]} - Setup required",
        gr.update(visible=is_nvidia),
    )


def refresh_provider(provider, receipts):
    config = inspect_provider(provider)
    updated = copy.deepcopy(receipts or {})
    if provider == "pinecone" and config.get("connected"):
        entries = updated.setdefault("shared", [])
        for entry in entries:
            entry.get("results", {}).pop("pinecone", None)
        entries[:] = [entry for entry in entries if entry.get("results")]
        for file in config.get("files", []):
            entry = next((item for item in entries if item.get("results", {}).get("pinecone", {}).get("file_id") == file["file_id"]), None)
            if entry is None:
                entry = {"name": file["name"], "results": {}}
                entries.append(entry)
            entry.setdefault("results", {})["pinecone"] = file
    return status_markdown(config), updated, library_rows(provider, updated)


def rag_answer(message, _history, provider="pinecone", top_k=5, enable_thinking=False):
    label = PROVIDER_LABELS[provider]
    if not message.strip():
        yield "Please enter a question about your documents.", f"{label} · No question submitted"
        return
    answer = ""
    started = time.perf_counter()
    first_token = None
    yield answer, f"{label} · Retrieving context and waiting for the first token…"
    try:
        service = get_chat_service(provider)
        for fragment in service.stream(message, top_k=int(top_k), enable_thinking=enable_thinking):
            if not fragment:
                continue
            elapsed = time.perf_counter() - started
            if first_token is None:
                first_token = elapsed
            answer += fragment
            yield answer, f"{label} · Streaming · First token {first_token:.1f}s · Elapsed {elapsed:.1f}s"
        elapsed = time.perf_counter() - started
        if not answer:
            yield "The provider returned no answer. Try another question.", f"{label} · Empty response · {elapsed:.1f}s"
        else:
            yield answer, f"{label} · Response complete · First token {first_token:.1f}s · Total {elapsed:.1f}s"
    except HTTPException as exc:
        yield answer + "\n\n" + str(exc.detail), f"{label} · Request could not complete"
    except Exception:
        logger.exception("Gradio request failed for %s", provider)
        yield answer + "\n\nThe selected provider could not complete the answer. Check its connection and try again.", f"{label} · Response interrupted"


async def ingest_file(file_path, receipts=None, provider="pinecone", progress=gr.Progress()):
    updated = copy.deepcopy(receipts or {})
    if not file_path or not Path(file_path).is_file():
        return "Select a PDF or TXT file first.", updated, library_rows(provider, updated)
    path = Path(file_path)
    started = time.perf_counter()
    try:
        progress(0, desc="Reading selected document")
        content = await asyncio.to_thread(path.read_bytes)
        report = lambda value, description: progress(value, desc=description)
        result = await asyncio.to_thread(process_document, path.name, content, provider, progress=report)
        record = {"name": path.name, "results": result.get("results", {}), "status": result["message"]}
        entries = updated.setdefault("shared", [])
        entries.append(record)
        progress(1, desc="Upload finished")
        message = f"Upload results: {result['message']} ({time.perf_counter() - started:.1f}s)"
        return message, updated, library_rows(provider, updated)
    except HTTPException as exc:
        return str(exc.detail), updated, library_rows(provider, updated)
    except Exception:
        logger.exception("Document upload failed")
        return "Upload failed. Check the provider connection, then refresh the library before retrying.", updated, library_rows(provider, updated)


CSS = """
.gradio-container { max-width: 1440px !important; }
#app-header { padding: 16px 4px 24px; }
#app-header h1 { font-size: 30px; letter-spacing: -0.7px; margin-bottom: 6px; }
#app-header p { color: #64748b; }
#provider-status { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 14px; }
#response-status { min-height: 40px; padding: 10px 14px; border-radius: 10px; background: var(--background-fill-secondary); }
"""


def create_demo():
    default = provider_configuration("pinecone")
    with gr.Blocks(title="Generative AI-Powered Retrieval System for Technical Documentation⁠", analytics_enabled=False) as demo:
        receipts = gr.State({"shared": []})
        gr.HTML('<div id="app-header"><h1>RAG Knowledge Assistant</h1><p>Choose a provider, upload a document, and ask a question.</p></div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                provider = gr.Dropdown(
                    choices=[("NVIDIA model", "nvidia"), ("Pinecone Assistant", "pinecone")],
                    value="pinecone", label="Answer provider",
                )
                connection_status = gr.Markdown(status_markdown(default), elem_id="provider-status")
                refresh = gr.Button("Check connection / refresh library")
                top_k = gr.Slider(1, 20, value=5, step=1, label="NVIDIA retrieved chunks", visible=False, info="Controls NVIDIA retrieval. Assistant manages its own retrieval.")
                with gr.Group(visible=False) as nvidia_options:
                    thinking = gr.Checkbox(value=default["enable_thinking"], label="Enable NVIDIA thinking", info="Off for faster answers. On for additional reasoning.")
                gr.Markdown("### Document library")
                library_note = gr.Markdown("Files upload to Pinecone Assistant. Refresh to check when processing is complete.")
                upload = gr.File(label="PDF or TXT document", file_types=[".pdf", ".txt"], type="filepath")
                upload_button = gr.Button("Upload to Pinecone Assistant", variant="primary")
                upload_status = gr.Markdown("Files upload only to Pinecone Assistant. Wait until their status is Available before chatting.")
                library = gr.Dataframe(headers=["Document", "Status"], datatype=["str", "str"], value=[], interactive=False, label="Selected provider documents")
            with gr.Column(scale=2, min_width=440):
                chatbot = gr.Chatbot(label="Document answers", height=530, placeholder="Upload a document, then ask a question about it.")
                response_status = gr.Markdown("Pinecone Assistant - Ready" if default["configured"] else "Pinecone Assistant - Setup required", elem_id="response-status")
                chat_interface = gr.ChatInterface(
                    fn=rag_answer, chatbot=chatbot,
                    additional_inputs=[provider, top_k, thinking],
                    additional_outputs=[response_status],
                    submit_btn="Ask", stop_btn="Stop", analytics_enabled=False,
                )
                gr.Markdown("NVIDIA retrieves from the Pinecone index; Assistant retrieves from its own uploaded files. Each question is independent.")
                gr.HTML('<a href="/docs" target="_blank">API documentation</a> · <a href="/health" target="_blank">System health</a>')
        provider.change(
            provider_panel, inputs=[provider, receipts],
            outputs=[connection_status, nvidia_options, thinking, upload_button, library, library_note, upload_status, response_status, top_k],
        )
        chat_interface.textbox.stop(
            lambda: "Response stopped. Partial answer kept.", outputs=response_status, queue=False,
        )
        chatbot.clear(lambda: "Ready for a new question.", outputs=response_status, queue=False)
        # Serialize library updates while an upload or refresh is running.
        busy_controls = [provider, refresh, upload_button]
        def lock_controls():
            return [gr.update(interactive=False) for _ in busy_controls]
        def unlock_controls():
            return [gr.update(interactive=True) for _ in busy_controls]
        refresh.click(lock_controls, outputs=busy_controls, queue=False).then(
            refresh_provider, inputs=[provider, receipts], outputs=[connection_status, receipts, library],
        ).then(unlock_controls, outputs=busy_controls, queue=False)
        upload_button.click(lock_controls, outputs=busy_controls, queue=False).then(
            ingest_file, inputs=[upload, receipts, provider], outputs=[upload_status, receipts, library],
        ).then(unlock_controls, outputs=busy_controls, queue=False)
    return demo


def mount_demo(app, demo):
    return gr.mount_gradio_app(
        app, demo, path="/", css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
    )
