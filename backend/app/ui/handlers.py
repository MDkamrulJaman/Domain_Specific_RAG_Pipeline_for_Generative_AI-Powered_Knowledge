import asyncio
import copy
import html
import logging
import time
from pathlib import Path

import gradio as gr
from fastapi import HTTPException

from app.services.ingestion_service import process_document
from app.core.config import AppSettings
from app.schemas.chat import ChatRequest
from app.services.chat_command import AnswerCommand
from pydantic import ValidationError
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


def refresh_provider(provider, receipts):
    config = inspect_provider(provider)
    # Copy-on-write session data prevents accidental mutation of the caller.
    # This is not an undo history (Memento) or a Prototype object factory.
    updated = copy.deepcopy(receipts or {})
    if provider == "pinecone" and config.get("connected"):
        entries = updated.setdefault("shared", [])
        for entry in entries:
            entry.get("results", {}).pop("pinecone", None)
        entries[:] = [entry for entry in entries if entry.get("results")]
        entries.extend({"name": file["name"], "results": {"pinecone": file}}
                       for file in config.get("files", []))
    return status_markdown(config), updated, library_rows(provider, updated)


def rag_answer(message, _history, provider="pinecone", web_search=False, skill="general"):
    label = PROVIDER_LABELS[provider]
    if not message or not message.strip():
        yield "Please enter a question about your documents.", f"{label} · No question submitted"
        return
    try:
        request = ChatRequest(query=message, provider=provider, web_search=web_search, skill=skill)
    except (ValidationError, ValueError):
        yield "Enter a question of up to 12,000 characters, and select a valid task.", f"{label} · Invalid request"
        return
    command = AnswerCommand.from_request(request)
    if command.local_reply is not None:
        yield command.local_reply, "Ready for your next question."
        return
    message = request.query
    answer = ""
    started = time.perf_counter()
    first_token = None
    yield answer, f"{label} · Retrieving context and waiting for the first token…"
    try:
        service = get_chat_service(provider)
        # Use the same Command as the API so provider options cannot drift.
        for fragment in command.execute(service):
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
        limit = AppSettings().max_upload_bytes
        if path.stat().st_size > limit:
            raise HTTPException(413, "Document exceeds the configured upload size limit.")
        def read_bounded():
            with path.open("rb") as handle:
                return handle.read(limit + 1)
        content = await asyncio.to_thread(read_bounded)
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
        return "Upload failed. Check the provider connection, then reload the page before retrying.", updated, library_rows(provider, updated)


