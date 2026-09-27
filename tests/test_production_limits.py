import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError
from app.schemas.chat import ChatRequest
from app.api.routes import ingest
from app.services import ingestion_service


@pytest.mark.parametrize("query", ["", "   ", "x" * 12001])
def test_invalid_questions_are_rejected(query):
    with pytest.raises(ValidationError):
        ChatRequest(query=query)


def test_api_reads_at_most_limit_plus_one_and_closes_file(monkeypatch):
    monkeypatch.setattr(ingest, "AppSettings", lambda: SimpleNamespace(max_upload_bytes=4))
    process = Mock()
    monkeypatch.setattr(ingest, "process_document", process)
    file = UploadFile(file=BytesIO(b"oversized"), filename="a.txt")
    with pytest.raises(HTTPException) as error:
        asyncio.run(ingest.upload_file(file, provider="pinecone"))
    assert error.value.status_code == 413
    assert file.file.closed
    process.assert_not_called()


def test_shared_service_rejects_large_files_before_provider_call(monkeypatch):
    monkeypatch.setattr(ingestion_service, "AppSettings", lambda: SimpleNamespace(max_upload_bytes=4))
    factory = Mock()
    monkeypatch.setattr(ingestion_service, "get_chat_service", factory)
    with pytest.raises(HTTPException) as error:
        ingestion_service.process_document("a.txt", b"oversized")
    assert error.value.status_code == 413
    factory.assert_not_called()


def test_theme_and_mount_apply_dark_mode_and_upload_limit(monkeypatch):
    from app.ui import frontend, styles
    assert styles.build_theme() is not None
    mount = Mock()
    monkeypatch.setattr(frontend.gr, "mount_gradio_app", mount)
    frontend.mount_demo(Mock(), Mock())
    options = mount.call_args.kwargs
    assert options["js"] == styles.INITIAL_THEME
    assert "classList.add('dark')" in options["head"]
    assert options["max_file_size"] == 20 * 1024 * 1024
    assert options["show_error"] is False


def test_file_loader_uses_same_parser_for_disk_and_bytes(tmp_path):
    from app.pipeline.loader import UniversalDocumentLoader
    path = tmp_path / "a.txt"
    path.write_text("Document text", encoding="utf-8")
    loader = UniversalDocumentLoader()
    assert loader.load_file(path)[0].page_content == loader.load_bytes("a.txt", b"Document text")[0].page_content
