"""Shared test fixtures. TestClient calls run in process; no server is started."""
import os

# Keep UI construction offline and disable Gradio analytics in every test run.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.routes.chat import chat_router
from app.api.routes.ingest import ingest_router


@pytest.fixture
def client():
    """A lightweight API client for chat and upload endpoint tests."""
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(ingest_router)
    with TestClient(app) as test_client:
        yield test_client
