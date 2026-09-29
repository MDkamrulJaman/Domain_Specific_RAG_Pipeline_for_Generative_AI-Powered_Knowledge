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
    """Provide an in-process test client for isolated API endpoint tests.

    This fixture is used instead of starting a real server, making tests faster
    and deterministic. It also provides a suitable boundary for mocking
    external dependencies—such as LLM providers, vector stores, or file
    storage—so tests can verify application behavior without network calls,
    credentials, side effects, or reliance on external service availability.
    """
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(ingest_router)
    with TestClient(app) as test_client:
        yield test_client
