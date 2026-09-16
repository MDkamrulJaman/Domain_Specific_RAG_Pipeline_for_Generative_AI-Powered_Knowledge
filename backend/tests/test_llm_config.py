import os

from app.core.config import Settings


def test_settings_exposes_ollama_configuration(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "45")

    settings = Settings()

    assert settings.OLLAMA_BASE_URL == "http://localhost:11434/"
    assert settings.OLLAMA_MODEL == "llama3"
    assert settings.OLLAMA_TIMEOUT_SECONDS == 45.0
