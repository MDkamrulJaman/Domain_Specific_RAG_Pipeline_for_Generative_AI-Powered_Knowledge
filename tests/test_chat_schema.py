"""Chat request defaults and input validation."""

import pytest
from pydantic import ValidationError
from app.schemas.chat import ChatRequest


def test_defaults_and_bounds():
    req = ChatRequest(query="q")
    assert req.provider == "pinecone"
    with pytest.raises(ValidationError):
        ChatRequest(query="q", provider="unsupported")


@pytest.mark.parametrize("query", ["", "   ", "x" * 12001])
def test_invalid_questions_are_rejected(query):
    with pytest.raises(ValidationError):
        ChatRequest(query=query)
