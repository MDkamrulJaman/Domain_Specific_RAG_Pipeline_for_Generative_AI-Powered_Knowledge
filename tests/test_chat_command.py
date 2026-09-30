"""Command behavior shared by HTTP and Gradio invokers."""
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest

from app.schemas.chat import ChatRequest
from app.services.chat_command import AnswerCommand


def test_command_snapshots_request_and_is_immutable():
    request = ChatRequest(query="original", provider="pinecone")
    command = AnswerCommand.from_request(request)
    request.query = "changed"
    assert command.query == "original"
    with pytest.raises(FrozenInstanceError):
        command.query = "changed"
    receiver = Mock()
    command.execute(receiver)
    receiver.stream.assert_called_once_with("original", web_search=False, skill="general")


def test_command_preserves_lazy_stream_and_failure():
    events = []
    class Receiver:
        def stream(self, query, web_search=False, skill="general"):
            events.append(query)
            yield "first"
            raise RuntimeError("upstream failed")
    command = AnswerCommand.from_request(ChatRequest(query="q"))
    stream = command.execute(Receiver())
    assert events == []
    assert next(stream) == "first"
    assert events == ["q"]
    with pytest.raises(RuntimeError, match="upstream failed"):
        next(stream)


def test_assistant_receiver_uses_basic_chat_contract():
    class Receiver:
        def stream(self, query, web_search=False, skill="general"):
            yield query
    command = AnswerCommand.from_request(ChatRequest(query="q"))
    assert list(command.execute(Receiver())) == ["q"]
