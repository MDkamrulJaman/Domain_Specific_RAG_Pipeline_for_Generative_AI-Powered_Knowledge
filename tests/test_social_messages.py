"""Short social messages avoid provider initialization, retrieval, and search."""
import pytest
from unittest.mock import Mock
from app.services.chat_command import AnswerCommand
from app.schemas.chat import ChatRequest


@pytest.mark.parametrize("provider", ["nvidia", "pinecone"])
@pytest.mark.parametrize("message", ["nice", "GOOD!", "wellcome", "Hello", "Thank you."])
def test_social_api_never_initializes_provider(client, monkeypatch, provider, message):
    factory = Mock(side_effect=AssertionError("must stay local"))
    monkeypatch.setattr("app.api.routes.chat.get_chat_service", factory)
    response = client.post("/chat/stream", json={"query":message,"provider":provider,"web_search":True})
    assert response.status_code == 200
    assert response.text == AnswerCommand.from_request(ChatRequest(query=message)).local_reply
    factory.assert_not_called()


def test_social_ui_never_initializes_provider(monkeypatch):
    from app.ui.handlers import rag_answer
    factory = Mock(side_effect=AssertionError("must stay local"))
    monkeypatch.setattr("app.ui.handlers.get_chat_service", factory)
    assert list(rag_answer("nice", [], "nvidia", web_search=True))[0][0] == "Glad to help!"
    factory.assert_not_called()


@pytest.mark.parametrize("query", ["Good, explain the requirements", "Hello, what is the voltage?", "What does NICE mean?"])
def test_real_questions_are_not_swallowed(query):
    assert AnswerCommand.from_request(ChatRequest(query=query)).local_reply is None
