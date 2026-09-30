"""Pinecone request policies: provider isolation, fallback, and API/UI validation."""
from unittest.mock import Mock
import pytest
from app.schemas.chat import ChatRequest
from app.services.chat_command import AnswerCommand
from app.services.assistant_service import AssistantService
from app.services.skills import SKILLS, assistant_policy
from app.services.answer_evidence import MISSING


@pytest.mark.parametrize("skill", list(SKILLS))
def test_assistant_policy_applies_to_both_answer_attempts(skill):
    service = AssistantService.__new__(AssistantService)
    service._chat = Mock(side_effect=[iter([MISSING]), iter(["Answer"])])
    service.web_search_tool = Mock()
    service.web_search_tool.search.return_value = [{"text":"source","url":"https://example.org"}]
    result = "".join(service.stream("original question", web_search=True, skill=skill))
    assert "Answer" in result
    assert service._chat.call_count == 2
    for call in service._chat.call_args_list:
        assert assistant_policy(skill) in call.args[0]
        assert MISSING in call.args[0]
    service.web_search_tool.search.assert_called_once_with("original question")


def test_api_forwards_pinecone_policy(client, monkeypatch):
    receiver = Mock()
    receiver.stream.return_value = iter(["answer"])
    monkeypatch.setattr("app.api.routes.chat.get_chat_service",lambda _:receiver)
    result = client.post("/chat/stream",json={"query":"q","skill":"requirements","web_search":True})
    assert result.status_code == 200
    receiver.stream.assert_called_once_with("q",web_search=True,skill="requirements")


def test_unknown_policy_rejected_before_sdk_initialization(client, monkeypatch):
    factory=Mock()
    monkeypatch.setattr("app.api.routes.chat.get_chat_service",factory)
    assert client.post("/chat/stream",json={"query":"q","skill":"invalid"}).status_code == 422
    factory.assert_not_called()


def test_ui_policy_visible_only_for_pinecone(monkeypatch):
    from app.ui import handlers
    receiver=Mock()
    receiver.stream.return_value=iter(["answer"])
    monkeypatch.setattr(handlers,"get_chat_service",lambda _:receiver)
    list(handlers.rag_answer("q",[],"pinecone",skill="summarize"))
    receiver.stream.assert_called_once_with("q",web_search=False,skill="summarize")


def test_policy_does_not_leak_to_later_request():
    service=AssistantService.__new__(AssistantService)
    service._chat=Mock(side_effect=[iter(["answer"]),iter(["answer"])])
    list(service.stream("q",skill="requirements"))
    list(service.stream("q"))
    assert SKILLS["requirements"].instruction not in service._chat.call_args.args[0]
    assert SKILLS["general"].instruction in service._chat.call_args.args[0]
