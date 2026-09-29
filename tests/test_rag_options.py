"""Generic NVIDIA budgets and request contracts, without task policies."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from app.services.llm_service import LLMService
from app.services.prompts import build_rag_prompt
from app.services.rag_service import RAGService
from app.schemas.chat import ChatRequest


@pytest.mark.parametrize('requested, expected', [(512, 512), (1024, 600), (None, 600)])
def test_output_budget_never_exceeds_configured_limit(requested, expected):
    settings = SimpleNamespace(
        MODEL_NAME='test', MODEL_MAX_TOKENS=600, MODEL_TEMPERATURE=0.2,
        MODEL_TOP_P=0.95, MODEL_ENABLE_THINKING=False,
    )
    client = Mock()
    stream = Mock()
    stream.__iter__ = Mock(return_value=iter([]))
    client.chat.completions.create.return_value = stream
    list(LLMService(settings=settings, client=client).stream('q', max_tokens=requested))
    assert client.chat.completions.create.call_args.kwargs['max_tokens'] == expected
    stream.close.assert_called_once()


def test_general_context_is_bounded_even_with_large_manual_depth():
    prompt = build_rag_prompt('q', [{'text': str(n) + 'x' * 3999} for n in range(10)])
    assert '0' + 'x' * 3999 in prompt
    assert '4' + 'x' * 3999 not in prompt


def test_skill_is_absent_from_public_request_schema():
    assert "skill" not in ChatRequest.model_json_schema()["properties"]


@pytest.mark.parametrize("top_k, expected", [(None, 5), (2, 2)])
def test_retrieval_default_and_override(top_k, expected):
    store, llm = Mock(), Mock()
    store.search.return_value = [{"text":"source"}]
    llm.stream.return_value = iter(["answer"])
    list(RAGService(store,llm).stream("q", top_k))
    store.search.assert_called_once_with("q", expected)
    llm.stream.assert_called_once()
    assert llm.stream.call_args.kwargs == {"enable_thinking": None}


def test_frontend_sends_plain_question_without_skill(monkeypatch):
    from app.ui.handlers import rag_answer
    service = Mock()
    service.stream.return_value = iter(["answer"])
    monkeypatch.setattr("app.ui.handlers.get_chat_service", lambda _: service)
    list(rag_answer("Summarize the document", [], "nvidia"))
    service.stream.assert_called_once_with("Summarize the document", None, enable_thinking=False)
