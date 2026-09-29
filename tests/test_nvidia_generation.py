"""NVIDIA streaming, output limits, and grounded prompt construction."""

from types import SimpleNamespace
from unittest.mock import Mock
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


def test_nvidia_yields_first_token_without_waiting_for_rest():
    events = []
    def fragments():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="first"))])
        events.append("second")
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="second"))])
    upstream = Mock()
    upstream.__iter__ = Mock(return_value=fragments())
    service = LLMService.__new__(LLMService)
    service.settings = SimpleNamespace(MODEL_NAME="nvidia", MODEL_TEMPERATURE=0.2, MODEL_TOP_P=0.95, MODEL_MAX_TOKENS=1024, MODEL_ENABLE_THINKING=False)
    service.client = Mock()
    service.client.chat.completions.create.return_value = upstream
    stream = service.stream("q")
    assert next(stream) == "first"
    assert events == []
    assert list(stream) == ["second"]
    upstream.close.assert_called_once()
    assert service.client.chat.completions.create.call_args.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_rag_stream_uses_retrieved_context():
    service = RAGService.__new__(RAGService)
    service.vectorstore = Mock()
    service.vectorstore.search.return_value = [{"text": "source text"}]
    service.llm = Mock()
    service.llm.stream.return_value = iter(["one", "two"])
    assert list(service.stream("question", 5)) == ["one", "two"]
    assert "source text" in service.llm.stream.call_args.args[0]


def test_nvidia_reports_output_limit_and_closes_stream():
    service=LLMService.__new__(LLMService)
    service.settings=SimpleNamespace(MODEL_NAME="nvidia",MODEL_TEMPERATURE=0.2,MODEL_TOP_P=0.95,MODEL_MAX_TOKENS=1024,MODEL_ENABLE_THINKING=False)
    service.client=Mock()
    upstream=Mock()
    upstream.__iter__=Mock(return_value=iter([
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Partial"),finish_reason=None)]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None),finish_reason="length")]),
    ]))
    service.client.chat.completions.create.return_value=upstream
    answer="".join(service.stream("q"))
    assert answer.startswith("Partial") and "length limit" in answer
    options=service.client.chat.completions.create.call_args.kwargs
    assert options["max_tokens"] == 1024
    assert options["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    upstream.close.assert_called_once()


def test_nvidia_prompt_deduplicates_context_and_requests_concise_answer():
    service=RAGService.__new__(RAGService)
    service.vectorstore=Mock()
    service.vectorstore.search.return_value=[{"text":"unique source"},{"text":" unique source "},{"text":"second source"}]
    service.llm=Mock()
    service.llm.stream.return_value=iter(["answer"])
    assert list(service.stream("question",5)) == ["answer"]
    prompt=service.llm.stream.call_args.args[0]
    assert prompt.count("unique source") == 1
    assert "second source" in prompt and "150 words" in prompt
    assert "question explicitly requests" in prompt


def test_document_request_preserves_pre_search_generation_parameters():
    """Guard the request shape used before web search; no provider call is made."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    from app.services.llm_service import LLMService
    settings = SimpleNamespace(MODEL_NAME="test-model", MODEL_TEMPERATURE=1.0,
        MODEL_TOP_P=0.95, MODEL_MAX_TOKENS=1024, MODEL_ENABLE_THINKING=False)
    response = Mock()
    response.__iter__ = Mock(return_value=iter([]))
    client = Mock()
    client.chat.completions.create.return_value = response
    list(LLMService(settings, client).stream("document question"))
    client.chat.completions.create.assert_called_once_with(
        model="test-model", messages=[{"role": "user", "content": "document question"}],
        temperature=1.0, top_p=0.95, max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}, stream=True)
    response.close.assert_called_once()


def test_timeout_sends_only_one_http_request_by_default():
    """Exercise the real SDK retry policy with a local fake transport, not NVIDIA."""
    import httpx
    import pytest
    from openai import OpenAI
    from fastapi import HTTPException
    from app.core.config import NvidiaSettings
    settings = NvidiaSettings(_env_file=None, MODEL_BASE_URL="https://example.invalid/v1",
        MODEL_API_KEY="offline-placeholder", MODEL_NAME="test", MODEL_TIMEOUT_SECONDS=30,
        MODEL_TEMPERATURE=1, MODEL_TOP_P=0.95, MODEL_MAX_TOKENS=1024)
    requests = []
    def timeout(request):
        requests.append(request)
        raise httpx.ReadTimeout("simulated timeout", request=request)
    with httpx.Client(transport=httpx.MockTransport(timeout)) as http:
        with OpenAI(api_key="offline-placeholder", base_url=settings.MODEL_BASE_URL,
                    http_client=http, max_retries=settings.MODEL_MAX_RETRIES) as client:
            with pytest.raises(HTTPException) as error:
                list(LLMService(settings, client).stream("question"))
    assert error.value.status_code == 504
    assert len(requests) == 1
