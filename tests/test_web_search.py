"""Offline search tests: no credentials, servers, or provider requests needed."""
import json
from unittest.mock import Mock

import httpx
import pytest
from fastapi import HTTPException

from app.core.config import WebSearchSettings
from app.schemas.chat import ChatRequest
from app.services.chat_command import AnswerCommand
from app.services.rag_service import RAGService
from app.services.web_search import TavilySearchAdapter
from app.services.answer_evidence import MISSING


def settings(**kwargs):
    return WebSearchSettings(_env_file=None, TAVILY_API_KEY="test-placeholder", **kwargs)


def pipeline(docs):
    store = Mock()
    store.search.return_value = docs
    llm = Mock()
    llm.stream.return_value = iter(["Answer [1]"])
    search = Mock()
    search.search.return_value = [{"text": "Public evidence", "url": "https://example.org/source"}]
    checker = Mock()
    checker.supports.return_value = False
    return RAGService(store, llm, search), search, checker


def test_opt_out_preserves_document_path():
    service, search, checker = pipeline([{"text": "Private evidence"}])
    assert service.ask("question", 5) == "Answer [1]"
    search.search.assert_not_called()
    checker.supports.assert_not_called()


@pytest.mark.parametrize("docs", [[], [{"text": "Unrelated document"}]])
def test_insufficient_documents_search_with_citations(docs):
    service, search, checker = pipeline(docs)
    service.llm.stream.side_effect = [iter([MISSING]), iter(["Answer"])] if docs else [iter(["Answer"])]
    answer = service.ask("question", 5, web_search=True)
    assert "Web search enabled" in answer
    assert "https://example.org/source" in answer
    search.search.assert_called_once_with("question")
    if not docs:
        checker.supports.assert_not_called()


def test_enabled_search_uses_one_generation_call():
    service, search, checker = pipeline([{"text": "Evidence"}])
    checker.supports.return_value = True
    assert "Answer [1]" in service.ask("question", 5, web_search=True)
    search.search.assert_not_called()
    service.llm.stream.assert_called_once()


def test_empty_search_returns_explicit_no_answer():
    service, search, _ = pipeline([])
    search.search.return_value = []
    assert "not find enough evidence" in service.ask("question", 5, web_search=True)
    service.llm.stream.assert_not_called()


@pytest.mark.parametrize("status", [401, 503])
def test_retrieval_failure_blocks_generation_with_parallel_search(status):
    service, search, _ = pipeline([])
    service.vectorstore.search.side_effect = HTTPException(status, "Failed")
    with pytest.raises(HTTPException):
        service.ask("q", 5, web_search=True)
    service.llm.stream.assert_not_called()


def test_missing_index_can_search():
    service, search, _ = pipeline([])
    service.vectorstore.search.side_effect = HTTPException(404, "Empty")
    assert "Web sources" in service.ask("q", 5, web_search=True)


def test_langchain_tool_request_and_source_filtering():
    def handle(request):
        payload = json.loads(request.content)
        assert payload["query"] == "public question"
        assert payload["max_results"] == 3
        assert payload["include_raw_content"] is False
        return httpx.Response(200, json={"results": [
            {"url": "javascript:alert(1)", "content": "bad"},
            {"url": "https://example.org/a", "content": "x" * 3000},
            {"url": "https://example.org/a", "content": "duplicate"},
        ]})
    adapter = TavilySearchAdapter(settings(), httpx.MockTransport(handle))
    docs = adapter.search("public question")
    assert len(docs) == 1
    assert len(docs[0]["text"]) == 2000


def test_search_error_is_sanitized():
    def handle(request):
        return httpx.Response(401, text="sensitive upstream payload")
    adapter = TavilySearchAdapter(settings(), httpx.MockTransport(handle))
    with pytest.raises(HTTPException) as caught:
        adapter.search("q")
    assert caught.value.status_code == 503
    assert "sensitive" not in caught.value.detail


def test_missing_key_is_actionable():
    adapter = TavilySearchAdapter(WebSearchSettings(_env_file=None, TAVILY_API_KEY=""))
    with pytest.raises(HTTPException, match="TAVILY_API_KEY"):
        adapter.search("q")


def test_api_routes_web_search_to_nvidia(client, monkeypatch):
    receiver = Mock()
    receiver.stream.return_value = iter(["answer"])
    monkeypatch.setattr("app.api.routes.chat.get_chat_service", lambda provider: receiver)
    response = client.post("/chat/stream", json={"query": "q", "provider": "nvidia", "web_search": True})
    assert response.status_code == 200
    receiver.stream.assert_called_once_with("q", 5, enable_thinking=None, web_search=True)


def test_assistant_receives_search_option():
    receiver = Mock()
    AnswerCommand.from_request(ChatRequest(query="q", web_search=True)).execute(receiver)
    receiver.stream.assert_called_once_with("q", 5, enable_thinking=None, web_search=True)


def test_ui_passes_search_option(monkeypatch):
    from app.ui.handlers import rag_answer
    receiver = Mock()
    receiver.stream.return_value = iter(["answer"])
    monkeypatch.setattr("app.ui.handlers.get_chat_service", lambda provider: receiver)
    list(rag_answer("q", [], "nvidia", 5, False, True))
    receiver.stream.assert_called_once_with("q", 5, enable_thinking=False, web_search=True)


def test_assistant_uses_search_without_nvidia():
    from app.core.config import AssistantSettings
    from app.services.assistant_service import AssistantService
    config = AssistantSettings(_env_file=None, PINECONE_ASSISTANT_API_KEY="test-key",
        PINECONE_ASSISTANT_NAME="test", PINECONE_ASSISTANT_MODEL="gpt-4o",
        PINECONE_ASSISTANT_TIMEOUT_SECONDS=30)
    sdk, search = Mock(), Mock()
    sdk.assistants.chat.return_value.text.side_effect = [iter([MISSING]), iter(["answer"])]
    search.search.return_value = [{"text": "public evidence", "url": "https://example.org"}]
    service = AssistantService(config, sdk, search)
    answer = "".join(service.stream("q", web_search=True))
    assert "https://example.org" in answer
    assert "public evidence" in sdk.assistants.chat.call_args.kwargs["messages"][0]["content"]
    search.search.assert_called_once_with("q")
    assert sdk.assistants.chat.call_count == 2


def test_nvidia_midstream_timeout_is_actionable():
    from app.services.llm_service import LLMService
    service = LLMService.__new__(LLMService)
    def stream(*args):
        yield "partial"
        raise httpx.ReadTimeout("upstream")
    service._stream = stream
    result = service.stream("q")
    assert next(result) == "partial"
    with pytest.raises(HTTPException) as caught:
        next(result)
    assert caught.value.status_code == 504
    assert "configured network timeout" in caught.value.detail


@pytest.mark.parametrize("configured_retries, expected", [(None, 0), (0, 0), (1, 1)])
def test_nvidia_retry_default_and_override(monkeypatch, configured_retries, expected):
    from app.core.config import NvidiaSettings
    from app.services import llm_service
    config = NvidiaSettings(_env_file=None, MODEL_BASE_URL="https://example.org",
        MODEL_NAME="test", MODEL_API_KEY="test-key", MODEL_TIMEOUT_SECONDS=30,
        MODEL_TEMPERATURE=0.1, MODEL_TOP_P=0.9, MODEL_MAX_TOKENS=512)
    if configured_retries is not None:
        config.MODEL_MAX_RETRIES = configured_retries
    client_factory = Mock()
    monkeypatch.setattr(llm_service, "OpenAI", client_factory)
    llm_service.LLMService(config)
    assert client_factory.call_args.kwargs["max_retries"] == expected


@pytest.mark.parametrize("status, message", [(400, "No files found"), (401, "Unauthorized")])
def test_assistant_no_files_excerpts_and_auth_errors(status, message):
    from app.services.assistant_service import AssistantService
    from pinecone.errors.exceptions import ApiError
    service = AssistantService.__new__(AssistantService)
    service.name, service.model, service.timeout = "test", "test", 30
    service.client = Mock()
    service.client.assistants.chat.side_effect = ApiError(message, status)
    service.web_search_tool = Mock()
    service.web_search_tool.search.return_value = [{"text": "public evidence", "url": "https://example.org"}]
    if status == 400:
        answer = "".join(service.stream("q", web_search=True))
        assert "not a generated answer" in answer
        assert "https://example.org" in answer
    else:
        with pytest.raises(ApiError):
            list(service.stream("q", web_search=True))


def test_assistant_search_api_option(client, monkeypatch):
    receiver = Mock()
    receiver.stream.return_value = iter(["answer"])
    monkeypatch.setattr("app.api.routes.chat.get_chat_service", lambda provider: receiver)
    response = client.post("/chat/stream", json={"query": "q", "web_search": True})
    assert response.status_code == 200
    receiver.stream.assert_called_once_with("q", 5, enable_thinking=None, web_search=True)


@pytest.mark.parametrize("partial", [False, True])
def test_nvidia_timeout_keeps_search_evidence(partial):
    service, search, _ = pipeline([])
    def timeout(*args, **kwargs):
        if partial:
            yield "partial answer"
        raise HTTPException(504, "NVIDIA timed out.")
    service.llm.stream.side_effect = timeout
    answer = service.ask("q", 5, web_search=True)
    assert "waiting for NVIDIA" in answer
    assert "answer generated by NVIDIA" not in answer
    assert "not a generated answer" in answer
    assert "https://example.org/source" in answer
    assert ("partial answer" in answer) is partial
    service.llm.stream.assert_called_once()


def test_nvidia_timeout_without_search_still_reports_error():
    service, _, _ = pipeline([{"text": "document"}])
    service.llm.stream.side_effect = HTTPException(504, "NVIDIA timed out.")
    with pytest.raises(HTTPException):
        service.ask("q", 5)


def test_source_excerpts_escape_embedded_links():
    from app.services.web_search import source_excerpts
    result = source_excerpts([{"text": "![image](https://untrusted.example) <script>", "url": "https://example.org"}])
    assert "![image](" not in result
    assert "<script>" not in result
    assert "[example.org](https://example.org)" in result


def test_nvidia_document_answer_does_not_search():
    service, search, _ = pipeline([{"text": "document"}])
    assert "Answer" in service.ask("q", 5, web_search=True)
    search.search.assert_not_called()
    service.llm.stream.assert_called_once()


def test_nvidia_web_budgets_do_not_mutate_sources():
    from app.core.config import NvidiaWebSettings
    service, search, _ = pipeline([{"text": "a" * 1100}, {"text": "b" * 1100}])
    service.web_settings = NvidiaWebSettings(_env_file=None, NVIDIA_WEB_DOCUMENT_CHARS=1200,
        NVIDIA_WEB_EXCERPT_CHARS=200, NVIDIA_WEB_MAX_TOKENS=256)
    original = {"text": "z" * 500, "url": "https://example.org"}
    search.search.return_value = [original]
    service.llm.stream.side_effect = [iter([MISSING]), iter(["answer"])]
    service.ask("q", 5, web_search=True)
    prompt = service.llm.stream.call_args_list[0].args[0]
    assert "a" * 1100 in prompt
    assert "b" * 1100 not in prompt
    assert "z" * 201 not in service.llm.stream.call_args.args[0]
    assert original["text"] == "z" * 500
    assert service.llm.stream.call_args.kwargs == {"enable_thinking": False, "max_tokens": 256}


def test_nvidia_web_honors_explicit_thinking():
    service, _, _ = pipeline([])
    list(service.stream("q", 5, enable_thinking=True, web_search=True))
    assert service.llm.stream.call_args.kwargs["enable_thinking"] is True


def test_named_source_links_and_excerpt_headings():
    from app.services.web_search import source_links, source_excerpts
    sources = [{"title": "NVIDIA Documentation", "url": "https://example.org/docs", "text": "Evidence"}]
    link = "[W1] [NVIDIA Documentation](https://example.org/docs)"
    assert link in source_links(sources)
    assert link in source_excerpts(sources)
    assert source_excerpts(sources).count("https://example.org/docs") == 1


def test_source_title_escapes_markdown_and_html():
    from app.services.web_search import source_links
    result = source_links([{"title": "[bad] <script>\nNew line", "url": "https://example.org"}])
    assert "<script>" not in result
    assert "\\[bad\\]" in result
    assert "&lt;script&gt; New line" in result


def test_search_retains_page_title():
    def handle(request):
        return httpx.Response(200, json={"results": [
            {"title": "  Publisher\nDocumentation  ", "url": "https://example.org", "content": "text"},
        ]})
    adapter = TavilySearchAdapter(settings(), httpx.MockTransport(handle))
    assert adapter.search("q")[0]["title"] == "Publisher Documentation"
