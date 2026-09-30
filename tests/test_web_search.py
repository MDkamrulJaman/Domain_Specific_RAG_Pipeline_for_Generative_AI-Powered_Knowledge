"""Offline search tests: no credentials, servers, or provider requests needed."""
import json
from unittest.mock import Mock

import httpx
import pytest
from fastapi import HTTPException

from app.core.config import WebSearchSettings
from app.schemas.chat import ChatRequest
from app.services.chat_command import AnswerCommand
from app.services.web_search import TavilySearchAdapter
from app.services.answer_evidence import MISSING


def settings(**kwargs):
    return WebSearchSettings(_env_file=None, TAVILY_API_KEY="test-placeholder", **kwargs)


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


def test_assistant_receives_search_option():
    receiver = Mock()
    AnswerCommand.from_request(ChatRequest(query="q", web_search=True)).execute(receiver)
    receiver.stream.assert_called_once_with("q", web_search=True, skill="general")


def test_ui_passes_search_option(monkeypatch):
    from app.ui.handlers import rag_answer
    receiver = Mock()
    receiver.stream.return_value = iter(["answer"])
    monkeypatch.setattr("app.ui.handlers.get_chat_service", lambda provider: receiver)
    list(rag_answer("q", [], "pinecone", True))
    receiver.stream.assert_called_once_with("q", web_search=True, skill="general")


def test_assistant_uses_search_after_missing_document_evidence():
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
    receiver.stream.assert_called_once_with("q", web_search=True, skill="general")


def test_source_excerpts_escape_embedded_links():
    from app.services.web_search import source_excerpts
    result = source_excerpts([{"text": "![image](https://untrusted.example) <script>", "url": "https://example.org"}])
    assert "![image](" not in result
    assert "<script>" not in result
    assert "[example.org](https://example.org)" in result


def test_named_source_links_and_excerpt_headings():
    from app.services.web_search import source_links, source_excerpts
    sources = [{"title": "Publisher Documentation", "url": "https://example.org/docs", "text": "Evidence"}]
    link = "[W1] [Publisher Documentation](https://example.org/docs)"
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
