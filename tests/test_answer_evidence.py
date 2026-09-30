"""Document-first decisions and streaming marker boundaries without network calls."""
import pytest
from unittest.mock import Mock
from fastapi import HTTPException
from app.services.answer_evidence import MISSING, supported_stream, NO_DOCUMENT_ANSWER, NO_WEB_ANSWER
from app.services.assistant_service import AssistantService


@pytest.mark.parametrize("cut", range(len(MISSING)+1))
def test_marker_split_at_every_boundary(cut):
    stream = supported_stream(iter([MISSING[:cut], MISSING[cut:]]))
    with pytest.raises(StopIteration) as result:
        next(stream)
    assert result.value.value is False


def test_supported_answer_streams_immediately():
    def fragments():
        yield "Answer"
        raise RuntimeError("later")
    stream = supported_stream(fragments())
    assert next(stream) == "Answer"
    with pytest.raises(RuntimeError):
        next(stream)


@pytest.mark.parametrize("web", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_document_first_matrix(web, supported):
    search = Mock()
    search.search.return_value = [{"text":"evidence", "url":"https://example.org"}]
    answers = [iter(["Document answer" if supported else MISSING]), iter(["Web answer"])]
    service = AssistantService.__new__(AssistantService)
    service.web_search_tool = search
    service._chat = Mock(side_effect=answers)
    generation = service._chat
    result = "".join(service.stream("q", web_search=web))
    assert search.search.call_count == int(web and not supported)
    assert generation.call_count == (2 if web and not supported else 1)
    assert ("Document answer" if supported else "Web answer" if web else NO_DOCUMENT_ANSWER) in result
    assert MISSING not in result


def test_web_evidence_also_must_support_answer():
    service = AssistantService.__new__(AssistantService)
    service._chat = Mock(side_effect=[iter([MISSING]), iter([MISSING])])
    service.web_search_tool = Mock()
    service.web_search_tool.search.return_value = [{"text":"irrelevant", "url":"https://example.org"}]
    result = "".join(service.stream("q",web_search=True))
    assert NO_WEB_ANSWER in result
    assert "Web sources:" not in result


def test_document_timeout_never_triggers_web():
    service = AssistantService.__new__(AssistantService)
    service._chat = Mock(side_effect=HTTPException(504, "timeout"))
    service.web_search_tool = Mock()
    with pytest.raises(HTTPException):
        list(service.stream("q", web_search=True))
    service.web_search_tool.search.assert_not_called()


@pytest.mark.parametrize("web", [False, True])
def test_natural_refusal_drives_fallback(web):
    search = Mock()
    search.search.return_value = [{"text":"web evidence", "url":"https://example.org"}]
    refusal = "You did not provide enough information to answer this question."
    answers = [iter(refusal), iter(["Web answer"])]
    service = AssistantService.__new__(AssistantService)
    service.web_search_tool = search
    service._chat = Mock(side_effect=answers)
    answer = "".join(service.stream("q", web_search=web))
    assert ("Web answer" if web else NO_DOCUMENT_ANSWER) in answer
    assert refusal not in answer
    assert search.search.call_count == int(web)


def test_partial_refusal_phrase_is_not_a_false_negative():
    text = "I don't have enough information about its history, but the voltage is 12V."
    assert "".join(supported_stream(iter(text))) == text
