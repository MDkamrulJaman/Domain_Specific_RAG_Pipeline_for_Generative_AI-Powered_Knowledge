"""Skill routing, bounded generation, grounding, and frontend controls."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.api.routes import chat
from app.services.llm_service import LLMService
from app.services.prompts import build_rag_prompt
from app.services.rag_service import RAGService
from app.services.skills import SKILLS


@pytest.mark.parametrize("skill", ["summarize", "explain", "requirements"])
def test_api_forwards_nvidia_skill(client, monkeypatch, skill):
    service = Mock()
    service.stream.return_value = iter(["answer"])
    monkeypatch.setattr(chat, "get_chat_service", lambda _: service)
    response = client.post('/chat/stream', json={
        'query': 'question', 'provider': 'nvidia', 'skill': skill,
    })
    assert response.status_code == 200
    assert response.text == 'answer'
    service.stream.assert_called_once_with('question', 5, enable_thinking=None, skill=skill)


def test_pinecone_does_not_receive_nvidia_skill(client, monkeypatch):
    service = Mock()
    service.stream.return_value = iter(['answer'])
    monkeypatch.setattr(chat, 'get_chat_service', lambda _: service)
    response = client.post('/chat/stream', json={'query': 'q', 'skill': 'summarize'})
    assert response.status_code == 200
    service.stream.assert_called_once_with('q', 5, enable_thinking=None)


def test_unknown_skill_rejected_before_provider_initialization(client, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(chat, 'get_chat_service', factory)
    assert client.post('/chat/stream', json={
        'query': 'q', 'provider': 'nvidia', 'skill': 'unknown',
    }).status_code == 422
    factory.assert_not_called()


@pytest.mark.parametrize('skill', ['summarize', 'explain', 'requirements'])
def test_skill_policy_and_limits_are_request_local(skill):
    retriever, generator = Mock(), Mock()
    retriever.search.return_value = [{'text': 'The controller shall start within 10 ms.'}]
    generator.stream.side_effect = lambda *a, **kw: iter(['answer'])
    service = RAGService(retriever, generator)
    assert ''.join(service.stream('startup', 3, skill=skill)) == 'answer'
    retriever.search.assert_called_once_with('startup', 3)
    prompt = generator.stream.call_args.args[0]
    assert SKILLS[skill].instruction in prompt
    assert '[1] The controller shall start within 10 ms.' in prompt
    assert generator.stream.call_args.kwargs['max_tokens'] == SKILLS[skill].max_tokens
    list(service.stream('another question'))
    assert 'max_tokens' not in generator.stream.call_args.kwargs
    assert SKILLS[skill].instruction not in generator.stream.call_args.args[0]


def test_context_budget_keeps_whole_passages_and_deduplicates():
    prompt = build_rag_prompt('q', [
        {'text': 'X' * 20001}, {'text': 'A shall be enabled.'},
        {'text': ' A shall be enabled. '},
    ], 'requirements')
    assert 'XXXXX' not in prompt
    assert prompt.count('A shall be enabled.') == 1
    assert 'exact wording' in prompt
    summary = build_rag_prompt('q', [{'text': 'source'}], 'summarize')
    assert 'not the entire document' in summary


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


def test_frontend_wires_skill_into_chat_and_keeps_pinecone_default():
    from app.ui.frontend import create_demo
    demo = create_demo()
    skill = next(b for b in demo.blocks.values() if getattr(b, 'label', None) == 'NVIDIA skill')
    assert skill.value == 'general'
    depth = next(b for b in demo.blocks.values() if getattr(b, 'label', None) == 'NVIDIA retrieval depth')
    assert depth.value == 'auto'
    parent = skill.parent
    ancestors = []
    while parent is not None:
        ancestors.append(parent)
        parent = getattr(parent, 'parent', None)
    assert any(getattr(block, 'visible', True) is False for block in ancestors)
    chatbot = next(b for b in demo.blocks.values() if getattr(b, 'elem_id', None) == 'knowledge-chat')
    selection = next(fn for fn in demo.fns.values() if (chatbot._id, 'example_select') in fn.targets)
    interface = selection.fn.__self__
    assert interface.additional_inputs[-1] is skill
    assert interface.additional_inputs[0].value == 'pinecone'


@pytest.mark.parametrize('provider', ['nvidia', 'pinecone'])
def test_frontend_forwards_skill_only_to_nvidia(monkeypatch, provider):
    from app.ui import handlers
    service = Mock()
    service.stream.return_value = iter(['answer'])
    monkeypatch.setattr(handlers, 'get_chat_service', lambda _: service)
    updates = list(handlers.rag_answer('q', [], provider, 5, False, 'explain'))
    assert updates[-1][0] == 'answer'
    options = service.stream.call_args.kwargs
    assert options.get('skill') == ('explain' if provider == 'nvidia' else None)


@pytest.mark.parametrize('skill, expected', [('general', 5), ('explain', 4), ('summarize', 8), ('requirements', 8)])
def test_automatic_retrieval_uses_skill_and_honors_manual_override(skill, expected):
    retriever, generator = Mock(), Mock()
    retriever.search.return_value = [{'text': 'source'}]
    generator.stream.side_effect = lambda *a, **kw: iter(['answer'])
    service = RAGService(retriever, generator)
    list(service.stream('q', None, skill=skill))
    retriever.search.assert_called_with('q', expected)
    list(service.stream('q', 2, skill=skill))
    retriever.search.assert_called_with('q', 2)


def test_api_accepts_automatic_retrieval(client, monkeypatch):
    service = Mock()
    service.stream.return_value = iter(['answer'])
    monkeypatch.setattr(chat, 'get_chat_service', lambda _: service)
    response = client.post('/chat/stream', json={
        'query': 'q', 'provider': 'nvidia', 'top_k': None, 'skill': 'requirements',
    })
    assert response.status_code == 200
    service.stream.assert_called_once_with('q', None, enable_thinking=None, skill='requirements')


def test_frontend_automatic_retrieval_reaches_service(monkeypatch):
    from app.ui import handlers
    service = Mock()
    service.stream.return_value = iter(['answer'])
    monkeypatch.setattr(handlers, 'get_chat_service', lambda _: service)
    assert list(handlers.rag_answer('q', [], 'nvidia', 'auto', False, 'summarize'))[-1][0] == 'answer'
    service.stream.assert_called_once_with('q', None, enable_thinking=False, skill='summarize')


def test_general_context_is_bounded_even_with_large_manual_depth():
    prompt = build_rag_prompt('q', [{'text': str(n) + 'x' * 3999} for n in range(10)])
    assert '0' + 'x' * 3999 in prompt
    assert '4' + 'x' * 3999 not in prompt
