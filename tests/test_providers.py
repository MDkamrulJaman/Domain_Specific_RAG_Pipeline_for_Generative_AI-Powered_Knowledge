from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes import chat, ingest as ingest_routes
from app.services import ingestion_service as ingest
from app.core.config import AssistantSettings
from app.schemas.chat import ChatRequest
from app.services import assistant_service, provider_service
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(chat.chat_router)
    app.include_router(ingest_routes.ingest_router)
    return TestClient(app)


@pytest.mark.parametrize("provider", ["nvidia", "pinecone"])
def test_chat_routes_selected_provider(client, monkeypatch, provider):
    service = Mock()
    service.stream.return_value = iter(["Hello", " world"])
    factory = Mock(return_value=service)
    monkeypatch.setattr(chat, "get_chat_service", factory)
    response = client.post("/chat/stream", json={"query": "question", "provider": provider})
    assert response.status_code == 200
    assert response.text == "Hello world"
    factory.assert_called_once_with(provider)
    service.stream.assert_called_once_with("question", 5, enable_thinking=None)


def test_invalid_provider_rejected(client):
    assert client.post("/chat/stream", json={"query": "q", "provider": "unknown"}).status_code == 422
    assert client.post("/ingest/upload", data={"provider": "unknown"}, files={"file": ("a.txt", b"abc")}).status_code == 422


def test_defaults_and_bounds():
    req = ChatRequest(query="q")
    assert (req.provider, req.top_k) == ("pinecone", 5)
    with pytest.raises(ValidationError):
        ChatRequest(query="q", top_k=0)


def test_missing_assistant_configuration(client, monkeypatch):
    def missing(_):
        raise HTTPException(503, "Set PINECONE_ASSISTANT_NAME")
    monkeypatch.setattr(chat, "get_chat_service", missing)
    response = client.post("/chat/stream", json={"query": "q", "provider": "pinecone"})
    assert response.status_code == 503
    assert "PINECONE_ASSISTANT_NAME" in response.text


def test_midstream_failure_preserves_partial_answer(client, monkeypatch):
    def broken(*args, **kwargs):
        yield "Partial answer"
        raise RuntimeError("private provider details")
    monkeypatch.setattr(chat, "get_chat_service", lambda _: SimpleNamespace(stream=broken))
    response = client.post("/chat/stream", json={"query": "q"})
    assert response.text.startswith("Partial answer")
    assert "interrupted" in response.text
    assert "private provider details" not in response.text


def test_assistant_upload_does_not_use_vector_index(client, monkeypatch):
    service = Mock()
    service.upload.return_value = {"file_id": "f", "name": "manual.txt", "status": "Processing"}
    monkeypatch.setattr(ingest, "get_chat_service", lambda _: service)
    vector = Mock(side_effect=AssertionError("must not use vector index"))
    monkeypatch.setattr(ingest, "get_vectorstore", vector)
    response = client.post("/ingest/upload", data={"provider":"pinecone"}, files={"file": ("manual.txt", b"hello")})
    assert response.status_code == 201
    assert response.json()["results"]["pinecone"]["status"] == "Processing"
    service.upload.assert_called_once_with("manual.txt", b"hello")
    vector.assert_not_called()



def test_nvidia_upload_still_indexes(client, monkeypatch):
    import numpy as np
    embeddings = Mock()
    embeddings.embed_chunks.return_value = np.ones((1, 3))
    store = Mock()
    monkeypatch.setattr(ingest, "EmbeddingService", lambda: embeddings)
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: store)
    response = client.post("/ingest/upload", data={"provider":"nvidia"}, files={"file": ("manual.txt", b"hello")})
    assert response.status_code == 201
    assert response.json()["results"]["nvidia"]["status"] == "Indexed"
    store.add.assert_called_once()


def test_assistant_chat_has_no_exclusion_filter(monkeypatch):
    settings = AssistantSettings(_env_file=None, PINECONE_ASSISTANT_API_KEY="assistant-key", PINECONE_ASSISTANT_NAME="manuals", PINECONE_ASSISTANT_MODEL="gpt-4o", PINECONE_ASSISTANT_TIMEOUT_SECONDS=60)
    monkeypatch.setattr(assistant_service, "AssistantSettings", lambda: settings)
    sdk = Mock()
    sdk.assistants.chat.return_value.text.return_value = iter(["answer"])
    monkeypatch.setattr(assistant_service, "Pinecone", lambda **kw: sdk)
    service = assistant_service.AssistantService()
    assert list(service.stream("question")) == ["answer"]
    options = sdk.assistants.chat.call_args.kwargs
    assert options["messages"] == [{"role":"user", "content":"question"}]
    assert "filter" not in options



def test_assistant_factory_never_initializes_vector_store(monkeypatch):
    provider_service.get_chat_service.cache_clear()
    vector = Mock(side_effect=AssertionError("Assistant must not retrieve from rag"))
    monkeypatch.setattr(provider_service, "get_vectorstore", vector)
    service = Mock()
    monkeypatch.setattr(assistant_service, "AssistantService", lambda: service)
    assert provider_service.get_chat_service("pinecone") is service
    vector.assert_not_called()
    provider_service.get_chat_service.cache_clear()



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


def test_gradio_selector_and_streaming_callback(monkeypatch):
    from app.ui import handlers as frontend
    from app.ui.frontend import create_demo
    demo = create_demo()
    dropdowns = [block for block in demo.blocks.values() if getattr(block, "label", None) == "Answer provider"]
    assert len(dropdowns) == 1
    assert dropdowns[0].value == "pinecone"
    service = Mock()
    service.stream.return_value = iter(["first", " second"])
    factory = Mock(return_value=service)
    monkeypatch.setattr(frontend, "get_chat_service", factory)
    updates = list(frontend.rag_answer("q", [], "pinecone"))
    assert [item[0] for item in updates] == ["", "first", "first second", "first second"]
    assert "Response complete" in updates[-1][1]
    factory.assert_called_once_with("pinecone")


def test_per_request_thinking_is_forwarded(client, monkeypatch):
    service = Mock()
    service.stream.return_value = iter(["answer"])
    monkeypatch.setattr(chat, "get_chat_service", lambda _: service)
    response = client.post("/chat/stream", json={"query": "q", "top_k": 9, "enable_thinking": True})
    assert response.status_code == 200
    service.stream.assert_called_once_with("q", 9, enable_thinking=True)


def test_provider_configuration_never_exposes_keys(monkeypatch):
    from app.core import config
    settings = SimpleNamespace(PINECONE_ASSISTANT_API_KEY="private-key", PINECONE_ASSISTANT_NAME="", PINECONE_ASSISTANT_MODEL="model")
    monkeypatch.setattr(config, "AssistantSettings", lambda: settings)
    monkeypatch.setattr(config, "RetrievalSettings", lambda: SimpleNamespace(PINECONE_API_KEY="vector-secret", PINECONE_INDEX_NAME="index", PINECONE_MODEL="rerank", HF_TOKEN="hf-secret", EMBEDDING_MODEL="embeddings"))
    result = provider_service.provider_configuration("pinecone")
    assert not result["configured"]
    assert "PINECONE_ASSISTANT_NAME" in result["message"]
    assert "private-key" not in str(result)


def test_provider_status_api(monkeypatch):
    from app.api.routes import providers
    app = FastAPI()
    app.include_router(providers.providers_router)
    monkeypatch.setattr(providers, "provider_configuration", lambda p: {"provider": p, "configured": False})
    check = Mock(return_value={"provider": "pinecone", "connected": True})
    monkeypatch.setattr(providers, "inspect_provider", check)
    with TestClient(app) as client:
        assert client.get("/providers/pinecone").json()["configured"] is False
        check.assert_not_called()
        assert client.get("/providers/pinecone?check_connection=true").json()["connected"] is True
        assert client.get("/providers/other").status_code == 422


def test_assistant_refresh_checks_own_files(monkeypatch):
    monkeypatch.setattr(provider_service, "provider_configuration", lambda p: {"configured":True,"provider":p})
    service = Mock()
    service.inspect_status.return_value = "Ready"
    service.list_files.return_value = [{"file_id":"f","name":"a.txt","status":"Processing"}]
    monkeypatch.setattr(provider_service, "get_chat_service", lambda _: service)
    result = provider_service.inspect_provider("pinecone")
    assert result["connected"]
    assert result["files"][0]["status"] == "Processing"
    assert "0/1" in result["message"]
    service.vectorstore.index.describe_index_stats.assert_not_called()



def test_frontend_switch_selects_only_provider_library(monkeypatch):
    from app.ui import handlers as frontend
    monkeypatch.setattr(frontend,"provider_configuration",lambda p:{"provider":p,"label":p,"configured":True,"model":"m","message":"Configured","enable_thinking":False})
    receipts={"shared":[{"name":"n.txt","results":{"nvidia":{"status":"Indexed"}}},{"name":"p.txt","results":{"pinecone":{"status":"Processing"}}}]}
    p=frontend.provider_panel("pinecone",receipts)
    n=frontend.provider_panel("nvidia",receipts)
    assert p[4] == [["p.txt","Processing"]]
    assert n[4] == [["n.txt","Indexed"]]
    assert not p[1]["visible"] and n[1]["visible"]
    assert p[3]["value"] == "Upload to Pinecone Assistant"
    assert n[3]["value"] == "Upload to NVIDIA model"



def test_frontend_upload_tracks_shared_library(tmp_path, monkeypatch):
    import asyncio
    from app.ui import handlers as frontend
    path = tmp_path / "manual.txt"
    path.write_text("sample content")
    calls = []
    def process(name, content, target, *, progress):
        calls.append((name, content))
        progress(0.5, "Processing")
        return {"status": "success", "message": "Processing", "results": {target: {"status":"Processing"}}}
    monkeypatch.setattr(frontend, "process_document", process)
    original = {"shared": []}
    message, receipts, rows = asyncio.run(frontend.ingest_file(str(path), original, progress=lambda *a, **kw: None))
    assert calls == [("manual.txt", b"sample content")]
    assert "Upload results" in message
    assert original == {"shared": []}
    assert len(receipts["shared"]) == 1
    assert frontend.library_rows("nvidia", receipts) == []
    assert frontend.library_rows("pinecone", receipts) == [["manual.txt","Processing"]]


def test_upload_validation_and_cleanup(monkeypatch):
    import asyncio
    from io import BytesIO
    from fastapi import UploadFile
    file = UploadFile(file=BytesIO(b"hello"), filename="unsafe.exe")
    with pytest.raises(HTTPException) as error:
        asyncio.run(ingest_routes.upload_file(file, provider="nvidia"))
    assert error.value.status_code == 400
    assert file.file.closed


def test_nvidia_ingestion_reports_stages(monkeypatch):
    import numpy as np
    embedding = Mock()
    embedding.embed_chunks.return_value = np.ones((1, 3))
    monkeypatch.setattr(ingest, "EmbeddingService", lambda: embedding)
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: Mock())
    stages = []
    result = ingest.process_document("manual.txt", b"hello", "nvidia", lambda fraction, text: stages.append((fraction, text)))
    assert result["chunks_created"] == 1
    assert [stage[0] for stage in stages] == [0.15, 0.3, 0.5, 0.8]


def test_shared_settings_do_not_require_generation_credentials():
    from app.core.config import RetrievalSettings
    settings = RetrievalSettings(_env_file=None, EMBEDDING_MODEL="embedding", HF_TOKEN="hf", PINECONE_API_KEY="vector", PINECONE_INDEX_NAME="index", PINECONE_DIMENSION=3, PINECONE_NAMESPACE="shared", PINECONE_MODEL="rerank")
    assert settings.PINECONE_API_KEY == "vector"
    assert "MODEL_API_KEY" not in RetrievalSettings.model_fields
    assert "PINECONE_ASSISTANT_API_KEY" not in RetrievalSettings.model_fields


def test_shared_index_receives_huggingface_vectors(monkeypatch):
    import numpy as np
    from app.pipeline import retrieval_service
    settings = SimpleNamespace(PINECONE_API_KEY="vector-key", PINECONE_NAMESPACE="shared", PINECONE_MODEL="rerank", PINECONE_INDEX_NAME="index", PINECONE_DIMENSION=3)
    sdk = Mock()
    sdk.describe_index.return_value = SimpleNamespace(schema=None, dimension=3)
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(retrieval_service, "RetrievalSettings", lambda: settings)
    monkeypatch.setattr(retrieval_service, "Pinecone", factory)
    store = retrieval_service.VectorStore()
    store.add(np.array([[1, 2, 3]]), [SimpleNamespace(page_content="chunk", metadata={"source": "a.txt"})])
    factory.assert_called_once_with(api_key="vector-key")
    assert sdk.Index.return_value.upsert.call_args.kwargs["vectors"][0]["values"] == [1, 2, 3]
    assert sdk.Index.return_value.upsert.call_args.kwargs["namespace"] == "shared"
    sdk.Index.return_value.upsert_records.assert_not_called()


def test_shared_search_embeds_query_with_huggingface(monkeypatch):
    import numpy as np
    from app.pipeline import retrieval_service
    store = retrieval_service.VectorStore.__new__(retrieval_service.VectorStore)
    store.namespace = "shared"
    store.rerank_model = "rerank"
    store.index = Mock()
    store.index.query.return_value = SimpleNamespace(matches=[SimpleNamespace(metadata={"text": "source"})])
    store.client = Mock()
    store.client.inference.rerank.return_value = SimpleNamespace(data=[SimpleNamespace(document={"text": "source"}, score=0.9)])
    embeddings = Mock()
    embeddings.embed_query.return_value = np.array([[1, 2, 3]])
    monkeypatch.setattr(retrieval_service, "EmbeddingService", lambda: embeddings)
    assert store.search("question", 4) == [{"text": "source", "score": None}]
    embeddings.embed_query.assert_called_once_with("question")
    store.index.query.assert_called_once_with(vector=[1, 2, 3], top_k=8, include_metadata=True, namespace="shared")
    store.index.search.assert_not_called()


def test_integrated_rag_uses_text_records_and_text_search(monkeypatch):
    from app.pipeline import retrieval_service as rs
    settings = SimpleNamespace(PINECONE_API_KEY="key", PINECONE_NAMESPACE="ns", PINECONE_MODEL="rerank", PINECONE_INDEX_NAME="rag", PINECONE_DIMENSION=1024)
    sdk = Mock()
    sdk.describe_index.return_value = {"schema":{"fields":{"text":{"type":"semantic_text"}}}}
    monkeypatch.setattr(rs, "RetrievalSettings", lambda: settings)
    monkeypatch.setattr(rs, "Pinecone", lambda **kw: sdk)
    embedding = Mock(side_effect=AssertionError("HF should not be used"))
    monkeypatch.setattr(rs, "EmbeddingService", embedding)
    store = rs.VectorStore()
    store.add(None, [SimpleNamespace(page_content="source", metadata={})])
    sdk.Index.return_value.upsert_records.assert_called_once()
    sdk.Index.return_value.upsert.assert_not_called()
    sdk.Index.return_value.search.return_value = {"result":{"hits":[{"fields":{"text":"source"}}]}}
    sdk.inference.rerank.return_value = SimpleNamespace(data=[SimpleNamespace(document={"text":"source"},score=1)])
    assert store.search("q") == [{"text":"source","score":None}]
    sdk.Index.return_value.query.assert_not_called()
    embedding.assert_not_called()


def test_default_upload_goes_only_to_assistant(monkeypatch):
    assistant=Mock()
    assistant.upload.return_value={"file_id":"f","name":"a.txt","status":"Processing"}
    monkeypatch.setattr(ingest,"get_chat_service",lambda _:assistant)
    index=Mock(side_effect=AssertionError("must not index"))
    monkeypatch.setattr(ingest,"index_document",index)
    result=ingest.process_document("a.txt",b"text")
    assert set(result["results"]) == {"pinecone"}
    index.assert_not_called()



def test_nvidia_failure_does_not_fall_back_to_assistant(monkeypatch):
    def fail(*a):
        raise HTTPException(503,"Index unavailable")
    monkeypatch.setattr(ingest,"index_document",fail)
    assistant=Mock(side_effect=AssertionError("must not upload"))
    monkeypatch.setattr(ingest,"get_chat_service",assistant)
    with pytest.raises(HTTPException):
        ingest.process_document("a.txt",b"text","nvidia")
    assistant.assert_not_called()



def test_integrated_upload_skips_huggingface(monkeypatch):
    store = Mock()
    store.integrated_embedding = True
    monkeypatch.setattr(ingest,"get_vectorstore",lambda:store)
    embedding = Mock(side_effect=AssertionError("HF should not be used"))
    monkeypatch.setattr(ingest,"EmbeddingService",embedding)
    result = ingest.process_document("a.txt",b"text","nvidia")
    assert result["results"]["nvidia"]["status"] == "Indexed"
    assert store.add.call_args.args[0] is None
    embedding.assert_not_called()


def test_assistant_upload_returns_processing_without_waiting():
    service = assistant_service.AssistantService.__new__(assistant_service.AssistantService)
    service.name = "assistant"
    service.client = Mock()
    def upload(**kw):
        assert kw["file_stream"].read() == b"content"
        assert kw["file_name"] == "a.txt"
        assert kw["timeout"] == -1
        return SimpleNamespace(id="f",name="a.txt",status="Processing")
    service.client.assistants.upload_file.side_effect = upload
    assert service.upload("a.txt",b"content")["status"] == "Processing"


def test_assistant_settings_do_not_load_vector_settings(monkeypatch):
    from app.core import config
    monkeypatch.setattr(config,"RetrievalSettings",Mock(side_effect=AssertionError("No index required")))
    monkeypatch.setattr(config,"AssistantSettings",lambda:SimpleNamespace(PINECONE_ASSISTANT_API_KEY="key",PINECONE_ASSISTANT_NAME="name",PINECONE_ASSISTANT_MODEL="model"))
    assert provider_service.provider_configuration("pinecone")["configured"]


def test_api_upload_default_is_pinecone(client,monkeypatch):
    service=Mock()
    service.upload.return_value={"file_id":"f","name":"a.txt","status":"Processing"}
    monkeypatch.setattr(ingest,"get_chat_service",lambda _:service)
    index=Mock(side_effect=AssertionError("index touched"))
    monkeypatch.setattr(ingest,"get_vectorstore",index)
    response=client.post("/ingest/upload",files={"file":("a.txt",b"content")})
    assert response.status_code == 201
    assert response.json()["provider"] == "pinecone"
    index.assert_not_called()


def test_refresh_removes_deleted_assistant_files_but_preserves_nvidia(monkeypatch):
    from app.ui import handlers as frontend
    monkeypatch.setattr(frontend,"inspect_provider",lambda _: {"provider":"pinecone","label":"Assistant","configured":True,"connected":True,"model":"m","message":"Ready","files":[{"file_id":"new","name":"new.txt","status":"Available"}]})
    receipts={"shared":[{"name":"old.txt","results":{"pinecone":{"file_id":"old","status":"Available"},"nvidia":{"status":"Indexed"}}}]}
    _,updated,rows=frontend.refresh_provider("pinecone",receipts)
    assert rows == [["new.txt","Available"]]
    assert frontend.library_rows("nvidia",updated) == [["old.txt","Indexed"]]
    assert "pinecone" in receipts["shared"][0]["results"]


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
