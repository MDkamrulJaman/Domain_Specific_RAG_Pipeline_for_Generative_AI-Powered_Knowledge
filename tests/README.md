# Offline test suite

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest -q
```

Tests use mocked external services and in-process TestClient requests; they do not start a local server. conftest.py disables Gradio analytics and supplies API fixtures. pytest.ini adds backend to the import path.

| File | Coverage |
| --- | --- |
| `test_answer_evidence.py` | Document-first decisions and streaming marker boundaries without network calls. |
| `test_api_routes.py` | Regression checks |
| `test_assistant_service.py` | Pinecone Assistant chat requests and asynchronous file processing. |
| `test_assistant_skills.py` | Pinecone request policies: provider isolation, fallback, and API/UI validation. |
| `test_chat_api.py` | Chat endpoint routing, streaming errors, and per-request options. |
| `test_chat_command.py` | Command behavior shared by HTTP and Gradio invokers. |
| `test_chat_schema.py` | Chat request defaults and input validation. |
| `test_frontend.py` | UI provider selection, streaming callbacks, library state, and dark theme. |
| `test_ingestion_service.py` | Document ingestion routing, progress, and failure isolation. |
| `test_pinecone_only.py` | Regression boundaries for the single-provider application. |
| `test_provider_service.py` | Provider factories, configuration isolation, and readiness checks. |
| `test_service_architecture.py` | Behavioral contracts for dependency injection and provider extension. |
| `test_social_messages.py` | Short social messages avoid provider initialization, retrieval, and search. |
| `test_upload_api.py` | Upload endpoint provider selection, size limits, and file cleanup. |
| `test_web_search.py` | Offline search tests: no credentials, servers, or provider requests needed. |

These checks do not verify live provider credentials, latency, or browser rendering. CI separately checks container startup.
