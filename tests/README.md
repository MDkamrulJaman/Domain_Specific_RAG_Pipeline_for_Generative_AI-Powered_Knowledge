# Test suite guide

Tests are grouped by the behavior they protect. External providers are mocked;
no local server is started. `conftest.py` supplies the shared in-process API
client and disables Gradio analytics.

| File | What it verifies |
| --- | --- |
| `test_api_routes.py` | Router composition and exposed endpoints |
| `test_chat_api.py` | Provider routing, streamed responses, errors, thinking options |
| `test_upload_api.py` | Upload destination, default provider, size limits, file cleanup |
| `test_chat_schema.py` | Request defaults, bounds, and invalid questions |
| `test_provider_service.py` | Provider construction, configuration, credential isolation, status |
| `test_assistant_service.py` | Assistant chat parameters and file processing |
| `test_nvidia_generation.py` | First-token streaming, prompt context, output limits |
| `test_ingestion_service.py` | Ingestion stages, provider isolation, early validation |
| `test_vector_retrieval.py` | Hugging Face vectors and Pinecone integrated text search |
| `test_index_schema.py` | Supported schemas, vector dimensions, SDK compatibility |
| `test_frontend.py` | Provider selection, uploads, library refresh, dark theme |
| `test_service_architecture.py` | Dependency injection, substitutable adapters, registry extension, app factory |
| `test_document_loader.py` | Parsing consistency between disk and in-memory files |

## Running tests

From the repository root, using the backend virtual environment:

```powershell
# Complete suite
backend/.venv/Scripts/python.exe -m pytest -q

# One feature
backend/.venv/Scripts/python.exe -m pytest tests/test_upload_api.py -v

# One behavior
backend/.venv/Scripts/python.exe -m pytest tests/test_nvidia_generation.py -k first_token -v
```

## Adding a test

Choose the file matching the feature. Use a descriptive `test_...` name and
keep setup, action, and assertions together. Use `client` for chat/upload HTTP
tests and `monkeypatch` to replace external services. Put fixtures in
`conftest.py` only when multiple modules need them; keep feature-specific
helpers next to their tests. Do not put real credentials or live API calls in
this offline suite.
