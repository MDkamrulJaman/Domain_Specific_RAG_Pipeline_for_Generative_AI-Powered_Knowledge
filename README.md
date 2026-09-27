# RAG Knowledge Assistant

A FastAPI + Gradio workspace for technical documents, with Pinecone Assistant
and NVIDIA. The interface starts in dark mode; use the Dark mode toggle to
switch. Pinecone Assistant is selected by default.

## How it works

The selected provider controls **upload, chat, library, and connection checks**.
An upload never silently switches providers or writes to both destinations.

| Provider | Documents | Retrieval | Generation |
| --- | --- | --- | --- |
| Pinecone Assistant | Assistant file library | Managed by Assistant | Assistant chat |
| NVIDIA | Configured Pinecone index (currently `rag`) | Integrated text search + reranking, or HF vectors for dense indexes | NVIDIA streaming API |

Assistant files must finish processing before chat. Refresh the file library
until their status is Available. NVIDIA's list shows session uploads; existing
index records remain searchable. Each question is independent of chat history.

## Project map

```text
backend/
  app/
    main.py                   FastAPI entry point and UI mount
    core/config.py            Environment settings and operational limits
    api/app_router/           Router composition
    api/routes/               Thin HTTP endpoints
    schemas/                  Request validation
    services/
      provider_service.py     Provider selection and cached clients
      ingestion_service.py    Upload validation and routing
      assistant_service.py    Assistant files and streamed answers
      rag_service.py          NVIDIA retrieval and grounded prompts
      llm_service.py          NVIDIA streaming client
    pipeline/                 Parsing, chunking, embeddings, index access
    ui/
      frontend.py             Components and event wiring
      handlers.py             UI request handlers and session state
      styles.py               Theme, header, and empty state
      theme.css               Responsive styling
  requirements.txt            Dependency source of truth
  Dockerfile                  Non-root runtime image
  .dockerignore               Excludes credentials, virtualenv, caches
 tests/                       Offline routing, streaming, limits, UI checks
```

## Configuration

Settings load from `backend/.env` regardless of the working directory; process
environment variables override that file. Keep credentials private and untracked.

- Assistant: `PINECONE_ASSISTANT_API_KEY`, `PINECONE_ASSISTANT_NAME`,
  `PINECONE_ASSISTANT_MODEL`, `PINECONE_ASSISTANT_TIMEOUT_SECONDS`.
- NVIDIA: `MODEL_API_KEY`, `MODEL_BASE_URL`, `MODEL_NAME`,
  `MODEL_TIMEOUT_SECONDS`, `MODEL_TEMPERATURE`, `MODEL_TOP_P`,
  `MODEL_MAX_TOKENS` (1024 currently), `MODEL_ENABLE_THINKING` (false by default).
- Index: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`,
  `PINECONE_DIMENSION`, `PINECONE_MODEL` (reranker), `EMBEDDING_MODEL`, `HF_TOKEN`.
- Operations: `MAX_UPLOAD_MB=20`, `UI_QUEUE_SIZE=32`, `UI_CONCURRENCY=4`.

A `text: semantic_text` index uses Pinecone-managed embeddings. A dense `_values`
index uses Hugging Face vectors with matching dimensions. The application does
not create, delete, or migrate remote indexes at startup.

## Run locally

From `backend`, install `pip install -r requirements.txt`, then run:

```sh
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `/` for the workspace, `/docs` for API documentation, `/health` for liveness.
For development only, add `--reload`.

## API

- `POST /chat/stream`: JSON `query`, `provider` (defaults to `pinecone`),
  optional NVIDIA `top_k` (1–50) and `enable_thinking`.
- `POST /ingest/upload`: multipart `file` and `provider` (defaults to `pinecone`).
- `GET /providers/{provider}`: public configuration summary; add
  `?check_connection=true` for a remote status check.
- `GET /health`: local liveness; does not certify provider availability.

PDF and TXT uploads have a configurable size limit. Questions must be nonblank
and no longer than 12,000 characters. Provider errors are reported without
exposing credentials. Streaming errors appear in the response body once HTTP
headers have been sent.

## Performance

Clients are reused. Assistant bypasses the NVIDIA retrieval pipeline. Integrated
indexes skip Hugging Face requests; single-result searches skip reranking.
NVIDIA prompts request concise answers and log first-token and total latency.
Output-limit notices explain incomplete answers. Increase `MODEL_MAX_TOKENS`
when longer answers are needed. Hosted-provider latency is outside this app's control.

## Deployment

```sh
docker build -t rag-assistant backend
docker run --env-file backend/.env -p 8000:8000 rag-assistant
```

The image runs as a non-root user with one worker because Gradio queue/session
state is process-local. Use sticky sessions and an appropriate shared-state
strategy before scaling horizontally. Temporary UI uploads are eligible for
cleanup after 24 hours, checked hourly. Backend API ingestion parses in memory.

Before public exposure, deploy behind authenticated HTTPS access with rate and
request-body limits. The app does **not** implement authentication, per-user
library isolation, or distributed rate limiting. Provider libraries are shared
within the configured account/namespace. Queue and upload limits are resource
bounds, not a substitute for access control. This code has offline tests; load,
security, container, and browser validation remain deployment responsibilities.

## Tests

From the repository root using the backend virtual environment:

```sh
python -m pytest -q
```

Tests mock remote providers and build UI components without starting a server.

See [the test suite guide](tests/README.md) for the feature-by-feature test map and focused test commands.
