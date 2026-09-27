# RAG Backend

This project is a lightweight backend for document ingestion, retrieval, and answer generation using a local RAG-style workflow. It is designed to help a frontend app or local tool search through uploaded documents and provide grounded responses based on retrieved context.
## Live Preview:  https://generative-ai-powered-retrieval-system-for-technical-d.fastapicloud.dev

## NVIDIA and Pinecone Assistant

The provider dropdown defaults to Pinecone Assistant. The two providers have
separate retrieval systems:

- NVIDIA: PDF/TXT -> chunks -> the configured Pinecone index (`rag`). A
  `text: semantic_text` index uses Pinecone integrated embedding through
  `upsert_records` and text `search`. Dense `_values` indexes still use Hugging
  Face embeddings and vector upsert/query. Retrieved chunks are reranked and
  sent to NVIDIA. Thinking is off by default.
- Pinecone Assistant: original files -> Assistant's managed library -> Assistant
  chat. No external vector search or Hugging Face call is made for this path.

The provider selector controls chat, upload, connection checks, and the document
list. **Pinecone Assistant is the default**. Uploads go only to the selected
provider. Switch to NVIDIA before uploading documents for its index. Assistant
uploads return while processing continues; refresh until files are Available.
The NVIDIA document list shows this session's uploads, while all existing index
records remain searchable. Assistant refresh reads its remote file library.

Assistant chat bypasses vector retrieval, Hugging Face, and external reranking.
NVIDIA's integrated index bypasses Hugging Face calls. Dense-index requests
reuse the Hugging Face client. Single-result searches skip reranking; multi-result
searches retain reranking quality. Both providers stream responses, with NVIDIA
thinking off by default. Actual response latency depends on the remote services.

### Configuration

Use `PINECONE_API_KEY` for the NVIDIA retrieval index and reranking, and
`PINECONE_ASSISTANT_API_KEY` for Assistant. Keep the private settings in
`backend/.env`. Assistant chat requires only Assistant settings; it does not
require an index connection or NVIDIA credentials.

### API

- `POST /ingest/upload`: multipart `file`, optional `provider` (`nvidia`, `pinecone`); default `pinecone`.
  Uploads write only to that provider; failures never switch providers.
- `POST /chat/stream`: `{"query":"Summarize the document", "provider":"pinecone"}`.
  NVIDIA accepts `top_k` and `enable_thinking`; Assistant manages retrieval itself.
- `GET /providers/{provider}?check_connection=true`: index status for NVIDIA,
  Assistant status and file processing state for Pinecone Assistant.

Run offline tests with `python -m pytest`. Tests mock external services.

## Overview
The backend includes:

- a FastAPI application
- document upload and ingestion flow
- chunking and indexing for searchable content
- vector similarity retrieval
- prompt-based answer generation using an LLM
- basic health and API routes

This service is intended for local development or controlled internal use. It is not intended to expose private content or credentials.

## Key Features

- Upload supported documents for indexing
- Split content into manageable chunks
- Generate embeddings for retrieval
- Search the indexed content semantically
- Use retrieved context to answer questions
- Serve a simple API for chat and ingestion
- Run locally or in a containerized environment

## Tech Stack

- Python 3.10+
- FastAPI
- Uvicorn
- LangChain
- FAISS
- Pydantic
- Python Dotenv
- PyPDF or similar document parsing library
- pytest

## Project Structure

```text
Domain_Specific_RAG_Pipeline_for_Generative_AI-Powered_Knowledge
├── .gitignore
├── pytest.ini
├── README.md
├── backend/
│   ├── .env                 # local only; ignored by Git
│   ├── .gitignore
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   ├── api/
│   │   │   ├── app_router/
│   │   │   │   ├── __init__.py
│   │   │   │   └── app_router.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── chat.py
│   │   │       ├── ingest.py
│   │   │       └── providers.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── ecu_prompts/
│   │   │   └── prompt.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py
│   │   │   ├── embedder.py
│   │   │   ├── loader.py
│   │   │   └── retrieval_service.py
│   │   ├── schemas/
│   │   │   ├── chat.py
│   │   │   └── ingest.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── assistant_service.py
│   │   │   ├── llm_service.py
│   │   │   ├── provider_service.py
│   │   │   └── rag_service.py
│   │   ├── ui/
│   │   └── frontend.py
├── tests/
│   ├── __init__.py
│   ├── test_api_routes.py
│   ├── test_index_schema.py
│   └── test_providers.py
```

## Privacy and Security Requirements

This project must be handled with strict privacy controls.

- Never commit secrets, API tokens, passwords, or personal data to the repository.
- Do not upload private documents, user content, or sensitive files to public repositories.
- Keep local data files in a private workspace or secure storage location.
- Use a local environment file only for non-sensitive configuration.
- Ensure uploaded documents are not logged or exposed in error output.
- Review any generated indexes, embeddings, or stored content before sharing the project.
- If deployed in a real environment, use secure storage, access control, and audit logging.

> Important: Never include real credentials or keys in source code, commits, screenshots, or documentation.

## Local Configuration

Create a local environment file only if your deployment needs it. Keep it private and do not commit it.

Example structure only:

```env
APP_ENV=local
APP_PORT=8000
MODEL_ENDPOINT=http://localhost:11434
MODEL_NAME=local-model
STORAGE_PATH=./data
```

Only keep non-sensitive defaults or locally managed values. Do not add secrets or API keys.

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Backend

### Local development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Standard app run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Routes

### Health check

```http
GET /health
```

Returns a basic status message for the application.

### Chat endpoint

```http
POST /chat/
```

Request body:

```json
{
  "query": "Summarize the uploaded documents.",
  "top_k": 5
}
```

### Document ingestion endpoint

```http
POST /ingest/upload
```

Uploads a supported file for processing and indexing.

## RAG Pipeline Flow

The project follows this general flow:

1. A document is uploaded.
2. The file is stored in a local working directory.
3. The content is loaded and processed.
4. Text is split into chunks.
5. Each chunk is embedded.
6. Relevant chunks are retrieved for a given query.
7. The LLM is given retrieved context and produces a response.

## Data Handling

The application may store:

- uploaded files in a local directory
- generated indexes or vector data in a local folder
- temporary processing files during local runs

This project should be used only with data that is approved for local processing and not with highly sensitive information unless proper safeguards are in place.

## Testing

Run the test suite with:

```bash
pytest
```

## Docker

Build the image:

```bash
docker build -t rag-backend .
```

Run the container:

```bash
docker run -p 8000:8000 rag-backend
```

## Notes

- This project is intended for local or controlled deployment scenarios.
- For production use, add stronger access control, secure secret management, and monitored storage.
- Avoid exposing user-uploaded content in public logs, demos, or screenshots.

## License

This project is provided for educational, internal, or prototype use unless a separate license is applied.

## Troubleshooting

### Import or dependency issues

```bash
pip install -r requirements.txt
```

### Local runtime issues

- confirm the virtual environment is active
- check the app startup command
- verify the application is running from the backend directory
- confirm local directories are writable

### Upload problems

- confirm the file type is supported
- validate the local storage path is available
- check application logs for any validation errors

---

This project is a basic RAG backend template and should be used with careful attention to privacy, safe data handling, and local-only configuration.


### Pinecone v10: missing index dimension

The SDK now describes vector dimensions in `index.schema.fields`. A
`semantic_text` index uses Pinecone integrated embeddings; the application
automatically selects text upsert/search rather than Hugging Face vectors. The app inspects the schema explicitly and shows
configuration errors in chat, uploads, and connection checks.

For a Hugging Face vector index, select or create an index
with one dense-vector field whose dimension matches your Hugging Face model
and `PINECONE_DIMENSION`. Point `PINECONE_INDEX_NAME` at that index and upload
original documents again. The NVIDIA retrieval path will use the selected index.
Changing only the configured dimension does not change an existing index's
schema. No remote index is created, modified, or deleted automatically.


For the vector `upsert`/`query` interface used by this application, the dense
field must be the reserved `_values` field. A custom named dense field belongs
to the Documents API and is not supported by this pipeline. The configured
shared index is shown by the UI connection check. Index validation happens
before upload embedding requests, and malformed vector widths are rejected
before upsert. When switching from a semantic-text index to a Hugging Face
vector index, re-upload the original files; vectors from different embedding
models are not interchangeable, even when their dimensions match.


### NVIDIA response latency

NVIDIA prompts request concise answers (normally 150 words, with detail when
requested or necessary). `MODEL_MAX_TOKENS` in `backend/.env` caps generation;
1024 is the configured response limit. Increase it for long answers. The UI
reports when a response hits that limit. `MODEL_ENABLE_THINKING=false` keeps
reasoning off unless explicitly enabled in the UI. NVIDIA logs distinguish
first-token latency from total generation time. A shorter output limit does
not force a short answer or remove upstream queueing delays. These settings
do not affect Pinecone Assistant.
