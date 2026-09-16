# RAG Backend

This project is a lightweight backend for document ingestion, retrieval, and answer generation using a local RAG-style workflow. It is designed to help a frontend app or local tool search through uploaded documents and provide grounded responses based on retrieved context.

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
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── app_router/
│   │   │   └── app_router.py
│   │   └── routes/
│   │       ├── chat.py
│   │       └── ingest.py
│   ├── core/
│   │   └── config.py
│   ├── pipeline/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── loader.py
│   │   └── retrieval_service.py
│   ├── schemas/
│   │   ├── chat.py
│   │   └── ingest.py
│   ├── services/
│   │   ├── llm_service.py
│   │   └── rag_service.py
│   └── utils/
│       └── helpers.py
├── data/
│   └── raw/
├── tests/
│   ├── test_api_routes.py
│   ├── test_chat.py
│   ├── test_embeddings.py
│   ├── test_llm_config.py
│   └── test_retrival.py
├── .env
├── Dockerfile
├── README.md
├── requirements.txt
└── .gitignore
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
