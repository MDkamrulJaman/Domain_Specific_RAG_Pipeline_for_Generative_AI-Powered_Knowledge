# ECU RAG System Backend

A FastAPI-based Retrieval-Augmented Generation (RAG) backend for document ingestion, vector search, and LLM-powered Q&A. This service allows users to upload documents, index them into a local vector store, and ask domain-specific questions grounded in the uploaded content.

## Overview

The backend is designed for document-centric knowledge retrieval and question answering using:

- FastAPI for API exposure
- LangChain for document processing and retrieval pipeline
- FAISS for efficient vector similarity search
- OpenAI / LLM-compatible services for generation
- Pydantic for request validation
- Local file-based document ingestion pipeline

This project is intended for use with a frontend application and supports a clean domain-specific knowledge assistant workflow.

## Key Features

- File upload and indexing for PDF documents
- Document chunking and embedding generation
- Vector store persistence using FAISS
- Semantic retrieval for user queries
- LLM-based answer generation with contextual grounding
- REST endpoints for chat and ingestion
- CORS-enabled API for frontend integration
- Health monitoring route

## Tech Stack

- Python 3.10+
- FastAPI
- Uvicorn
- LangChain
- LangChain Community
- OpenAI
- FAISS
- Pydantic / Pydantic Settings
- Python Dotenv
- PyPDF
- pytest

## Project Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── app_router/
│   │   │   ├── __init__.py
│   │   │   └── app_router.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       └── ingest.py
│   ├── core/
│   │   └── config.py
│   ├── ecu_prompts/
│   │   └── prompt.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── loader.py
│   │   ├── retrieval_service.py
│   │   └── tempCodeRunnerFile.py
│   ├── schemas/
│   │   ├── chat.py
│   │   └── ingest.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py
│   │   └── rag_service.py
│   └── utils/
│       └── helpers.py
├── data/
│   ├── raw/
│   └── vectorstore/
├── tests/
│   ├── __init__.py
│   ├── test_api_routes.py
│   ├── test_chat.py
│   └── test_retrival.py
├── .env
├── Dockerfile
├── README.md
├── requirements.txt
└── run.py
```

## Environment Configuration

Create a `.env` file in the backend root using the following structure:

```env
FRONTEND_URL=http://localhost:3000
FRONTEND_PRODUCTION_URL=https://your-production-frontend-url.com
DATABASE_URL=your_database_connection_string
SECRET_KEY=your_secure_secret_key
```

> Note: The actual environment variables used by the app may vary depending on the LLM provider and deployment setup. Make sure the values match your runtime configuration.

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

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

### Production-style run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Root

```http
GET /
```

Returns a welcome message for the backend application.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Chat Endpoint

```http
POST /chat/
```

Request body:

```json
{
  "query": "What is the summary of this document?",
  "top_k": 5
}
```

Response:

```json
{
  "query": "What is the summary of this document?",
  "answer": "Based on the retrieved context..."
}
```

### Document Ingestion

```http
POST /ingest/upload
```

Upload a supported file such as `.pdf` or `.txt` to be processed and indexed into the vector store.

## RAG Pipeline Flow

The system follows this general flow:

1. User uploads a document.
2. File is saved to the local raw data directory.
3. Document is loaded and parsed.
4. Content is chunked into smaller pieces.
5. Embeddings are generated for each chunk.
6. Chunks are stored in a local FAISS vector store.
7. User question is sent to the retrieval and generation pipeline.
8. Relevant chunks are fetched and injected into the LLM prompt.
9. Final answer is returned to the client.

## Data Storage

The application stores:

- uploaded raw files under `data/raw/`
- indexed vector data under `data/vectorstore/`

This setup is suitable for local development and lightweight deployment scenarios.

## Testing

Run the test suite with:

```bash
pytest
```

If using async tests or environment-based setup, ensure the relevant environment variables are loaded correctly before execution.

## Docker Support

A Dockerfile is included for containerized backend deployment.

Build the image:

```bash
docker build -t ecu-rag-backend .
```

Run the container:

```bash
docker run -p 8000:8000 ecu-rag-backend
```

## Notes

- This project is designed for local and demo-grade RAG use cases.
- For production deployment, consider using:
  - a managed vector database
  - secure environment variable management
  - more robust authentication and authorization
  - model rate-limit and retry handling
  - monitoring and observability tools

## License

This project is intended for educational and internal application use unless a separate license is specified.

## Maintainers

This backend is typically maintained by the project owner or development team responsible for the AI and retrieval pipeline.

## Troubleshooting

### Common issues

#### 1. Module import errors

Make sure you are running the app from the backend directory and that dependencies are installed correctly.

```bash
pip install -r requirements.txt
```

#### 2. Frontend CORS errors

Verify that `FRONTEND_URL` in `.env` matches the frontend origin exactly.

#### 3. Vector store initialization fails

Check if the `data/vectorstore/` directory exists and is writable.

#### 4. Upload fails for certain file types

Only allowed file extensions are accepted. Check the `ALLOWED_EXTENSIONS` configuration in the ingestion route.

---

This backend provides a solid foundation for a production-ready domain-specific RAG application and can be scaled further with enterprise-grade storage, authentication, monitoring, and deployment tooling.
