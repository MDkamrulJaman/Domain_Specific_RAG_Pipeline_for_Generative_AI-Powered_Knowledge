# Domain-Specific RAG Pipeline for Generative AI-Powered Knowledge

<div align="center">
  <img src="https://img.shields.io/badge/Project-TypeScript%20%2B%20Python-blue" alt="Project stack" />
  <img src="https://img.shields.io/badge/Frontend-Next.js%2016-000000" alt="Frontend" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688" alt="Backend" />
  <img src="https://img.shields.io/badge/AI-RAG%20Pipeline-7C3AED" alt="RAG" />
  <img src="https://img.shields.io/badge/VectorDB-FAISS-FF6B6B" alt="Vector DB" />
</div>

## Overview

This project is a domain-specific Retrieval-Augmented Generation (RAG) system designed to enable intelligent, context-aware question answering over uploaded documents. It combines modern generative AI techniques with an efficient retrieval pipeline to provide grounded, reliable, and explainable responses from domain knowledge sources.

The solution is built using a full-stack architecture with a Next.js frontend for user interaction and a FastAPI backend for document ingestion, vector search, and LLM-based answer generation. It is optimized for knowledge retrieval in scenarios where users need answers grounded in uploaded business, technical, or domain-specific documents.

---

## Why This Project Matters

Traditional chat systems often rely only on general knowledge and may hallucinate when asked domain-specific questions. This project addresses that limitation by:

- ingesting and indexing user documents into a vector store,
- retrieving the most relevant content using semantic search,
- constructing prompts grounded in retrieved context,
- generating precise answers using an LLM while maintaining domain relevance.

This makes the system highly suitable for internal knowledge assistants, support documentation search, technical document Q&A, and enterprise knowledge retrieval workflows.

---

## Key Features

- Document upload and processing for PDF and text files
- Local vector database support using FAISS
- Chunk-based semantic indexing for efficient retrieval
- Retrieval-Augmented Generation pipeline for answer generation
- FastAPI-based backend with structured API endpoints
- Next.js frontend for a clean user experience
- CORS-enabled integration between frontend and backend
- Health check and monitoring support
- Modular architecture for scalability and extension
- Environment-based configuration for secure deployment

---

## System Architecture

The application is structured into two main layers:

1. Frontend Layer
   - Built using Next.js
   - Supports user upload and chat interactions
   - Connects to backend APIs for document ingestion and Q&A

2. Backend Layer
   - Built with FastAPI
   - Handles file upload, parsing, chunking, embedding generation, and retrieval
   - Uses a local vector store and retrieval pipeline to serve contextual answers

### High-Level Flow

```text
User Uploads Document
        ↓
Frontend Sends File to API
        ↓
Backend Stores Document
        ↓
Document Is Loaded and Split into Chunks
        ↓
Embeddings Are Generated
        ↓
Chunks Are Stored in FAISS Vector Store
        ↓
User Query Is Sent to Retrieval + LLM Pipeline
        ↓
Relevant Context Is Retrieved
        ↓
LLM Produces a Context-Grounded Answer
        ↓
Answer Is Returned to the User
```

---

## Technology Stack

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS
- shadcn-style UI patterns

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic
- Python Dotenv

### AI and Retrieval Stack
- LangChain
- LangChain Community
- LangChain OpenAI
- LangChain Ollama
- FAISS
- Sentence Transformers
- OpenAI-compatible LLM integration

### Document Processing
- PyPDF
- Python Docx
- Unstructured documents support
- Chunking and embedding pipeline

### Testing and Quality
- Pytest
- Async testing support

---

## Project Structure

```text
Domain_Specific_RAG_Pipeline_for_Generative_AI-Powered_Knowledge/
├── README.md
├── backend/
│   ├── .env
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── app_router/
│   │   │   └── routes/
│   │   ├── core/
│   │   ├── ecu_prompts/
│   │   ├── pipeline/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── utils/
│   ├── data/
│   │   ├── raw/
│   │   └── vectorstore/
│   └── tests/
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── lib/
│   ├── public/
│   ├── .env.local
│   ├── package.json
│   ├── tsconfig.json
│   └── next.config.ts
└── data/
    ├── raw/
    └── vectorstore/
```

---

## Core Modules

### Backend

- `app/main.py`  
  Main FastAPI application entry point with CORS configuration and app initialization.

- `app/api/routes/chat.py`  
  Handles user queries and invokes the retrieval generation workflow.

- `app/api/routes/ingest.py`  
  Accepts uploaded documents, validates input, stores files, and indexes them into the vector database.

- `app/services/rag_service.py`  
  Coordinates retrieval and LLM-based answer generation.

- `app/pipeline/loader.py`  
  Loads documents from disk or uploaded sources.

- `app/pipeline/chunker.py`  
  Splits long content into manageable chunks.

- `app/pipeline/embedder.py`  
  Converts chunks into embeddings for semantic search.

- `app/pipeline/retrieval_service.py`  
  Performs similarity search against FAISS index and retrieves relevant documents.

### Frontend

- `frontend/app/page.tsx`  
  Main landing screen for the chat UI.

- `frontend/components/Chat.tsx`  
  User-facing chat interface.

- `frontend/components/Upload.tsx`  
  File upload interface for document ingestion.

---

## Features in Practice

This project supports a real-world workflow similar to:

- uploading technical manuals,
- indexing internal policy documents,
- answering questions using the most relevant passages,
- reducing dependency on static, pre-trained general answers,
- enabling retrieval from a curated set of domain documents.

This makes it especially valuable for sectors including:

- education,
- research and development,
- enterprise knowledge management,
- technical support,
- document-heavy operations.

---

## Prerequisites

Before running the project, ensure the following are installed:

- Python 3.10+
- Node.js 18+
- npm or pnpm
- Git
- Access to a compatible LLM backend or OpenAI API key

---

## Environment Setup

### Backend Environment

Create a `.env` file inside the `backend` directory with the required values.

Example:

```env
FRONTEND_URL=http://localhost:3000
FRONTEND_PRODUCTION_URL=https://your-frontend-url.com
DATABASE_URL=your_database_connection_string
SECRET_KEY=your_secure_secret_key
OLLAMA_BASE_URL=http://localhost:11434/
OLLAMA_MODEL=gemma3
OLLAMA_TIMEOUT_SECONDS=30.0
```

### Frontend Environment

The frontend uses `.env.local` for client-side runtime configuration.

Example:

```env
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
OPEN_AI_API_KEY=
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MDkamrulJaman/Domain_Specific_RAG_Pipeline_for_Generative_AI-Powered_Knowledge.git
cd Domain_Specific_RAG_Pipeline_for_Generative_AI-Powered_Knowledge
```

### 2. Set up the backend

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

- Windows:

```bash
.venv\Scripts\activate
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set up the frontend

```bash
cd ../frontend
npm install
```

If you are using pnpm:

```bash
pnpm install
```

---

## Running the Application

### Start the backend

From the `backend` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the frontend

From the `frontend` directory:

```bash
npm run dev
```

or

```bash
pnpm dev
```

The frontend typically runs on:

```text
http://localhost:3000
```

The backend API typically runs on:

```text
http://localhost:8000
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Returns backend status and version information.

### Chat Query

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

### File Ingestion

```http
POST /ingest/upload
```

Uploads a supported document and indexes it into the vector store.

---

## Example Workflow

1. Open the frontend application.
2. Upload a PDF or text document.
3. The backend parses and chunks the file.
4. The vector store stores the document embeddings.
5. Ask a question in natural language.
6. The system retrieves the relevant content and generates a grounded answer.

---

## Validation and Testing

The backend includes test coverage for:

- API route behavior
- chat request handling
- LLM configuration checks
- retrieval logic verification

To run tests:

```bash
cd backend
pytest
```

---

## Production Readiness and Scalability Considerations

This project is structured to support future enhancements such as:

- cloud deployment with Docker and Kubernetes,
- persistent database storage for metadata,
- support for multiple knowledge domains,
- authentication and authorization,
- role-based access control,
- improved prompt optimization and evaluation,
- enterprise monitoring and observability.

---

## Project Impact

This project demonstrates the practical application of generative AI in a business-relevant environment by connecting:

- AI-powered Q&A,
- knowledge retrieval,
- document intelligence,
- modern web application interfaces.

It reflects a strong understanding of retrieval-augmented generation, full-stack integration, and AI system design.

---

## Future Roadmap

Planned improvements include:

- support for more file formats and larger document collections,
- multi-user and tenant-aware architecture,
- vector store persistence improvements,
- richer analytics dashboards,
- advanced prompt evaluation and answer quality monitoring,
- API security hardening for production deployment.

---

## Conclusion

The Domain-Specific RAG Pipeline for Generative AI-Powered Knowledge is a modern AI application that brings together document processing, retrieval systems, and large language models to create a useful and practical knowledge assistant. The project reflects strong technical capabilities in full-stack development, AI integration, and software engineering best practices.

It is designed not only as a functional prototype but also as a scalable foundation for real-world knowledge retrieval solutions.

---

## License

This project is intended for educational, research, and professional demonstration purposes. Please review repository policies before commercial deployment or redistribution.

---

## Contact

For questions or collaboration opportunities, please reach out through the project repository or the contact details associated with the maintainer.

---

This README was prepared to present the project in a professional, reviewer-friendly format suitable for technical evaluation and HR review.
