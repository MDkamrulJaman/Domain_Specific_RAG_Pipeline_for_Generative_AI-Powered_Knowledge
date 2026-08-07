# Frontend Application

A modern Next.js frontend for the domain-specific RAG knowledge assistant. This application provides the user interface for uploading documents, asking questions, and interacting with the backend retrieval and generation pipeline.

## Overview

The frontend is built with Next.js and uses a clean, chat-first interface inspired by AI assistant experiences. It communicates with the FastAPI backend to:

- upload knowledge documents
- create and manage chat sessions
- send user queries to the RAG pipeline
- display generated responses from the backend

## Tech Stack

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS
- shadcn/ui-inspired component patterns
- ESLint
- Docker-ready setup

## Project Structure

```text
frontend/
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── Chat.tsx
│   └── Upload.tsx
├── hooks/
│   └── use-mobile.ts
├── lib/
│   └── utils.ts
├── public/
├── .env.local
├── .gitignore
├── components.json
├── Dockerfile
├── envConfig.ts
├── eslint.config.mjs
├── next-env.d.ts
├── next.config.ts
├── package.json
├── pnpm-lock.yaml
├── pnpm-workspace.yaml
├── postcss.config.mjs
├── README.md
├── tsconfig.json
└── .next/
```

## Features

- AI-style chat interface
- Side navigation for chat history
- Document upload panel for ingestion into the backend
- Real-time interaction with the backend RAG service
- Responsive layout for desktop and modern web use
- CORS-ready communication with the API layer

## Prerequisites

Before running the frontend, ensure you have the following installed:

- Node.js 18+
- pnpm (recommended) or npm
- Access to the backend API running locally or remotely

## Installation

Install the dependencies:

```bash
pnpm install
```

If using npm instead:

```bash
npm install
```

## Environment Configuration

Create a `.env.local` file in the frontend root if your app needs environment-level variables such as API URLs.

Example:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

> The project includes `envConfig.ts` to load environment variables using `@next/env`.

## Running the App

Start the development server:

```bash
pnpm dev
```

Then open the browser at:

```text
http://localhost:3000
```

## Production Build

Build the app:

```bash
pnpm build
```

Run the production server:

```bash
pnpm start
```

## Main Application Flow

1. User opens the chat interface.
2. User uploads a document through the ingestion component.
3. Frontend sends the file to the backend ingestion endpoint.
4. Backend processes the document and stores embeddings in the vector database.
5. User enters a prompt in the chat box.
6. Frontend sends the query to the backend `/chat/` endpoint.
7. Backend retrieves context and returns the response.
8. Frontend displays the answer in the interface.

## API Communication

The frontend currently connects directly to the backend using the following patterns:

- `http://127.0.0.1:8000/chat/`
- `http://127.0.0.1:8000/chat/history`
- `http://127.0.0.1:8000/chat/session`
- `http://127.0.0.1:8000/ingest/upload`

These values should be replaced or managed via environment variables for deployment readiness.

## Components

### Chat

The main user interface component that handles:

- chat input
- prompt submission
- displaying responses
- chat history interaction
- sidebar management

### Upload

Handles document upload to the backend ingestion route and shows upload status feedback.

## Styling

The frontend uses modern utility-based styling with Tailwind and custom design patterns. The app is intentionally styled to provide a polished AI assistant interface with dark-mode UI aesthetics.

## Linting

Run the linter with:

```bash
pnpm lint
```

## Docker

A Dockerfile is provided for containerization.

Build the image:

```bash
docker build -t rag-frontend .
```

Run the container:

```bash
docker run -p 3000:3000 rag-frontend
```

## Notes

- This frontend is designed for local development and demonstration workflows.
- For production use, it is recommended to add:
  - secure environment variable handling
  - authentication and authorization
  - structured API configuration
  - centralized error handling
  - deployment-specific configuration for production domains

## Troubleshooting

### Common Issues

#### 1. Backend connection errors

Check whether the FastAPI backend is running and accessible on the expected port.

#### 2. Upload fails

Verify the backend ingestion route is live and the file type is supported.

#### 3. App fails to start

Ensure dependencies are installed and Node.js version matches project requirements.

## License

This project is intended for educational, research, or internal product use unless explicitly stated otherwise.

---

This frontend provides a clean user experience for interacting with the RAG backend and is a solid foundation for a production-grade AI-powered knowledge application.

