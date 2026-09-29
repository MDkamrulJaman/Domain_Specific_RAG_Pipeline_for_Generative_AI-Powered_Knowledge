# Technical Document Knowledge Assistant

A retrieval-augmented generation (RAG) application for asking questions about technical documents. FastAPI provides the API, Gradio provides the browser workspace, and hosted AI services provide retrieval and generation.

**Pinecone Assistant is the default provider.** Users can switch to NVIDIA for a configurable retrieval workflow. The interface starts in dark mode.

## Contents

- [Capabilities](#capabilities)
- [Provider workflows](#provider-workflows)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [Local setup](#local-setup)
- [Configuration](#configuration)
- [Using the workspace](#using-the-workspace)
- [NVIDIA answering](#nvidia-answering)
- [API reference](#api-reference)
- [Testing](#testing)
- [Docker and CI/CD](#docker-and-cicd)
- [Operational considerations](#operational-considerations)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

## Capabilities

- PDF and TXT uploads with configurable size validation.
- Provider-specific uploads, chat, connection checks, and library views.
- Streamed answers with first-token and elapsed-time feedback.
- Responsive Gradio workspace with dark/light themes and clickable question suggestions.
- Lazy provider construction and cached service clients.
- Separate API, application services, provider adapters, and document-processing modules.
- Feature-organized offline tests and automated container checks in GitHub Actions.

The application calls hosted models; it does not download or run NVIDIA model weights locally.

## Provider workflows

| Behavior              | Pinecone Assistant                   | NVIDIA                                                           |
| --------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| Document destination  | Assistant file library               | Configured Pinecone index and namespace                          |
| Parsing and retrieval | Managed by Assistant                 | Application parsing/chunking plus index search                   |
| Embeddings            | Managed by Assistant                 | Pinecone integrated embeddings or hosted Hugging Face embeddings |
| Answer generation     | Assistant chat API                   | NVIDIA API through an OpenAI-compatible client                   |
| Library display       | Remote Assistant files after refresh | Upload receipts from the current UI session                      |

### Pinecone Assistant

```text
PDF/TXT -> validate upload -> Assistant file library -> processing
Question -> Assistant managed retrieval and generation -> streamed answer
```

Wait for uploaded files to become available before asking questions. Use **Refresh connection & files** to check their state.

### NVIDIA

```text
PDF/TXT -> validate -> extract text -> split into chunks
    -> integrated text records OR Hugging Face vectors -> Pinecone index

Question -> index search -> optional reranking -> grounded answer prompt
    -> NVIDIA generation -> streamed answer
```

The index schema determines the embedding path. A `text: semantic_text` field uses Pinecone-managed embeddings. The supported dense-vector path uses Hugging Face vectors with a matching index dimension; document-schema indexes must expose the supported `_values` field. Schema checks reject incompatible configurations before indexing.

Uploads go only to the selected provider. Assistant files and NVIDIA index records are separate; the application does not synchronize them or silently fall back to another provider. Existing NVIDIA index records remain searchable even if they are absent from the current session's upload list. Startup does not create, delete, or migrate remote indexes.

## Project structure

```text
.
|-- .github/
|   `-- workflows/ci-cd.yml       Tests, Docker checks/publication, cloud deployment
|-- backend/
|   |-- app/
|   |   |-- main.py              ASGI entry point
|   |   |-- application.py       Application factory, health endpoint, UI mounting
|   |   |-- api/
|   |   |   |-- app_router/      Router composition and compatibility exports
|   |   |   `-- routes/          Chat, upload, and provider HTTP endpoints
|   |   |-- core/config.py       Settings grouped by provider and operational concern
|   |   |-- schemas/             Validated API data models
|   |   |-- services/            Application workflows, contracts, provider adapters
|   |   |-- pipeline/            Parsing, splitting, embeddings, and index operations
|   |   |-- ui/                  Gradio layout, handlers, theme, and CSS
|   |   `-- requirements.txt     Compatibility include for ../requirements.txt
|   |-- requirements.txt         Dependencies used by pip, tests, and Docker
|   |-- pyproject.toml           Cloud project metadata, dependencies, ASGI entry point
|   |-- Dockerfile              Multi-stage, non-root Python 3.11 container
|   |-- .dockerignore           Excludes secrets, caches, and local deployment files
|   `-- .gitignore              Backend-specific source-control exclusions
|-- tests/                      Offline tests organized by feature
|   |-- conftest.py             Shared fixtures and in-process API client
|   `-- README.md               Test coverage map and focused commands
|-- pytest.ini                  Test discovery and backend import path
|-- ARCHITECTURE.md              SOLID principles and extension guidance
|-- .gitignore                  Repository-wide exclusions
`-- README.md                   Project guide
```

Python `__init__.py` files establish packages or expose public imports. Local `.venv`, `.env`, `.fastapicloud`, cache directories, and editor-generated `tempCodeRunnerFile.py` files are development artifacts, not application modules to deploy. Do not commit credentials or local environment state.

### API and schemas

| Module                           | Responsibility                                                     |
| -------------------------------- | ------------------------------------------------------------------ |
| `api/app_router/app_router.py` | Composes routers; preserves legacy router aliases                  |
| `api/routes/chat.py`           | Validates chat requests and returns streamed text                  |
| `api/routes/ingest.py`         | Accepts multipart uploads, bounds reads, and closes uploaded files |
| `api/routes/providers.py`      | Reports configuration or explicitly checks remote connectivity     |
| `schemas/chat.py`              | Validates provider, question, retrieval count                      |
| `schemas/ingest.py`            | Defines ingestion data models                                      |

### Services

| Module                   | Responsibility                                                         |
| ------------------------ | ---------------------------------------------------------------------- |
| `contracts.py`         | Small protocols for chat, generation, retrieval, parsing, and indexing |
| `provider_registry.py` | Provider definitions and lookup                                        |
| `provider_service.py`  | Wires adapters, caches clients, and provides readiness summaries       |
| `assistant_service.py` | Assistant chat, file upload, listing, and status                       |
| `rag_service.py`       | Retrieves context and coordinates NVIDIA generation                    |
| `llm_service.py`       | NVIDIA streaming, token limits, timing, and stream cleanup             |
| `ingestion_service.py` | Validates uploads and dispatches to the selected provider              |
| `indexing_service.py`  | Coordinates loader, chunker, optional embedder, and index writer       |
| `prompts.py`           | Builds grounded prompts with deduplicated passages                     |
| `skills.py`            | Inactive reference; unused by the application                          |

The shared `services/chat_command.py` module implements an immutable answer Command used by both HTTP and Gradio. See [the full GoF pattern catalog](ARCHITECTURE.md#design-pattern-policy) for all 23 patterns, their applicability, and the preserved SOLID boundaries.

### Document pipeline and UI

| Module                            | Responsibility                                                 |
| --------------------------------- | -------------------------------------------------------------- |
| `pipeline/loader.py`            | Parses TXT and text-bearing PDFs from bytes or files           |
| `pipeline/chunker.py`           | Splits documents into retrieval chunks                         |
| `pipeline/embedder.py`          | Calls hosted Hugging Face inference through a reusable client  |
| `pipeline/index_schema.py`      | Checks supported vector schemas and dimensions                 |
| `pipeline/retrieval_service.py` | Writes records, searches, and reranks results                  |
| `ui/frontend.py`                | Builds components and connects their events                    |
| `ui/handlers.py`                | Handles streamed answers, uploads, and session library updates |
| `ui/styles.py`                  | Defines theme configuration and presentation assets            |
| `ui/theme.css`                  | Styles the responsive workspace                                |

## Architecture

The project applies SOLID principles through focused modules, small protocols, a provider registry, and dependency injection. RAG and indexing workflows receive their dependencies instead of constructing SDK clients internally. Provider-specific capabilities remain separate; an Assistant file adapter does not need to implement vector indexing.

The API and UI call the same application services. `create_app(include_ui=False)` supports API-only tests; the normal ASGI entry point mounts the browser workspace.

See [Architecture and SOLID principles](ARCHITECTURE.md) for dependency boundaries and instructions for adding providers.

## Local setup

Deployment targets **Python 3.11**. Use that version locally to match CI and the container. Git is required for the release workflow. Docker Desktop is optional: GitHub-hosted runners build and test containers online.

From the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt pytest
```

Create an untracked `backend/.env` and configure the providers you intend to use. Environment-variable names are documented below; no actual credentials are included in this repository guide.

To start the application yourself:

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the workspace or `/docs` for interactive API documentation. Add `--reload` only during development.

## Configuration

Settings load `backend/.env` independently of the current working directory. Process environment variables take precedence. Restart the app after changing provider settings because service instances are cached.

| Group              | Environment variables                                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pinecone Assistant | `PINECONE_ASSISTANT_API_KEY`, `PINECONE_ASSISTANT_NAME`, `PINECONE_ASSISTANT_MODEL`, `PINECONE_ASSISTANT_TIMEOUT_SECONDS`                    |
| NVIDIA generation  | `MODEL_API_KEY`, `MODEL_BASE_URL`, `MODEL_NAME`, `MODEL_TIMEOUT_SECONDS`, `MODEL_TEMPERATURE`, `MODEL_TOP_P`, `MODEL_MAX_TOKENS`       |
| NVIDIA retrieval   | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `PINECONE_DIMENSION`, `PINECONE_MODEL`, `EMBEDDING_MODEL`, `HF_TOKEN` |

`PINECONE_MODEL` selects the reranker. `PINECONE_DIMENSION` must match dense embeddings when that path is used. Current NVIDIA configuration validation still requires the retrieval settings listed above, even when integrated embeddings skip Hugging Face inference calls. Assistant configuration does not require NVIDIA credentials.

| Optional setting          | Default   | Meaning                                         |
| ------------------------- | --------- | ----------------------------------------------- |
| `MODEL_ENABLE_THINKING` | `false` | Default NVIDIA reasoning option                 |
| `MAX_UPLOAD_MB`         | `5`     | Upload limit; converted using 1024 x 1024 bytes |
| `UI_QUEUE_SIZE`         | `8`     | Gradio queue capacity                           |
| `UI_CONCURRENCY`        | `1`     | Default Gradio event concurrency                |

These are conservative defaults for a small hosting instance, not a guarantee of memory usage. UI concurrency does not limit direct API calls or guarantee a single in-flight operation across every event type. Existing cloud environment values override defaults.

Keep `backend/requirements.txt` and `backend/pyproject.toml` dependency declarations synchronized; CI checks their equality. `backend/app/requirements.txt` forwards to the backend requirements file. Use `pinecone`, not the deprecated `pinecone-client` package.

## Using the workspace

1. Select **Pinecone Assistant** or **NVIDIA model**.
2. Refresh the connection and inspect readiness.
3. Upload a PDF or TXT file to the selected provider.
4. For Assistant, wait for processing to finish. For NVIDIA, wait for indexing to complete.
5. Enter a question or click a suggested question to submit it.
6. For NVIDIA, five passages are retrieved automatically; thinking mode remains optional.

Switching providers changes the destination for subsequent operations. Questions are independent: visible chat history is not sent as conversational context to either provider.

## NVIDIA answering

Questions go directly into the grounded answer prompt. The frontend retrieves up to five passages automatically; API clients can override `top_k` (1-50), with null selecting five. Assistant manages its own retrieval. Generic document context is bounded to 16,000 characters, and model output uses `MODEL_MAX_TOKENS`. Existing NVIDIA web-mode budgets still apply when web fallback is enabled.

`services/skills.py` is retained as an inactive reference file. Nothing in the application imports it. There is no skill field, routing, or task-specific policy in the API, UI, commands, prompts, or services. Summaries and explanations are requested through the question itself. Older clients sending an extra `skill` property receive the same generic behavior because unknown request properties are ignored.

Passage labels identify supplied context, not verified page citations. Summaries cover retrieved passages, not necessarily the whole document.

## API reference

| Method | Path                      | Purpose                                            |
| ------ | ------------------------- | -------------------------------------------------- |
| GET    | `/`                     | Gradio workspace                                   |
| GET    | `/api`                  | API welcome response                               |
| GET    | `/health`               | Application liveness                               |
| GET    | `/docs`                 | Interactive OpenAPI documentation                  |
| POST   | `/chat/stream`          | Stream an answer as plain text                     |
| POST   | `/ingest/upload`        | Upload multipart`file` and optional `provider` |
| GET    | `/providers/{provider}` | Configuration summary                              |

Add `?check_connection=true` to a provider request to perform a remote status check. Liveness alone does not verify provider credentials or availability.

Example NVIDIA chat body:

```json
{
  "query": "What startup requirements are stated in the documents?",
  "provider": "nvidia",
  "top_k": 5,
  "enable_thinking": false
}
```

Provider defaults to `pinecone`, retrieval count to `5`. Questions must be nonblank and at most 12,000 characters. API retrieval counts range from 1 to 50; the frontend always uses the default retrieval count. Streaming failures after headers are sent appear in the response body rather than changing the HTTP status.

## Testing

From the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest -q
.\backend\.venv\Scripts\python.exe -m pytest tests/test_rag_options.py -v
.\backend\.venv\Scripts\python.exe -m pip check
```

`pytest.ini` selects `tests/` and adds `backend/` to the import path. Tests cover routing, upload limits and cleanup, provider isolation, retrieval modes, schema validation, streaming, UI wiring, and dependency-injection contracts. External services are mocked, and API calls use an in-process test client.

See [the test suite guide](tests/README.md) for the file-by-file map. Offline tests do not validate real provider responses, browser interaction, or production load. CI separately starts a container to check its health and frontend response.

## Docker and CI/CD

### Container design

The [Dockerfile](backend/Dockerfile) uses separate build and runtime stages based on Python 3.11 Debian slim. It installs dependencies into a virtual environment, checks dependency consistency, copies application source and UI assets, and runs Uvicorn as a non-root user with one worker.

The health check requests `http://127.0.0.1:8000/health` **inside the container**. This address does not refer to the developer's computer or the public deployment URL. The build context is `backend/`, so root-level tests are excluded. The runtime does not include a local `.env` file.

### Automated release

[`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) runs on GitHub-hosted Ubuntu runners. Choose the work for a push to `main` using markers in the head commit message:

| Choice    | Commit marker               | Python tests and dependency checks | Docker build and smoke checks | Docker Hub publication and FastAPI Cloud deployment |
| --------- | --------------------------- | ---------------------------------- | ----------------------------- | --------------------------------------------------- |
| Test      | `[test]`                  | Yes                                | No                            | No                                                  |
| Build     | `[build]`                 | Yes                                | Yes                           | No                                                  |
| Deploy    | `[deploy]`                | Yes                                | Yes                           | Yes                                                 |
| All three | `[test] [build] [deploy]` | Yes                                | Yes                           | Yes                                                 |
| Default   | No marker                   | Yes                                | Yes                           | No                                                  |

Build includes tests, and deploy includes tests, Docker health checks, and the frontend HTTP check. Combining markers selects the furthest stage: `[test] [build]` runs tests and Docker checks; any combination containing `[deploy]` requests the full release. A failed job blocks downstream publication and deployment.

Pull requests targeting `main` always run tests and Docker checks, regardless of markers, and never publish or deploy. Pushes to other branches do not trigger this workflow.

Images receive `sha-<commit>` and `latest` tags; `latest` identifies the latest published image, not necessarily a successful cloud deployment. FastAPI Cloud builds uploaded source separately; this workflow does not deploy the Docker Hub image to FastAPI Cloud. Docker Desktop is not required on the local machine.

### Choose test, build, or deploy on a push

Stage the intended files with `git add path/to/changed-file`, replacing that example path with your changed files. Then choose **one** of these four examples while on `main`:

**1. Tests only**

```powershell
git commit -m "Check document parsing [test]"
git push origin main
```

**2. Tests and Docker build checks**

```powershell
git commit -m "Verify container changes [build]"
git push origin main
```

**3. Full release**

```powershell
git commit -m "Release document improvements [deploy]"
git push origin main
```

**4. Explicitly request all three**

```powershell
git commit -m "Release current version [test] [build] [deploy]"
git push origin main
```

An ordinary commit without any marker keeps the default test-and-build behavior without publication or deployment. If the code is already committed, an empty commit can request a release:

```powershell
git commit --allow-empty -m "Deploy current version [deploy]"
git push origin main
```

Only the **head (latest) commit message of the push** is checked, including its body. Markers in earlier commits in the same push do not determine the selected stages. Matching is case-insensitive. For a merge or squash, put the desired markers in the final merge or squash commit message. Deployment still requires every preceding check to pass.

Keep FastAPI Cloud's separate automatic source-repository deployment disconnected so ordinary pushes cannot bypass this opt-in rule. These markers control the GitHub workflow; they do not disable independently configured deployment triggers.

### Deployment configuration

Configure these repository **Actions secrets** using your own values:

| Secret                   | Purpose                                        |
| ------------------------ | ---------------------------------------------- |
| `DOCKERHUB_USERNAME`   | Docker Hub account owning the image repository |
| `DOCKERHUB_TOKEN`      | Token authorized to publish images             |
| `FASTAPI_CLOUD_TOKEN`  | Deployment token for the cloud application     |
| `FASTAPI_CLOUD_APP_ID` | Full UUID of the target application            |

The workflow publishes to `<Docker Hub username>/rag-knowledge-assistant`; create that repository or update the workflow's image name. Set the FastAPI Cloud **Application Directory** to `backend` and configure provider credentials in its environment settings. Avoid a second automatic deployment trigger that bypasses the workflow's test gates.

Keep deployment credentials in GitHub secrets and runtime credentials in the hosting environment. Do not paste real values into YAML, source files, examples, or documentation.

After reviewing changes, push to `main` or merge a checked pull request. Include `[deploy]` in the final commit message only when a release is intended. Monitor GitHub Actions; after a release, verify the live health endpoint, uploads, and chat for both providers.

## Operational considerations

- Gradio uses process-local queue/session state. Verify session routing and shared-state requirements before deploying multiple workers or replicas.
- Uploaded PDF parsing can consume more memory than the file size. The NVIDIA PDF loader extracts text but does not implement OCR for scanned images.
- Gradio temporary uploads are eligible for cleanup after 24 hours, checked hourly. API document parsing uses in-memory content.
- Provider clients are reused. Integrated index queries skip Hugging Face calls; single-result searches skip reranking. NVIDIA streams tokens and records timing information.
- Output/context limits reduce request size; they do not guarantee faster hosted-model responses. No cross-request retrieval-result cache is implemented.
- Hosting scale-to-zero can add cold-start latency. Check actual resource usage and provider latency under representative traffic.
- Authentication, per-user library isolation, and distributed rate limiting are not implemented. Provider libraries are shared within the configured account/namespace. Protect public access and apply appropriate request limits before handling private documents.
- Dependencies have compatibility ranges but are not fully locked. The Docker and cloud builds are separate, so exact dependency reproducibility is not guaranteed.

## Troubleshooting

| Symptom                              | Check                                                                             |
| ------------------------------------ | --------------------------------------------------------------------------------- |
| Provider shows setup required        | Required environment-variable names and values; restart after changes             |
| Assistant cannot find NVIDIA uploads | Upload to Assistant separately; storage is provider-specific                      |
| Assistant files are still processing | Refresh until the remote files become available                                   |
| Dense-vector dimension/schema error  | Index schema, embedding width, and configured dimension                           |
| Upload is rejected                   | PDF/TXT extension and`MAX_UPLOAD_MB`                                            |
| Response stops at the length limit   | Narrow the question or use General answer with a suitable configured token limit  |
| CI dependency consistency fails      | Match requirements and project dependency declarations                            |
| Container job fails                  | Startup logs, dependency compatibility, health status, and frontend HTTP response |
| Cloud deployment fails               | Deployment secrets, token validity, application directory, and cloud build logs   |

This guide contains configuration names and generic examples only. Supply credentials privately in your own environment.

## Acknowledgments

This project uses the following API services and acknowledges the teams that provide them:

| Provider                                                             | Contribution to this project                                                                                                                                        |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Pinecone](https://www.pinecone.io/)                                  | Pinecone Assistant for managed document uploads, retrieval, and chat; Pinecone indexes for NVIDIA retrieval, integrated embeddings where configured, and reranking. |
| [NVIDIA](https://build.nvidia.com/)                                   | Hosted model inference for streamed answers in the NVIDIA provider workflow.                                                                                        |
| [Hugging Face](https://huggingface.co/docs/inference-providers/index) | Hosted feature-extraction inference through`huggingface_hub.InferenceClient` for the dense-vector embedding path.                                                 |

### SDKs, frameworks, and delivery tools

Thanks also to the maintainers of FastAPI, Gradio, Pydantic, LangChain Core and Text Splitters, NumPy, pypdf, Uvicorn, pytest, and the other dependencies that support this application.

The [OpenAI Python SDK](https://github.com/openai/openai-python) provides the client used to call the configured NVIDIA OpenAI-compatible endpoint. SDK usage here does not mean the NVIDIA workflow sends its requests to OpenAI's hosted API.

The delivery workflow uses GitHub Actions for automation, Docker and Docker Hub for container builds and image publication, and FastAPI Cloud for application hosting.

Model selection is configurable. Credit for individual model weights belongs to their respective authors and publishers; consult the selected model's documentation for attribution and licensing details. These acknowledgments describe technology usage and do not imply sponsorship or endorsement.

## Optional document-first web fallback

Both providers first answer from their own document library. **Allow web search** is off by default. With it disabled, insufficient document evidence produces an explicit no-knowledge message. With it enabled, insufficient evidence triggers one Tavily search, followed by an answer from the selected provider with linked page titles. Empty or insufficient web evidence produces a no-answer message.

The ordinary document-answer call is also the evidence decision: the prompt requests a private control marker only if evidence cannot answer the question. A bounded streaming gate suppresses that marker even across token boundaries; normal answers stream immediately once the marker is ruled out. There is no separate classifier call. A supported document answer makes one generation call and zero search calls; fallback with nonempty documents can make two generation calls. Empty NVIDIA retrieval skips the first generation. Timeouts, authentication failures, empty model streams, and malformed partial markers are errors, not fallback decisions.

This is model-based evidence assessment, not proof of answerability. A model can miss evidence or fail to follow the marker instruction; a response without the marker or a recognized explicit refusal is treated as its document answer. Pinecone's managed instructions may also affect this behavior. Live provider validation has not been performed.

Optional backend settings (use private environment values):

```dotenv
TAVILY_API_KEY=your-tavily-api-key
WEB_SEARCH_MAX_RESULTS=3
WEB_SEARCH_TIMEOUT_SECONDS=8
MODEL_MAX_RETRIES=0
NVIDIA_WEB_DOCUMENT_CHARS=6000
NVIDIA_WEB_EXCERPT_CHARS=1000
NVIDIA_WEB_MAX_TOKENS=512
```

API: `POST /api/chat/stream` with `{"query":"your question","provider":"pinecone","web_search":true}`. Use `nvidia` for NVIDIA generation. Tavily receives only the original question on fallback, not uploaded document contents. A question itself may contain private information. The LangChain Core tool performs one basic search with bounded excerpts and network timeout; no MCP server or background indexing is required. Document retrieval and web search are intentionally sequential to avoid unnecessary external requests.

NVIDIA uses its configured web-mode evidence/output budgets when fallback is enabled. Its SDK retries default to zero for interactive chat. Setting `MODEL_MAX_RETRIES` to 1 or 2 explicitly opts in to additional attempts. Timeouts remain network-operation limits, not end-to-end deadlines. Increasing them allows longer waits rather than faster inference. On a timeout during web-answer generation, labelled search excerpts remain available. A document-answer timeout does not trigger search.

Assistant requires an available uploaded file to generate. If it reports no files, web search can return labelled excerpts and links, but the app does not silently switch models. Search results are never uploaded to either library. Named links are drawn from the search response, with hostname fallback; they are not model-invented URLs.

`answer_evidence.py` owns the shared streaming decision protocol. Provider services orchestrate document-first execution; `web_search.py` implements the LangChain search adapter and source formatting. Tests in `test_answer_evidence.py` cover both providers, marker fragmentation, opt-out, unsupported web evidence, and failures. `test_web_search.py` covers the tool and integration behavior.

Credits: [Tavily search](https://docs.tavily.com/documentation/api-reference/endpoint/search), [LangChain StructuredTool](https://reference.langchain.com/python/langchain-core/tools/structured/StructuredTool), and [Pinecone Assistant](https://sdk.pinecone.io/python/how-to/assistant.html).

### Automatic connection and library refresh

The frontend has no **Refresh connection & files** button. It checks the selected provider and reloads its library when the page opens, when the answer provider changes, and after an upload. Provider selection and upload controls are temporarily disabled while these updates complete. Switching to NVIDIA preserves its session upload list; switching to Assistant loads its remote file list. Assistant processing is asynchronous: if a file is still Processing, select another provider and switch back later to reload its status. No periodic polling is performed.

### Greetings and natural-language refusals

Exact short greetings and acknowledgments (such as `hello`, `nice`, `good`, `wellcome`, or `thank you`) receive a local reply in the API and UI, even with search enabled. They do not initialize a provider or call retrieval/search. Messages containing a real question continue through the normal pipeline.

The evidence gate also recognizes a small list of explicit leading refusals, including ?You did not provide enough information to answer this question?, across streaming chunks. These trigger the same conditional web fallback as the control marker. This is conservative phrase matching, not a universal language classifier; unfamiliar refusals may still require additional handling. Transport errors remain errors rather than search decisions.
