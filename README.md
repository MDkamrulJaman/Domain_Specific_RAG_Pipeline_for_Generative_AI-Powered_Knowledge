# Technical Document Knowledge Assistant

A retrieval-augmented generation (RAG) application for asking questions about technical documents. FastAPI provides the API, Gradio provides the browser workspace, and hosted AI services provide retrieval and generation.

**Pinecone Assistant is the only active provider.** It manages document storage, retrieval, and answer generation. The interface starts in dark mode and offers task policies and optional document-first web search.

## Contents

- [Capabilities](#capabilities)
- [Provider workflows](#provider-workflows)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [Local setup](#local-setup)
- [Configuration](#configuration)
- [Using the workspace](#using-the-workspace)
- [Pinecone answering and task policies](#pinecone-answering-and-task-policies)
- [API reference](#api-reference)
- [Testing](#testing)
- [Docker and CI/CD](#docker-and-cicd)
- [Operational considerations](#operational-considerations)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)
- [Optional document-first web fallback](#optional-document-first-web-fallback)
- [Sidebar and saved chat sessions](#sidebar-and-saved-chat-sessions)
- [Future improvements](#future-improvements)

## Capabilities

- PDF and TXT uploads with extension, nonempty-content, and configurable size validation.
- Pinecone Assistant file uploads, remote file status, managed retrieval, and streamed answers.
- General, Summarize, Explain, and Find requirements task policies.
- Optional Tavily search through LangChain Core when document evidence cannot answer.
- Named web-source hyperlinks and explicit messages when evidence is insufficient.
- Local replies to short greetings and acknowledgments without provider or search calls.
- Responsive Gradio sidebar with dark/light themes, saved chats, and clickable question suggestions.
- First-token and elapsed-time feedback, lazy service construction, and cached Assistant clients.
- Shared API/UI workflows, small service interfaces, and injected SDK/search dependencies.
- Feature-organized offline tests and automated container checks in GitHub Actions.

The application calls hosted services. It does not download model weights or run local embedding/inference models. Managed retrieval means the application does not control Assistant's internal chunking or passage selection.

## Provider workflows

### Pinecone Assistant

| Stage                    | Responsibility                                                       |
| ------------------------ | -------------------------------------------------------------------- |
| Upload validation        | Application checks file extension, size, and nonempty content        |
| Document destination     | Configured Assistant file library                                    |
| Document processing      | Managed by Pinecone Assistant                                        |
| Retrieval and generation | Assistant chat API                                                   |
| Optional public evidence | Tavily search, only after an insufficient-document-evidence decision |
| Library display          | Remote Assistant filenames and processing status                     |

```text
PDF/TXT -> validate upload -> Assistant file library -> processing -> available
Question -> local greeting check -> Assistant retrieval and generation
                                    | supported -> stream answer
                                    | unsupported, search off -> no-knowledge message
                                    ` unsupported, search on -> Tavily -> Assistant web answer + links
```

Wait for uploaded files to become available before asking questions. Connection and file status refresh on page load and after an upload. Reload the page to check later processing changes; no manual refresh button or periodic polling is implemented.

All active uploads and chat requests use the Assistant library. Separate vector indexes are not searched, synchronized, created, migrated, or deleted by this application. Existing documents stored elsewhere must be uploaded to Assistant to become available here. Future provider expansion is described under [Future improvements](#future-improvements).

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
|   |   |-- pipeline/            Inactive package/scratch remnants; no runtime pipeline
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
|-- image/                      Existing illustrations, including historical diagrams
|-- LICENSE                     Project license
|-- pytest.ini                  Test discovery and backend import path
|-- ARCHITECTURE.md              Project design, SOLID, all 23 GoF patterns, extension guidance
|-- .gitignore                  Repository-wide exclusions
`-- README.md                   Project guide
```

Python `__init__.py` files establish packages or expose public imports. Local `.venv`, `.env`, `.fastapicloud`, cache directories, and editor-generated `tempCodeRunnerFile.py` files are development artifacts, not application modules to deploy. Do not commit credentials or local environment state.

### API and schemas

| Module                           | Responsibility                                                                   |
| -------------------------------- | -------------------------------------------------------------------------------- |
| `api/app_router/app_router.py` | Composes routers; preserves legacy router aliases                                |
| `api/routes/chat.py`           | Validates chat requests and returns streamed text                                |
| `api/routes/ingest.py`         | Accepts multipart uploads, bounds reads, and closes uploaded files               |
| `api/routes/providers.py`      | Reports configuration or explicitly checks remote connectivity                   |
| `schemas/chat.py`              | Validates question, Pinecone provider, search option, and task policy            |
| `schemas/ingest.py`            | Legacy ingestion model; current upload route returns service result dictionaries |

### Services

| Module                   | Responsibility                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `contracts.py`         | Small protocols for streamed chat, file upload, upload dispatch, and search            |
| `provider_registry.py` | Provider capability definitions and lookup; only Pinecone is registered                |
| `provider_service.py`  | Wires adapters, caches the Assistant service, and summarizes configuration/readiness   |
| `assistant_service.py` | Managed chat, uploads, file listing, status, and document-first fallback orchestration |
| `chat_command.py`      | Immutable request Command shared by API and UI; handles exact social messages locally  |
| `ingestion_service.py` | Shared upload validation and dispatch to Assistant                                     |
| `skills.py`            | Immutable task definitions and per-request grounding/formatting instructions           |
| `answer_evidence.py`   | Bounded streaming gate for insufficient-evidence markers and recognized refusals       |
| `web_search.py`        | LangChain Core Tavily adapter, bounded source excerpts, and safe named hyperlinks      |

### Document pipeline and UI

Document processing is managed remotely by Assistant. The earlier application-side loader, splitter, embedder, index schema checks, retrieval, and reranking modules are no longer active or included as implemented features. Remaining `pipeline/` package/scratch files do not participate in requests.

| Module             | Responsibility                                                                  |
| ------------------ | ------------------------------------------------------------------------------- |
| `ui/frontend.py` | Builds sidebar/chat components, native saved-history controls, and event wiring |
| `ui/handlers.py` | Streams answers, handles uploads, and copies/refreshes library state            |
| `ui/styles.py`   | Theme configuration, initial dark mode, and presentation assets                 |
| `ui/theme.css`   | Responsive workspace styling                                                    |

The `image/` folder contains project illustrations. Historical images may describe earlier designs; the current workflow is documented here and in `ARCHITECTURE.md`.

## Architecture

This project helps users find and understand facts in technical manuals, specifications, and requirements documents. Retrieval grounds answers in an uploaded knowledge library; optional web fallback covers questions that the library cannot support. It is an application around managed retrieval and generation, not a locally trained model or a custom vector database implementation.

The project applies SOLID principles through focused modules, small protocols, an extensible registry, and dependency injection. API and UI boundaries both create an `AnswerCommand`. The composition root supplies the Assistant and search adapters. Evidence handling, task instructions, upload validation, and presentation have separate responsibilities.

`create_app(include_ui=False)` supports API-only tests; the normal ASGI entry point mounts the browser workspace. Clients are constructed lazily and reused per process. Browser chat history and the remote file library have different lifetimes and do not provide user-level data isolation.

See [Architecture and SOLID principles](ARCHITECTURE.md) for the execution diagram, module boundaries, limitations, and extension guidance. The [GoF catalog](ARCHITECTURE.md#design-pattern-policy) covers all 23 patterns and identifies actual usage, framework-provided behavior, and patterns reserved for future needs.

## Local setup

Deployment targets **Python 3.11**. Use that version locally to match CI and the container. Git is required for the release workflow. Docker Desktop is optional: GitHub-hosted runners build and test containers online.

From the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt pytest
```

Create an untracked `backend/.env` and configure Pinecone Assistant and, optionally, Tavily. Environment-variable names are documented below; no actual credentials are included in this repository guide.

To start the application yourself:

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the workspace or `/docs` for interactive API documentation. Add `--reload` only during development.

## Configuration

Settings load `backend/.env` independently of the current working directory. Process environment variables take precedence. Restart after changing Assistant settings because service instances are cached. Do not commit real credentials or place them in frontend components.

| Required setting                       | Purpose                                                    |
| -------------------------------------- | ---------------------------------------------------------- |
| `PINECONE_ASSISTANT_API_KEY`         | Credential authorized for the configured Assistant         |
| `PINECONE_ASSISTANT_NAME`            | Name of the existing Assistant file library                |
| `PINECONE_ASSISTANT_MODEL`           | Generation model supported by your Assistant configuration |
| `PINECONE_ASSISTANT_TIMEOUT_SECONDS` | Timeout passed to Assistant chat requests                  |

| Optional setting               | Default | Meaning                                                        |
| ------------------------------ | ------- | -------------------------------------------------------------- |
| `TAVILY_API_KEY`             | Empty   | Required only when web fallback actually searches              |
| `WEB_SEARCH_MAX_RESULTS`     | `3`   | Search result limit, from 1 to 5                               |
| `WEB_SEARCH_TIMEOUT_SECONDS` | `8`   | Search request timeout, from 1 to 30 seconds                   |
| `MAX_UPLOAD_MB`              | `5`   | Upload limit, from 1 to 100; converted using 1024 x 1024 bytes |
| `UI_QUEUE_SIZE`              | `8`   | Gradio queue capacity, from 1 to 1000                          |
| `UI_CONCURRENCY`             | `1`   | Gradio event concurrency, from 1 to 32                         |

These are conservative defaults for a small hosting instance, not a guarantee of memory usage. UI concurrency does not limit direct API calls or guarantee a single in-flight operation across every event type. Hosted-model latency and provider quotas remain external constraints.

The active app does not require separate vector-index credentials or embedding settings. Extra environment fields are ignored. Configuration summaries expose field names and model information, not credential values.

Keep `backend/requirements.txt` and `backend/pyproject.toml` dependencies synchronized; CI checks their equality. `backend/app/requirements.txt` forwards to backend requirements. Use the `pinecone` package; the application does not require the former local document-processing dependencies.

## Using the workspace

1. Open the workspace; it checks Pinecone Assistant readiness and loads the remote file list.
2. Upload a PDF or TXT file using the sidebar.
3. Wait for Available status. Reload the page if processing is still in progress.
4. Select a Pinecone task: General, Summarize, Explain, or Find requirements.
5. Enable **Allow web search** only when public-source fallback is wanted.
6. Enter a question, or click a suggested question to send it using the current controls.
7. Use **New chat** and saved conversation titles to organize browser-local transcripts.

Each question is independent. Visible or restored chat history is not sent as conversational context. All users of the same configured backend access the same Assistant library; a new chat does not create a private document collection.

## Pinecone answering and task policies

Assistant manages retrieval internally. The app sends the question, a request-scoped task policy, and evidence instructions; it does not set a custom `top_k`, reranker, local embedding model, or thinking switch.

| Skill value      | Sidebar label     | Requested behavior                                                                   |
| ---------------- | ----------------- | ------------------------------------------------------------------------------------ |
| `general`      | General           | Answer directly; expand when the question requires detail                            |
| `summarize`    | Summarize         | Up to five bullets, preserving caveats and acknowledging limited evidence coverage   |
| `explain`      | Explain           | Clear explanation and essential details; examples only when supported                |
| `requirements` | Find requirements | Preserve exact wording, identifiers, conditions, and mandatory/optional distinctions |

All policies request grounded claims and available source citations, prohibit invented facts/URLs, and treat retrieved content as reference data rather than instructions. Insufficient-evidence handling takes precedence over formatting. Summaries cover retrieved evidence, not necessarily every page of every uploaded document.

The same policy applies to a web fallback answer. Only the original question is sent to Tavily. Policies do not mutate the remote Assistant's global configuration, add a classification request, or guarantee model compliance. The default is `general`, and policy state does not leak between requests.

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

Example Pinecone chat body:

```json
{
  "query": "What startup requirements are stated in the documents?",
  "provider": "pinecone",
  "skill": "requirements",
  "web_search": false
}
```

`provider` defaults to `pinecone`, `skill` to `general`, and `web_search` to `false`. Only Pinecone is accepted; unsupported providers or skills return HTTP 422. Questions must be nonblank and at most 12,000 characters. The optional `chat_id` is accepted but does not enable server-side conversational memory. Unknown JSON fields are ignored; old retrieval/thinking fields have no effect.

Uploads use multipart `file`, with optional `provider=pinecone`, and return HTTP 201 after the service accepts the upload. File processing may still be ongoing. Invalid file content/extension, oversized uploads, and invalid provider selection produce validation errors. Streaming failures after headers are sent appear in the response body rather than changing the HTTP status.

## Testing

From the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest -q
.\backend\.venv\Scripts\python.exe -m pytest tests/test_assistant_skills.py -v
.\backend\.venv\Scripts\python.exe -m pip check
```

`pytest.ini` selects `tests/` and adds `backend/` to the import path. Tests cover routing, upload limits and cleanup, Pinecone-only routing, task policies, web fallback, schema validation, streaming, UI wiring, and dependency-injection contracts. External services are mocked, and API calls use an in-process test client.

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

After reviewing changes, push to `main` or merge a checked pull request. Include `[deploy]` in the final commit message only when a release is intended. Monitor GitHub Actions; after a release, verify the live health endpoint, uploads, and Pinecone chat.

## Operational considerations

- Gradio queue/session state and cached clients are process-local. Verify routing and shared-state requirements before deploying multiple workers or replicas.
- Gradio temporary uploads are eligible for cleanup after 24 hours, checked hourly. API uploads read at most the configured byte limit plus one byte and close incoming files in a `finally` block.
- Assistant handles document parsing and retrieval remotely. This app does not implement local OCR, local PDF parsing, or independent embedding/index management.
- Assistant uploads use the SDK's asynchronous-processing mode (`timeout=-1`); the chat timeout setting is not an overall upload/processing deadline. Check remote availability after upload.
- A supported document answer uses one generation call and no search. Web fallback can require one search and a second generation call, increasing total latency and external usage.
- No cross-request answer or retrieval-result cache is implemented. Larger timeouts allow longer waits; they do not speed up inference. Scale-to-zero can add cold-start latency.
- Authentication, per-user library isolation, and distributed rate limiting are not implemented. Protect public access before handling private documents.
- Browser-local transcripts can persist on shared computers. Questions sent to public search may contain private information even though uploaded file contents are not forwarded to Tavily.
- Dependencies have compatibility ranges but are not fully locked. Docker and cloud builds are separate, so exact dependency reproducibility is not guaranteed.

## Troubleshooting

| Symptom                                 | Check                                                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Assistant shows setup required          | Required environment-variable names and values; restart after changes                                   |
| Connection fails                        | Assistant access, name, credentials, remote service availability, and network access                    |
| Files remain Processing                 | Reload the page later; processing is remote and no periodic polling runs                                |
| Document not found in answers           | Confirm it is uploaded to the configured Assistant and available; another index/library is not searched |
| Upload is rejected                      | PDF/TXT extension, nonempty bytes, and`MAX_UPLOAD_MB`                                                 |
| Answer says knowledge is unavailable    | Ask a more specific question, add relevant documents, or permit web fallback                            |
| Search reports missing configuration    | Set`TAVILY_API_KEY` privately and restart                                                             |
| Only search excerpts appear             | Assistant reported no files; upload an available file for generated answers                             |
| An answer is slow                       | Compare first-token/total time and remote service status; fallback makes additional network calls       |
| Saved chat is missing on another device | History is browser-local, not synchronized to the backend                                               |
| CI dependency consistency fails         | Match requirements and project dependency declarations                                                  |
| Container job fails                     | Startup logs, dependency compatibility, health status, and frontend HTTP response                       |
| Cloud deployment fails                  | Deployment secrets, token validity, application directory, and cloud build logs                         |

This guide contains configuration names and generic examples only. Supply credentials privately in your own environment.

## Acknowledgments

This project acknowledges the API companies and maintainers whose services support its current implementation:

| Provider                            | Contribution                                                     |
| ----------------------------------- | ---------------------------------------------------------------- |
| [Pinecone](https://www.pinecone.io/) | Managed Assistant file library, retrieval, and answer generation |
| [Tavily](https://tavily.com/)        | Optional public web search when uploaded evidence cannot answer  |

### SDKs, frameworks, and delivery tools

Thanks to the maintainers of FastAPI, Gradio, Pydantic, LangChain Core, HTTPX, Uvicorn, pytest, and the project's other dependencies. LangChain Core provides the structured search-tool integration; it is not an MCP server or an additional model-generation layer here.

GitHub Actions provides automation, Docker and Docker Hub provide image build/publication, and FastAPI Cloud provides the configured deployment destination. Credit for individual models belongs to their publishers. These acknowledgments do not imply sponsorship or endorsement.

## Optional document-first web fallback

**Allow web search** is off by default. Pinecone first attempts an answer from its document library. If evidence is insufficient, search stays off unless the current request explicitly enables it. Supported document answers never trigger search.

The ordinary document-answer request also supplies the evidence decision: instructions request a private control marker only when evidence cannot answer. `answer_evidence.py` buffers a possible opening marker or recognized refusal across token boundaries, up to its decision threshold, and streams ordinary text once that ambiguity is resolved. There is no separate classifier call.

On insufficient document evidence, one Tavily search returns bounded excerpts. Assistant then receives those excerpts with the original question and the same task policy. Source links use page titles from search metadata, with a hostname fallback. Unsafe schemes and duplicate results are filtered; source text is treated as untrusted evidence. Search results are not uploaded to the library.

Empty search results or an unsupported web answer produce an explicit no-answer message. If Assistant reports no files, the application can display labelled public excerpts and links instead of claiming a generated answer. Timeouts, authentication failures, empty model streams, and malformed partial markers are errors, not fallback decisions.

This is model-based evidence assessment, not proof of answerability. A model may miss evidence or fail to follow instructions. An answer without the marker or a recognized refusal is treated as supported by the document path. Prompt policies and phrase checks cannot guarantee factual correctness.

Use `POST /chat/stream` with `{"query":"your question","provider":"pinecone","web_search":true}`. Tavily receives only the original question; the question itself may contain private information. Search uses LangChain Core with one basic request, bounded results, and a configured network timeout. There is no MCP server or background indexing process.

### Automatic connection and library refresh

The frontend checks the connection and reloads the file list on page load and after upload. Upload controls are temporarily disabled while these updates complete. For a file still Processing, reload the page later to request its current status. No provider-switching control, manual refresh button, or periodic polling is needed for the current single-provider layout.

### Greetings and natural-language refusals

Exact short messages such as `hello`, `nice`, `good`, `wellcome`, and `thank you` receive local API/UI replies without initializing a provider or searching. Messages that contain a real question follow the normal workflow.

The evidence gate recognizes specific leading refusal phrases, including "You did not provide enough information to answer this question", across streamed chunks. Matching is conservative rather than a universal language classifier. Transport errors remain errors instead of being interpreted as missing knowledge.

## Sidebar and saved chat sessions

The collapsible sidebar contains **New chat**, saved conversation titles, dark mode, Assistant connection/model status, the Pinecone task selector, optional web fallback, document upload, and filenames/status. Pinecone is the sole active provider; no model-provider selector or thinking control is shown.

Saved conversations use Gradio's built-in browser-local history. New chat clears the visible conversation; selecting an earlier title restores its transcript. History is not synchronized between devices and may remain on a shared computer. Delete conversations with the chat controls or clear site data to remove stored history.

Sessions restore messages, not task settings or a private document collection. Current sidebar settings control the next question. Restored history is displayed but not sent as LLM context. Document names and statuses come from the remote Assistant library shared by the configured backend.

## Future improvements

These are possible extensions, not capabilities enabled by this release.

### Optional NVIDIA provider

NVIDIA inference could be reintroduced as a separately configured provider after validating latency, streaming behavior, retry budgets, and operational cost. The former LLM helper has been removed; reintroduction would require a new, explicitly integrated generation adapter.

A future implementation would need explicit settings, a supported retrieval source, an upload/storage contract, registry/schema/UI integration, and tests. Pinecone Assistant files must not be assumed to be directly interchangeable with arbitrary vector-index records. Any shared-storage proposal needs verification before claiming both providers use the same database workflow.

A hosted embedding adapter, including Hugging Face, could be evaluated if a future custom retrieval path requires it. NVIDIA and Hugging Face powered earlier project experiments; neither is an active runtime integration in this release. See [future extension rules](ARCHITECTURE.md#future-extension-rules) for preserving SOLID boundaries.

### Other improvements

- Authentication and per-user document access controls before supporting private multi-user deployments.
- Server-side conversation storage and opt-in conversational context with clear retention controls.
- Bounded processing-status polling or notifications if asynchronous uploads require better feedback.
- Structured provider metrics and representative latency/load evaluation.
- Reproducible dependency locking and deployment promotion checks.
- Evidence-quality evaluation, citation verification, and better handling of unrecognized refusals.
