# Architecture and SOLID principles

SOLID describes design principles rather than a development process. This app
uses small Python protocols and explicit dependency injection, avoiding an
inheritance hierarchy for providers with different capabilities.

## Responsibilities and dependencies

```text
main.py -> application.create_app()
                |-> FastAPI routes
                |-> Gradio layout -> UI handlers
                                      |
                       provider_service / ingestion_service
                                      |
                    provider registry + composition factories
                       |                          |
                Assistant adapter           RAG workflow
                                              |       |
                                          Retriever  TextGenerator

Upload validation -> selected upload strategy -> Assistant file library
                                             -> IndexingService
                                                loader / chunker / writer / embedder
```

| Principle             | Implementation                                                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Single responsibility | HTTP routes handle transport; UI handlers handle session presentation; indexing orchestrates parsing and writing; prompts and index schema validation have separate modules.                |
| Open/closed           | The provider registry supplies factories, upload strategies, configuration and readiness callbacks. The generic dispatch and validation workflows do not branch on provider names.          |
| Liskov substitution   | RAG consumes any compatible retriever and text generator; indexing consumes compatible loader, chunker, writer and embedder implementations. Contract tests use substitutes without SDKs.   |
| Interface segregation | `contracts.py` separates chat, generation, retrieval, document parsing, chunking, embedding, vector writing and file upload. Adapters are not required to implement unrelated operations. |
| Dependency inversion  | RAG and indexing receive capabilities in constructors. Concrete SDK adapters are assembled at the composition boundary; adapters also accept injected clients/settings for tests.           |

## Where to make changes

- **Provider wiring:** `backend/app/services/provider_service.py` is the
  composition root. It owns lazy construction, cached service instances and
  built-in registrations. `provider_registry.py` holds metadata and lookup only.
- **NVIDIA generation:** `llm_service.py` consumes `NvidiaSettings`, independent
  of retrieval credentials. `RAGService(retriever, generator)` does not build SDKs.
- **Ingestion:** `process_document` validates once and dispatches through the
  registry. `IndexingService` receives its loader, splitter, writer and lazy
  embedder. The embedder is never constructed for integrated text indexes.
- **Prompt policy:** `services/prompts.py`.
- **Index validation:** `pipeline/index_schema.py`; existing import locations
  remain compatible for callers.
- **Application assembly:** `application.create_app(include_ui=False)` creates
  an API-only instance for in-process testing. `main.py` remains the ASGI entry point.
- **UI:** layout, styling and handlers remain separate; no SDK construction in views.

## Adding a provider

1. Implement the chat contract and a document-upload strategy, as needed.
2. Supply factories and safe configuration/readiness callbacks in a
   `ProviderDefinition`; register it in the composition root.
3. Explicitly extend the public `Provider` request literal and UI choices.
   Public API/UI exposure is intentional, not automatic plugin discovery.
4. Add contract and routing tests. Do not add provider branches to RAG or indexing.

## Compatibility and scope

Pinecone Assistant stays the default. Uploads still target only the selected
provider. NVIDIA keeps its configured `rag` retrieval path, and Assistant keeps
its own file library. Existing API response shapes, streaming behavior, UI,
limits and credential names are preserved. Compatibility entry points remain
thin facades. Expected service errors still use the existing HTTPException
contract for API/UI compatibility; this is not a framework-free domain rewrite.

No architecture pattern alone certifies production readiness. The deployment
requirements in README (authenticated access, HTTPS and appropriate session
routing) still apply. These changes do not migrate data or start a local server.
