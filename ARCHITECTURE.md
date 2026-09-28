# Architecture, SOLID, and Gang of Four patterns

SOLID describes design principles rather than a development process. This app
uses small Python protocols and explicit dependency injection, avoiding an
inheritance hierarchy for providers with different capabilities.

## Responsibilities and dependencies

```text
main.py -> application.create_app()
                |-> FastAPI routes
                |-> Gradio layout -> UI handlers
                                      |
                       AnswerCommand / ingestion_service
                                      |
                               provider_service
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
- **Chat commands:** `services/chat_command.py` snapshots validated request values
  and executes against an injected chat receiver. HTTP and Gradio invoke the same
  command; schemas validate input and transports format errors.
- **Prompt policy:** `services/prompts.py`.
- **NVIDIA skills:** immutable task policies in `services/skills.py`; the API
  accepts `skill`: `general`, `summarize`, `explain`, or `requirements`.
  NVIDIA's frontend options expose the same policies. Pinecone ignores this
  NVIDIA-only setting. Each skill limits context and output per request without
  changing cached service settings. The configured model token limit remains
  an upper bound. General answer keeps the existing generation budget.
  Summaries cover retrieved passages only, not complete documents. Passage
  labels refer to supplied context, not verified document/page citations.
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


## Design-pattern policy

SOLID remains the architectural foundation. Design patterns name recurring
solutions inside that architecture; they are not a requirement to add 23 new
classes. The catalog below covers all 23 GoF patterns and distinguishes an actual
implementation from a related idiom or a future extension. A function named
`build`, a Python decorator, or a cached object does not by itself establish the
corresponding GoF pattern.

**Status legend**

- **Used:** the application's current structure implements the pattern's intent.
- **Related idiom:** some ideas are present, but the full GoF structure is absent.
- **Not used:** there is no current requirement that justifies the extra machinery.

### Creational patterns (5)

| Pattern | Plain-language meaning | Status and project decision |
| --- | --- | --- |
| Abstract Factory | Create compatible families of related objects through a common factory interface. | **Related idiom.** `ProviderDefinition` bundles compatible provider capabilities and a chat factory. It does not define an abstract factory that creates a family of products. Use one if each provider needs multiple independently constructed, coordinated adapters. |
| Builder | Assemble a complex object step by step, separating construction from its representation. | **Not used as a GoF pattern.** `create_app`, `create_demo`, and `build_rag_prompt` are ordinary assembly functions. Add a builder only if several complex assembly sequences must produce different representations. |
| Factory Method | Let subclasses decide which concrete product a creation method returns. | **Related idiom.** `_nvidia_chat`, `_assistant_chat`, and `create_app` are simple factories using functions/composition, not overridden creation methods. This keeps provider construction explicit without a subclass hierarchy. |
| Prototype | Create objects by cloning configured prototypes. | **Not used.** Copying library receipts protects session data; it is not prototype-based object creation. Cloning SDK clients could also duplicate unsafe connection state. |
| Singleton | Enforce one instance and provide a global access point. | **Related idiom.** `lru_cache` reuses provider services within a process but does not enforce one construction during concurrent cache misses or across workers/replicas. Tests can clear caches and inject instances. Do not rely on it for distributed coordination. |

### Structural patterns (7)

| Pattern | Plain-language meaning | Status and project decision |
| --- | --- | --- |
| Adapter | Translate an external interface into the interface the application needs. | **Used.** `LLMService`, `AssistantService`, `VectorStore`, `DocumentSplitter`, and the loader/embedder adapt SDK or library behavior to small application-facing capabilities. |
| Bridge | Separate an abstraction hierarchy from an implementation hierarchy so each can vary independently. | **Related idiom.** `RAGService` composes a `Retriever` and a `TextGenerator`. That provides independent variation, but there are no two explicit hierarchies forming a classic Bridge. Strategy and dependency injection describe this code more precisely. |
| Composite | Treat individual objects and nested groups uniformly through one interface. | **Not used in application logic.** Gradio supplies nested layouts, but the project has no custom leaf/container processing hierarchy. Consider it only if nested document collections need uniform operations. |
| Decorator | Wrap an object with the same interface to add behavior without changing the wrapped implementation. | **Not used as a GoF object pattern.** Python `@lru_cache` and route decorators are language/framework decorators. A future metered `TextGenerator` wrapper could be a GoF Decorator if it preserves streaming and resource cleanup. |
| Facade | Offer a simpler entry point over a subsystem's several operations. | **Used.** `provider_service` exposes construction, configuration, and readiness; `process_document` exposes validation and provider-specific ingestion. API/UI callers do not orchestrate SDK calls themselves. |
| Flyweight | Share intrinsic object state while passing per-use state separately to reduce memory. | **Related idiom.** Frozen `Skill` policies are shared and request-specific choices remain separate. This is a small immutable registry, not a flyweight factory managing large numbers of fine-grained objects. |
| Proxy | Stand in for another object and control access while preserving its interface. | **Not used.** Lazy factory construction and remote SDK calls do not make the application's wrappers a custom Proxy; those wrappers change interfaces and are Adapters. An access-control proxy would require a real identity/authorization design. |

### Behavioral patterns (11)

| Pattern | Plain-language meaning | Status and project decision |
| --- | --- | --- |
| Chain of Responsibility | Pass a request through linked handlers until one handles it or the chain completes. | **Not used.** Upload guards are sequential validation, not linked handler objects. A chain becomes useful only when independently configurable validation stages are needed. |
| Command | Package an action and its parameters so callers can invoke it independently of the receiver. | **Used.** Frozen `AnswerCommand` captures a validated request and exposes `execute(receiver)`. Both API and UI use it, centralizing NVIDIA skill forwarding. There is no implied undo, durable queue, or automatic retry. |
| Interpreter | Represent a language's grammar and evaluate expressions in that language. | **Not used.** Prompt strings and Pydantic validation are not a query-language interpreter. Introduce it only for a defined filter DSL with explicit syntax and execution rules. |
| Iterator | Access items sequentially without exposing how the collection or stream is implemented. | **Used through Python's iterator protocol.** Generators forward provider fragments through the RAG workflow and transports. Consumers receive text incrementally; `ask`/`generate` intentionally collect the stream for non-streaming callers. |
| Mediator | Centralize interactions among collaborating objects to reduce direct coupling. | **Related idiom.** Gradio event wiring and UI handlers coordinate components, while indexing orchestrates pipeline stages. There is no dedicated GoF mediator object with colleague-to-mediator messaging. |
| Memento | Capture encapsulated object state so it can be restored later. | **Not used.** Receipts are copied before updates, but no restore/undo history exists. Never add rollback semantics that imply an external upload was reversed when it was not. |
| Observer | Notify subscribed dependents when an object's state changes. | **Related idiom.** Indexing emits progress through an injected callback and Gradio dispatches UI events. The application has no subject/subscriber collection or notification bus. A single callback meets the current requirement. |
| State | Delegate behavior to state objects as an object's state changes. | **Not used.** Provider status strings and UI readiness messages are data, not a State object hierarchy. Add a state machine if resumable ingestion gains explicit transition and recovery rules. |
| Strategy | Select interchangeable behavior behind a common contract. | **Used.** The provider registry selects upload callables; RAG accepts interchangeable retrieval/generation capabilities. Skill policies supply task-specific data to a common prompt-building algorithm. |
| Template Method | Define an algorithm in a base class and let subclasses override selected steps. | **Not used.** `IndexingService` defines an ordered workflow but receives collaborators through composition. Constructor injection keeps those stages independently testable without inheritance. |
| Visitor | Add operations across an object structure without changing its element classes. | **Not used.** Documents currently pass through a small linear parsing/indexing workflow. A visitor would be justified by a heterogeneous document tree needing many separate operations. |

## Pattern implementation walkthrough

### One answer through the API or UI

1. `ChatRequest` validates the provider, query, skill, and retrieval count.
2. The transport asks the provider-service facade for a receiver. Registered
   factory callables construct the adapter lazily and reuse it through a cache.
3. `AnswerCommand.from_request` snapshots scalar request values. Later mutation
   of the Pydantic request cannot change this command.
4. `AnswerCommand.execute(receiver)` passes NVIDIA skills only to the receiver
   that supports them. General chat remains compatible with `ChatService`.
5. NVIDIA's RAG workflow delegates retrieval and generation to injected
   capabilities. Its automatic passage count comes from the chosen skill;
   an explicit count overrides it. Context/output budgets remain bounded.
6. Adapters translate the application's calls to provider SDK calls. Iterators
   carry response fragments back to the API/UI as they arrive.
7. The transport retains responsibility for HTTP or presentation-specific error
   handling. Commands do not swallow provider failures or start new retries.

Pinecone remains the default. Its Assistant workflow handles its own retrieval
and does not use NVIDIA skill policies or the NVIDIA index pipeline.

### One upload

`process_document` is the facade. It resolves an upload Strategy and applies
shared validation before executing it. Assistant delegates to its file adapter.
NVIDIA constructs an `IndexingService` from independent loader, splitter,
embedder factory, and writer capabilities. The embedder remains lazy so integrated
indexes do not make unnecessary Hugging Face calls. Progress is reported through
a callback. This is composition and dependency inversion, not Template Method.

### Folder-level responsibilities

| Folder / file | Design responsibility |
| --- | --- |
| `backend/app/main.py`, `application.py` | Entry point and simple application factory; assemble transports |
| `backend/app/api/` | HTTP invokers and router composition; format HTTP errors |
| `backend/app/schemas/` | Validate request data; keep SDK execution out of schemas |
| `backend/app/core/` | Typed configuration and operational limits; no provider orchestration |
| `backend/app/services/` | Commands, strategies, facades, contracts, policies, and adapter composition |
| `backend/app/pipeline/` | Concrete parsing, chunking, embedding, and index adapters |
| `backend/app/ui/` | Command invoker, event wiring, presentation, and session copies |
| `tests/` | Behavioral contracts and fake receivers/collaborators; no live provider credentials |
| `.github/workflows/` | Delivery orchestration, not application design-pattern implementations |
| `backend/Dockerfile`, dependency files | Runtime packaging and reproducibility concerns, not GoF classes |

## How SOLID stays intact

- **SRP:** schemas validate, commands express a use case, adapters talk to SDKs,
  and transports format responses. Pattern comments explain boundaries instead
  of repeating each line of code.
- **OCP:** register a new provider capability without rewriting generic upload
  validation. New public provider/skill names still require intentional schema
  and UI changes; open/closed does not mean every extension needs zero edits.
- **LSP:** replacements must preserve signatures, iterator behavior, and expected
  failure semantics. NVIDIA's optional skill interface extends basic chat;
  it is not forced onto Assistant.
- **ISP:** generation, retrieval, file upload, and vector writing remain separate
  protocols. No universal provider base class collects unrelated methods.
- **DIP:** workflows and the command receive capabilities/receivers. The
  composition root owns concrete construction and SDK selection.

## Validation and extension rules

`test_chat_command.py` checks the command's request snapshot, receiver delegation,
stream laziness, and error propagation. Existing API, frontend, skills, provider,
and architecture tests protect the cross-layer behavior.

Before introducing any currently unused pattern, identify a concrete requirement,
name the objects and responsibility it separates, and add a behavioral test for
that requirement. Avoid pattern-only class hierarchies, transparent credential
logging, or hidden retries of remote writes. A pattern is useful when it makes an
actual change easier to understand, implement, or verify.
