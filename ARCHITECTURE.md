# Architecture

## Runtime flow

FastAPI routes and Gradio handlers validate a ChatRequest and create an immutable AnswerCommand. Exact greetings receive local replies. Other requests resolve the cached Pinecone Assistant adapter. No other answer provider is registered.

AssistantService first streams an answer from its managed file library. The evidence gate buffers only enough text to distinguish an unsupported-answer marker or recognized refusal. If documents cannot answer and the request permits web search, the injected LangChain Core Tavily adapter searches the original question. Assistant then answers from the bounded web excerpts using the same task policy. Named links come from search metadata. Provider errors do not trigger fallback.

Uploads pass shared extension, size and nonempty-content checks, then go directly to Assistant. Pinecone owns retrieval and document processing. There is no active local chunking, embedding, reranking or vector-index pipeline. The retained llm_service.py is an inactive reference requiring explicit settings; application composition never imports it.

The UI owns presentation, browser-local saved transcripts, upload receipts and status. It starts in dark mode. Settings and prior conversation messages are not restored as model context. Library status refreshes on page load and upload; no periodic polling occurs. Libraries are shared at the configured Assistant level.

## SOLID boundaries

| Principle | Application |
| --- | --- |
| Single responsibility | Schemas validate; routes format HTTP; commands execute; adapters call SDKs; handlers present state |
| Open/closed | Registry and injected adapters support extension without adding SDK logic to HTTP handlers |
| Liskov substitution | Tests substitute implementations of the small streaming/search/upload interfaces |
| Interface segregation | ChatService, SearchTool and upload contracts expose only needed capabilities |
| Dependency inversion | Workflows consume injected services; provider_service wires concrete implementations |

skills.py contains immutable request policies, not SDK clients. answer_evidence.py owns evidence classification during streaming; web_search.py owns search transport and link formatting. Errors expose sanitized messages. Queue/upload limits are independent of provider credentials. Client caching is process-local, not distributed state.

## Gang of Four pattern applicability

Patterns are used where their behavior is present; adding all 23 would create unnecessary machinery.

| Category | Pattern | Current applicability |
| --- | --- | --- |
| Creational | Abstract Factory | Not implemented as a family of related product factories |
| Creational | Builder | Not needed; create_app/create_demo are construction functions |
| Creational | Factory Method | Callable factories are used, without a subclass-based GoF Factory Method |
| Creational | Prototype | Not used; copying UI receipts is not prototype-based object construction |
| Creational | Singleton | Not enforced; lru_cache reuses a client per process |
| Structural | Adapter | AssistantService and TavilySearchAdapter translate external APIs into application operations |
| Structural | Bridge | Not needed; no separate abstraction/implementation hierarchies |
| Structural | Composite | No domain-level Composite; Gradio owns its component tree |
| Structural | Decorator | Python route/cache decorators are used; no GoF object-wrapper hierarchy |
| Structural | Facade | inspect_provider summarizes configuration, readiness and files |
| Structural | Flyweight | Not needed; immutable policy constants are simply shared values |
| Structural | Proxy | No explicit proxy object; factories initialize services lazily |
| Behavioral | Chain of Responsibility | Not implemented; document-first fallback is an explicit conditional workflow |
| Behavioral | Command | AnswerCommand snapshots a validated request for API/UI execution |
| Behavioral | Interpreter | Not needed; skill names are validated enum values, not a language grammar |
| Behavioral | Iterator | Generator streams deliver answer fragments lazily |
| Behavioral | Mediator | Gradio event wiring coordinates views; no separate domain mediator |
| Behavioral | Memento | Browser history stores transcripts; no application undo/snapshot protocol |
| Behavioral | Observer | Gradio supplies event dispatch; no custom observer framework |
| Behavioral | State | Status values are data, not polymorphic state objects |
| Behavioral | Strategy | Upload capability and request policies are selected as data/callables; only Pinecone is registered |
| Behavioral | Template Method | Not needed; orchestration uses composition rather than subclass hooks |
| Behavioral | Visitor | Not used; there is no heterogeneous domain tree requiring visiting |

## Verification and delivery

Tests use fake services and in-process HTTP clients. They cover streamed errors, document-first fallback, fragmented evidence markers, request-policy isolation, upload cleanup, UI wiring, and rejection of removed provider values. A regression check prevents active source imports of the retained helper. CI uses Python 3.11, verifies dependency-manifest consistency, then builds and smoke-tests Docker. Only pushes to main whose head commit includes [deploy] publish and deploy. FastAPI Cloud builds source separately from the Docker Hub image.

Offline tests do not establish live provider latency, prompt compliance or browser rendering. Authentication, user-isolated file libraries and distributed session/queue storage are not implemented.
