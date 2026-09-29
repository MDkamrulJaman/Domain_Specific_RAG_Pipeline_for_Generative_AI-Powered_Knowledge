import logging
import time
from app.core.config import NvidiaWebSettings
from fastapi import HTTPException

from app.services.contracts import Retriever, TextGenerator, SearchTool
from app.services.prompts import build_rag_prompt

logger = logging.getLogger(__name__)


# Strategy + DIP: retrieval and generation can vary independently through protocols.
class RAGService:
    def __init__(self, vectorstore: Retriever, llm: TextGenerator,
                 web_search_tool: SearchTool | None = None, web_settings: NvidiaWebSettings | None = None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.web_search_tool = web_search_tool
        self.web_settings = web_settings if web_settings is not None else NvidiaWebSettings()

    def ask(self, query: str, top_k: int | None, web_search: bool = False) -> str:
        return "".join(self.stream(query, top_k, web_search=web_search))

    def stream(self, query: str, top_k: int | None = None, enable_thinking: bool | None = None,
               web_search: bool = False):
        start_time = time.perf_counter()
        effective_top_k = 5 if top_k is None else top_k

        if not query or not query.strip():
            logger.info("RAGService.ask completed in %.3f seconds", time.perf_counter() - start_time)
            yield "Please provide a valid query."
            return
        if effective_top_k < 1 or effective_top_k > 50:
            logger.info("RAGService.ask completed in %.3f seconds", time.perf_counter() - start_time)
            yield "The number of retrieved passages must be between 1 and 50."
            return

        try:
            from app.services.answer_evidence import (
                evidence_instruction, supported_stream, NO_DOCUMENT_ANSWER, NO_WEB_ANSWER,
            )
            from app.services.web_search import web_context, source_links, source_excerpts
            try:
                docs = self.vectorstore.search(query, effective_top_k)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                docs = []
            options = {"enable_thinking": enable_thinking}
            if web_search:
                options["max_tokens"] = self.web_settings.NVIDIA_WEB_MAX_TOKENS
                options["enable_thinking"] = False if enable_thinking is None else enable_thinking
            if docs:
                prompt = build_rag_prompt(query, docs, max_context_chars=(
                    self.web_settings.NVIDIA_WEB_DOCUMENT_CHARS if web_search else None))
                prompt += evidence_instruction("the supplied document passages")
                # The normal answer call doubles as the evidence decision. No classifier call.
                answered = yield from supported_stream(self.llm.stream(prompt, **options))
                if answered:
                    return
            if not web_search:
                yield NO_DOCUMENT_ANSWER
                return
            if self.web_search_tool is None:
                raise HTTPException(503, "Web search is not configured.")
            sources = self.web_search_tool.search(query)
            sources = [dict(doc, text=doc["text"][:self.web_settings.NVIDIA_WEB_EXCERPT_CHARS])
                       for doc in sources]
            if not sources:
                yield NO_WEB_ANSWER
                return
            yield "*Documents do not contain the answer. Web search enabled; waiting for NVIDIA.*\n\n"
            prompt = web_context(query, sources) + evidence_instruction("the supplied web evidence")
            prompt += "\nTask: " + "Answer concisely."
            try:
                answered = yield from supported_stream(self.llm.stream(prompt, **options))
            except HTTPException as exc:
                if exc.status_code != 504:
                    raise
                yield str(exc.detail) + "\n\nPublic search excerpts (not a generated answer):\n\n"
                yield source_excerpts(sources)
                return
            if answered:
                yield source_links(sources)
            else:
                yield NO_WEB_ANSWER


        except HTTPException as exc:
            if exc.status_code == 404:
                yield "No documents are indexed yet. Upload a document first."
                return
            raise
        except Exception as e:
            logger.error("Error executing RAG pipeline: %s", e, exc_info=True)
            raise
        finally:
            logger.info(
                "RAGService.ask completed in %.3f seconds",
                time.perf_counter() - start_time,
            )
