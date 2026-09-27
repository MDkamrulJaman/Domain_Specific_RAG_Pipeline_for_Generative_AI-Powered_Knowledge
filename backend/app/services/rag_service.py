import logging
import time
from fastapi import HTTPException

from app.services.contracts import Retriever, TextGenerator
from app.services.prompts import build_rag_prompt

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, vectorstore: Retriever, llm: TextGenerator):
        self.vectorstore = vectorstore
        self.llm = llm

    def ask(self, query: str, top_k: int) -> str:
        return "".join(self.stream(query, top_k))

    def stream(self, query: str, top_k: int = 5, enable_thinking: bool | None = None):
        start_time = time.perf_counter()

        if not query or not query.strip():
            logger.info("RAGService.ask completed in %.3f seconds", time.perf_counter() - start_time)
            yield "Please provide a valid query."
            return
        if top_k < 1:
            logger.info("RAGService.ask completed in %.3f seconds", time.perf_counter() - start_time)
            yield "The number of retrieved documents must be at least 1."
            return

        try:
            docs = self.vectorstore.search(query, top_k)
            if not docs:
                yield "No relevant documents were found in the knowledge base."
                return

            prompt = build_rag_prompt(query, docs)
            received = False
            for text in self.llm.stream(prompt, enable_thinking=enable_thinking):
                received = True
                yield text
            if not received:
                yield "The model did not return an answer."

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