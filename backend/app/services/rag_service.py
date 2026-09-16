import logging
import time
from fastapi import HTTPException

from app.pipeline.retrieval_service import VectorStore
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        try:
            self.vectorstore = VectorStore()
            self.llm = LLMService()
        except Exception as e:
            logger.critical(f"RAG component initialization failed: {str(e)}")
            raise

    def ask(self, query: str, top_k: int) -> str:
        start_time = time.perf_counter()

        if not query or not query.strip():
            logger.info("RAGService.ask completed in %.3f seconds", time.perf_counter() - start_time)
            return "Please provide a valid query."
        if top_k < 1:
            logger.info("RAGService.ask completed in %.3f seconds", time.perf_counter() - start_time)
            return "The number of retrieved documents must be at least 1."

        try:
            docs = self.vectorstore.search(query, top_k)
            if not docs:
                return "No relevant documents were found in the knowledge base."

            context = "\n\n---\n\n".join(
                document.get("text", "") for document in docs if document.get("text")
            )

            prompt = f"""Answer the question using only the context below.
            If the context does not contain the answer, say that you do not know.
            Do not invent facts.   

            Context:
            {context}

            Question:
            {query}

            Answer:"""
            response = self.llm.generate(prompt)
            return response or "The model did not return an answer."

        except HTTPException as exc:
            if exc.status_code == 404:
                return "No documents are indexed yet. Upload a document first."
            raise
        except Exception as e:
            logger.error("Error executing RAG pipeline: %s", e, exc_info=True)
            return "An internal error occurred while executing the search pipeline."
        finally:
            logger.info(
                "RAGService.ask completed in %.3f seconds",
                time.perf_counter() - start_time,
            )