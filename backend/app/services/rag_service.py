import logging
import time
from fastapi import HTTPException

from app.services.contracts import Retriever, TextGenerator
from app.services.prompts import build_rag_prompt
from app.services.skills import SkillName, get_skill

logger = logging.getLogger(__name__)


# Strategy + DIP: retrieval and generation can vary independently through protocols.
class RAGService:
    def __init__(self, vectorstore: Retriever, llm: TextGenerator):
        self.vectorstore = vectorstore
        self.llm = llm

    def ask(self, query: str, top_k: int | None, skill: SkillName = "general") -> str:
        return "".join(self.stream(query, top_k, skill=skill))

    def stream(self, query: str, top_k: int | None = None, enable_thinking: bool | None = None,
               skill: SkillName = "general"):
        start_time = time.perf_counter()
        policy = get_skill(skill)
        effective_top_k = policy.retrieval_top_k if top_k is None else top_k

        if not query or not query.strip():
            logger.info("RAGService.ask completed in %.3f seconds", time.perf_counter() - start_time)
            yield "Please provide a valid query."
            return
        if effective_top_k < 1 or effective_top_k > 50:
            logger.info("RAGService.ask completed in %.3f seconds", time.perf_counter() - start_time)
            yield "The number of retrieved passages must be between 1 and 50."
            return

        try:
            docs = self.vectorstore.search(query, effective_top_k)
            logger.info("NVIDIA retrieval skill=%s mode=%s requested=%d returned=%d",
                        skill, "automatic" if top_k is None else "manual", effective_top_k, len(docs))
            if not docs:
                yield "No relevant documents were found in the knowledge base."
                return

            prompt = build_rag_prompt(query, docs, skill)
            options = {"enable_thinking": enable_thinking}
            if policy.max_tokens is not None:
                options["max_tokens"] = policy.max_tokens
            received = False
            # Iterator: forward fragments lazily instead of buffering the entire answer.
            for text in self.llm.stream(prompt, **options):
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
