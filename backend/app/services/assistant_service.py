from io import BytesIO
from fastapi import HTTPException
from pinecone import Pinecone
from pinecone.errors.exceptions import ApiError
from app.core.config import AssistantSettings
from app.services.skills import SkillName, assistant_policy


# Adapter: expose Assistant SDK operations through application-facing methods.
class AssistantService:
    """Managed document library and document-first streamed answers."""
    def __init__(self, settings=None, client=None, web_search_tool=None):
        self.web_search_tool = web_search_tool
        settings = settings if settings is not None else AssistantSettings()
        self.client = client if client is not None else Pinecone(api_key=settings.PINECONE_ASSISTANT_API_KEY)
        self.name = settings.PINECONE_ASSISTANT_NAME
        self.model = settings.PINECONE_ASSISTANT_MODEL
        self.timeout = settings.PINECONE_ASSISTANT_TIMEOUT_SECONDS

    def stream(self, query: str, web_search: bool = False, skill: SkillName = "general"):
        if not query.strip():
            yield "Please provide a valid query."
            return
        from app.services.web_search import web_context, source_links, source_excerpts
        from app.services.answer_evidence import (
            evidence_instruction, supported_stream, NO_DOCUMENT_ANSWER, NO_WEB_ANSWER,
        )
        policy = assistant_policy(skill)
        no_files = False
        try:
            answered = yield from supported_stream(self._chat(
                query + policy + evidence_instruction("your uploaded document library")))
            if answered:
                return
        except ApiError as exc:
            if "No files found" not in str(exc):
                raise
            no_files = True
        if not web_search:
            yield NO_DOCUMENT_ANSWER
            return
        if self.web_search_tool is None:
            raise HTTPException(503, "Web search is not configured.")
        sources = self.web_search_tool.search(query)
        if not sources:
            yield NO_WEB_ANSWER
            return
        if no_files:
            # Managed Assistant cannot generate without files. Never silently change providers.
            yield "Assistant requires an available uploaded file to generate an answer. Public search excerpts (not a generated answer):\n\n"
            yield source_excerpts(sources)
            return
        yield "*Documents do not contain the answer. Searching public sources.*\n\n"
        answered = yield from supported_stream(self._chat(
            web_context(query, sources) + policy + evidence_instruction("the supplied web evidence")))
        yield source_links(sources) if answered else NO_WEB_ANSWER

    def _chat(self, query):
        response = self.client.assistants.chat(
            assistant_name=self.name, messages=[{"role": "user", "content": query}],
            model=self.model, stream=True, timeout=self.timeout,
        )
        try:
            yield from response.text()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def upload(self, filename: str, content: bytes):
        with BytesIO(content) as stream:
            file = self.client.assistants.upload_file(
                assistant_name=self.name, file_stream=stream, file_name=filename, timeout=-1,
            )
        return {"file_id": file.id, "name": file.name, "status": str(file.status)}

    def list_files(self):
        return [{"file_id": file.id, "name": file.name, "status": str(file.status)}
                for file in self.client.assistants.list_files(assistant_name=self.name)]

    def inspect_status(self):
        return str(self.client.assistants.describe(name=self.name).status)
