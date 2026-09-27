from io import BytesIO
from fastapi import HTTPException
from pinecone import Pinecone
from pinecone.errors.exceptions import ApiError
from app.core.config import AssistantSettings


class AssistantService:
    """Assistant uses its own uploaded file library, independently of rag."""
    def __init__(self):
        settings = AssistantSettings()
        self.client = Pinecone(api_key=settings.PINECONE_ASSISTANT_API_KEY)
        self.name = settings.PINECONE_ASSISTANT_NAME
        self.model = settings.PINECONE_ASSISTANT_MODEL
        self.timeout = settings.PINECONE_ASSISTANT_TIMEOUT_SECONDS

    def stream(self, query: str, top_k: int = 5, enable_thinking: bool | None = None):
        if not query.strip():
            yield "Please provide a valid query."
            return
        try:
            response = self.client.assistants.chat(
                assistant_name=self.name,
                messages=[{"role": "user", "content": query}],
                model=self.model, stream=True, timeout=self.timeout,
            )
            yield from response.text()
        except ApiError as exc:
            if "No files found" in str(exc):
                raise HTTPException(409, "Assistant has no available files. Select Pinecone Assistant and upload a document, "
                                    "then refresh until Assistant files are Available. Existing rag "
                                    "index documents are not automatically in Assistant.") from exc
            raise

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
