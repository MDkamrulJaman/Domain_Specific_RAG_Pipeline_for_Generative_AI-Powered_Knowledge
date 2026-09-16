from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    chat_id: str | None = None