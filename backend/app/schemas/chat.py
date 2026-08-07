from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3