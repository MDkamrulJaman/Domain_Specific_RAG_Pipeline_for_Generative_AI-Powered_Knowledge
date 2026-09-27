from typing import Literal
from pydantic import BaseModel, Field

Provider = Literal["nvidia", "pinecone"]


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    chat_id: str | None = None
    provider: Provider = "pinecone"
    enable_thinking: bool | None = None
