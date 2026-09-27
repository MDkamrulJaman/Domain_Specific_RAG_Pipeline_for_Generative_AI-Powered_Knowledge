from typing import Literal
from pydantic import BaseModel, Field, field_validator

Provider = Literal["nvidia", "pinecone"]


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=12000)
    top_k: int = Field(default=5, ge=1, le=50)
    chat_id: str | None = None
    provider: Provider = "pinecone"
    enable_thinking: bool | None = None

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("Enter a question.")
        return value.strip()
