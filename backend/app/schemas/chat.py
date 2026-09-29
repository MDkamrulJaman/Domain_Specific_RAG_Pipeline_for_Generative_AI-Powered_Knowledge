"""SRP: transport validation only; provider execution lives in AnswerCommand."""
from typing import Literal
from pydantic import BaseModel, Field, field_validator

Provider = Literal["nvidia", "pinecone"]


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=12000)
    top_k: int | None = Field(default=5, ge=1, le=50, description="NVIDIA passage limit; null selects the default of 5. Ignored by Assistant.")
    web_search: bool = Field(default=False, description="Allow Tavily fallback only when documents cannot answer; sends the question externally on fallback.")
    chat_id: str | None = None
    provider: Provider = "pinecone"
    enable_thinking: bool | None = None

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("Enter a question.")
        return value.strip()
