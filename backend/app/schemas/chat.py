"""SRP: transport validation only; provider execution lives in AnswerCommand."""
from typing import Literal
from app.services.skills import SkillName
from pydantic import BaseModel, Field, field_validator

Provider = Literal["pinecone"]


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=12000)
    web_search: bool = Field(default=False, description="Pinecone Assistant only: allow web fallback when documents cannot answer.")
    skill: SkillName = Field(default="general", description="Pinecone Assistant task policy.")
    chat_id: str | None = None
    provider: Provider = "pinecone"

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("Enter a question.")
        return value.strip()
