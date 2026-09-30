"""Command: one validated answer request, independent of its API/UI invoker."""
from collections.abc import Iterator
from dataclasses import dataclass
import re
import unicodedata

from app.schemas.chat import ChatRequest, Provider
from app.services.contracts import ChatService
from app.services.skills import SkillName


@dataclass(frozen=True)
class AnswerCommand:
    """Snapshot request values; execute against an injected receiver without SDK setup.

    Execution streams immediately. This command does not implement persistence,
    automatic retries, or undo: those would need explicit product semantics.
    """
    query: str
    provider: Provider
    web_search: bool = False
    skill: SkillName = "general"

    @classmethod
    def from_request(cls, request: ChatRequest) -> "AnswerCommand":
        return cls(request.query, request.provider, request.web_search, request.skill)

    @property
    def local_reply(self) -> str | None:
        # Match the entire message so "good, explain X" remains a real question.
        text = unicodedata.normalize("NFKC", self.query).casefold().strip()
        text = re.sub(r"[.!?,]+$", "", text).strip()
        text = " ".join(text.split())
        if text in {"hi", "hello", "hey", "welcome", "wellcome", "good morning", "good afternoon", "good evening"}:
            return "Hello! How can I help with your documents?"
        if text in {"thanks", "thank you", "thanks a lot", "thank you very much"}:
            return "You're welcome!"
        if text in {"nice", "good", "great", "okay", "ok", "awesome", "well done"}:
            return "Glad to help!"
        return None

    def execute(self, receiver: ChatService | None) -> Iterator[str]:
        reply = self.local_reply
        if reply is not None:
            return iter([reply])
        if receiver is None:
            raise ValueError("A chat provider is required for this question.")
        return receiver.stream(self.query, web_search=self.web_search, skill=self.skill)
