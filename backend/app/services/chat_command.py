"""Command: one validated answer request, independent of its API/UI invoker."""
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from app.schemas.chat import ChatRequest, Provider
from app.services.contracts import ChatService, SkilledChatService
from app.services.skills import SkillName


@dataclass(frozen=True)
class AnswerCommand:
    """Snapshot request values; execute against an injected receiver without SDK setup.

    Execution streams immediately. This command does not implement persistence,
    automatic retries, or undo: those would need explicit product semantics.
    """
    query: str
    top_k: int | None
    provider: Provider
    enable_thinking: bool | None
    skill: SkillName

    @classmethod
    def from_request(cls, request: ChatRequest) -> "AnswerCommand":
        return cls(request.query, request.top_k, request.provider,
                   request.enable_thinking, request.skill)

    def execute(self, receiver: ChatService) -> Iterator[str]:
        # ISP: only the NVIDIA receiver is asked for the optional skill capability.
        if self.provider == "nvidia" and self.skill != "general":
            return cast(SkilledChatService, receiver).stream(
                self.query, self.top_k, enable_thinking=self.enable_thinking, skill=self.skill,
            )
        return receiver.stream(self.query, self.top_k, enable_thinking=self.enable_thinking)
