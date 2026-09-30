"""Small structural interfaces consumed by application workflows.

Adapters need only implement the capabilities they use; no shared base class
forces a chat model to implement uploading or vector operations.
"""
from collections.abc import Callable, Iterator
from typing import Any, Protocol
from app.services.skills import SkillName

Progress = Callable[[float, str], None]


def ignore_progress(fraction: float, message: str) -> None:
    pass


class ChatService(Protocol):
    def stream(self, query: str, web_search: bool = False,
               skill: SkillName = "general") -> Iterator[str]: ...


class FileLibrary(Protocol):
    def upload(self, filename: str, content: bytes) -> dict[str, Any]: ...


class DocumentUploader(Protocol):
    def __call__(self, filename: str, content: bytes, progress: Progress) -> dict[str, Any]: ...


class SearchTool(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...
