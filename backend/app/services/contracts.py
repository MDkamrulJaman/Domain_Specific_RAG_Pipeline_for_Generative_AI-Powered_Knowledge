"""Small structural interfaces consumed by application workflows.

Adapters need only implement the capabilities they use; no shared base class
forces a chat model to implement uploading or vector operations.
"""
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol
from app.services.skills import SkillName

Progress = Callable[[float, str], None]


def ignore_progress(fraction: float, message: str) -> None:
    pass


class ChatService(Protocol):
    def stream(self, query: str, top_k: int | None = 5,
               enable_thinking: bool | None = None) -> Iterator[str]: ...


class TextGenerator(Protocol):
    def stream(self, prompt: str, enable_thinking: bool | None = None,
               max_tokens: int | None = None) -> Iterator[str]: ...


class SkilledChatService(ChatService, Protocol):
    def stream(self, query: str, top_k: int | None = 5, enable_thinking: bool | None = None,
               skill: SkillName = "general") -> Iterator[str]: ...


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class DocumentLoader(Protocol):
    def load_bytes(self, filename: str, content: bytes) -> Sequence[Any]: ...


class DocumentChunker(Protocol):
    def chunk_documents(self, documents: Sequence[Any]) -> Sequence[Any]: ...


class ChunkEmbedder(Protocol):
    def embed_chunks(self, chunks: Sequence[Any]) -> Any: ...


class VectorWriter(Protocol):
    integrated_embedding: bool
    def add(self, embeddings: Any, chunks: Sequence[Any]) -> None: ...
    def save(self) -> None: ...


class FileLibrary(Protocol):
    def upload(self, filename: str, content: bytes) -> dict[str, Any]: ...


class DocumentUploader(Protocol):
    def __call__(self, filename: str, content: bytes, progress: Progress) -> dict[str, Any]: ...
