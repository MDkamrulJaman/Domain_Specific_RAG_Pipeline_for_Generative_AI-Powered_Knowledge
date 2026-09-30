"""Provider metadata and capabilities; independent of API and SDK clients."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from app.services.contracts import ChatService, DocumentUploader


@dataclass(frozen=True)
class ProviderDefinition:
    label: str
    chat_factory: Callable[[], ChatService]
    upload: DocumentUploader
    configuration: Callable[[], tuple[Any, tuple[str, ...], str]]
    inspect: Callable[[ChatService], dict[str, Any]]


# Strategy registry: choose interchangeable upload behavior by provider name.
# Factories are callables, not a GoF Factory Method subclass hierarchy.
class ProviderRegistry:
    def __init__(self, definitions: dict[str, ProviderDefinition]):
        self._definitions = dict(definitions)

    def resolve(self, provider: str) -> ProviderDefinition:
        try:
            return self._definitions[provider]
        except KeyError:
            raise ValueError(f"Unsupported provider: {provider}") from None

    @property
    def labels(self):
        return {key: value.label for key, value in self._definitions.items()}
