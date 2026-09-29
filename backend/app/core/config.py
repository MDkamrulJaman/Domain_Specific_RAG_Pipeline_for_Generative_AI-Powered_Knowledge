from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env", extra="ignore",
    )


class RetrievalSettings(EnvironmentSettings):
    """Shared Hugging Face embeddings and Pinecone vector database."""
    EMBEDDING_MODEL: str
    HF_TOKEN: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    PINECONE_DIMENSION: int
    PINECONE_NAMESPACE: str
    PINECONE_MODEL: str


class NvidiaSettings(EnvironmentSettings):
    MODEL_BASE_URL: str
    MODEL_NAME: str
    MODEL_API_KEY: str
    MODEL_TIMEOUT_SECONDS: float
    MODEL_TEMPERATURE: float
    MODEL_TOP_P: float
    MODEL_MAX_TOKENS: int
    MODEL_ENABLE_THINKING: bool = False
    # Interactive chat must not silently repeat a full timed-out request.
    # Operators may opt in to retries, accepting the extra latency.
    MODEL_MAX_RETRIES: int = Field(default=0, ge=0, le=2)


class Settings(RetrievalSettings, NvidiaSettings):
    """Compatibility settings for applications loading both NVIDIA components."""


class AssistantSettings(EnvironmentSettings):
    """Separate credentials for Assistant generation, not vector storage."""
    PINECONE_ASSISTANT_API_KEY: str
    PINECONE_ASSISTANT_NAME: str
    PINECONE_ASSISTANT_MODEL: str
    PINECONE_ASSISTANT_TIMEOUT_SECONDS: float


class AppSettings(EnvironmentSettings):
    """Operational limits independent of provider credentials."""
    MAX_UPLOAD_MB: int = Field(default=5, ge=1, le=100)
    UI_QUEUE_SIZE: int = Field(default=8, ge=1, le=1000)
    UI_CONCURRENCY: int = Field(default=1, ge=1, le=32)

    @property
    def max_upload_bytes(self):
        return self.MAX_UPLOAD_MB * 1024 * 1024


class WebSearchSettings(EnvironmentSettings):
    """Optional search credentials; never required for ordinary document chat."""
    TAVILY_API_KEY: SecretStr = SecretStr("")
    WEB_SEARCH_MAX_RESULTS: int = Field(default=3, ge=1, le=5)
    WEB_SEARCH_TIMEOUT_SECONDS: float = Field(default=8, ge=1, le=30)


class NvidiaWebSettings(EnvironmentSettings):
    """Smaller budgets apply only to NVIDIA requests with web search enabled."""
    NVIDIA_WEB_DOCUMENT_CHARS: int = Field(default=6000, ge=1000, le=20000)
    NVIDIA_WEB_EXCERPT_CHARS: int = Field(default=1000, ge=200, le=2000)
    NVIDIA_WEB_MAX_TOKENS: int = Field(default=512, ge=128, le=4096)
