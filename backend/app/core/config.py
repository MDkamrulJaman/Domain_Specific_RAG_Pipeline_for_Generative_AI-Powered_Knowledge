from pathlib import Path
from pydantic import Field
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
