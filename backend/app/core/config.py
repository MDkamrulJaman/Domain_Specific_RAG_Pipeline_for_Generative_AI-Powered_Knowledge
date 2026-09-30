from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env", extra="ignore",
    )


class AssistantSettings(EnvironmentSettings):
    """Credentials for the managed Assistant file library and generation."""
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
