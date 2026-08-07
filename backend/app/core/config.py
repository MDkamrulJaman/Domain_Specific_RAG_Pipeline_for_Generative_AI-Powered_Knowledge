from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    FRONTEND_URL: str
    FRONTEND_PRODUCTION_URL: str
    DATABASE_URL: str
    SECRET_KEY: str
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434/")
    OLLAMA_MODEL: str = Field(default="gemma3")
    OLLAMA_TIMEOUT_SECONDS: float = Field(default=30.0)

    model_config = SettingsConfigDict(env_file=".env")
