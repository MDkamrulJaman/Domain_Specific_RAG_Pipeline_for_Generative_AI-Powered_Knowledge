from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_BASE_URL: str
    MODEL_NAME: str
    MODEL_API_KEY: str
    MODEL_TIMEOUT_SECONDS: float
    MODEL_TEMPERATURE: float 
    MODEL_TOP_P: float 
    MODEL_MAX_TOKENS: int 
    EMBEDDING_MODEL: str 
    HF_TOKEN: str 
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    PINECONE_DIMENSION: int
    PINECONE_NAMESPACE: str 
    PINECONE_MODEL: str 
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        extra="ignore",
    )
