from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    DOCKER_SECRET: str = Field(..., env="DOCKER_SECRET")

    LOG_LEVEL: str = Field("DEBUG")
    TEST_MODE_DB: bool = Field(False, env="TEST_MODE_DB")

    REDIS_PASS: str = Field(..., env="REDIS_PASS")

    QWEN_API_KEY: str = Field(..., env="QWEN_API_KEY")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    DEEPSEEK_API_KEY: str = Field(..., env="DEEPSEEK_API_KEY")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

    EMBEDDING_MODEL_ID: str = Field(..., env="EMBEDDING_MODEL_ID")
    MODEL_SLEEP_AI: str = Field(..., env="MODEL_SLEEP_AI")
    MODEL_DIALOG_AI: str = Field(..., env="MODEL_DIALOG_AI")
    CHAT_HISTORY_URL: str = Field(..., env="CHAT_HISTORY_URL")

    QDRANT_HOST: str = Field(..., env="QDRANT_HOST")
    QDRANT_PORT: int = Field(..., env="QDRANT_PORT")
    COLLECTION_NAME_SLEEP_AI: str = Field(..., env="COLLECTION_NAME_SLEEP_AI")
    COLLECTION_NAME_DIALOG_AI: str = Field(..., env="COLLECTION_NAME_DIALOG_AI")
    VECTOR_DIMENSION: int = Field(..., env="VECTOR_DIMENSION")
    BATCH_SIZE: int = Field(..., env="BATCH_SIZE")
    RERANK_MODEL: str = Field(..., env="RERANK_MODEL")

    LANGFUSE_SECRET_KEY: str = Field(None, env="LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: str = Field(None, env="LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASE_URL: str = Field(None, env="LANGFUSE_BASE_URL")

    KAFKA_BROKER_URL: str = Field(..., env="KAFKA_BROKER_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = Settings()