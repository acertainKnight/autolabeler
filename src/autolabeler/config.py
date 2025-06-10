from __future__ import annotations

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/openai/gpt-3.5-turbo"  # default model id
    openai_api_key: str | None = None  # for embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    corporate_api_key: str | None = None
    corporate_base_url: str | None = None
    corporate_model: str = "gpt-3.5-turbo"

    class Config:
        env_file = ".env"
        case_sensitive = False
