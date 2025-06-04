from __future__ import annotations

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/openai/gpt-3.5-turbo"  # default model id
    openai_api_key: str | None = None  # for embeddings
    embedding_model: str = "text-embedding-ada-002"

    class Config:
        env_file = ".env"
        case_sensitive = False
