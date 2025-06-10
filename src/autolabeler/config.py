from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Configuration
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/openai/gpt-3.5-turbo"  # default model id
    openai_api_key: str | None = None  # for embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    corporate_api_key: str | None = None
    corporate_base_url: str | None = None
    corporate_model: str = "gpt-3.5-turbo"

    # Knowledge Base Configuration
    knowledge_base_dir: Path = Path("knowledge_bases")
    vector_store_format: str = "faiss"  # Future: support other formats
    max_examples_per_query: int = 5
    similarity_threshold: float = 0.7  # Minimum similarity for examples

    # Model Generation Tracking
    track_model_labels: bool = True
    include_generation_metadata: bool = True
