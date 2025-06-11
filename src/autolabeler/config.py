from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables (.env file).

    Manages API keys, model identifiers, and other global configurations.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Provider Keys
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    corporate_api_key: str | None = None

    # LLM Endpoints & Models
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    corporate_base_url: str | None = None

    # Default model selections
    llm_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Knowledge Store & RAG
    knowledge_base_dir: Path = Path("knowledge_bases")
    max_examples_per_query: int = 5
    similarity_threshold: float = 0.8

    # Directory Paths
    data_dir: Path = Path("datasets")
    results_dir: Path = Path("results")
    prompt_dir: Path = Path("prompt_store")
    ruleset_dir: Path = Path("rulesets")

    # Default Template Paths
    template_path: Path | None = None

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "{time} | {level} | {message}"
