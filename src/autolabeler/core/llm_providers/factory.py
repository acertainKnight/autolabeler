from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from loguru import logger

from ...config import Settings
from .corporate import CorporateOpenAIClient
from .openrouter import OpenRouterClient

if TYPE_CHECKING:
    from ..configs import LabelingConfig


def get_llm_client(
    settings: Settings, config: "LabelingConfig"
) -> BaseChatModel:
    """
    Factory function to get the appropriate LLM client based on settings.

    Args:
        settings: The application settings from the environment.
        config: The labeling-specific configuration.

    Returns:
        An instance of a LangChain BaseChatModel.
    """
    if settings.corporate_base_url and settings.corporate_api_key:
        logger.info("Using Corporate LLM client.")
        return CorporateOpenAIClient(
            api_key=settings.corporate_api_key,
            base_url=settings.corporate_base_url,
            model=config.model_name or "gpt-3.5-turbo",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if settings.openrouter_api_key:
        logger.info("Using OpenRouter LLM client.")
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            model=config.model_name or settings.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ValueError(
        "No LLM provider configured. Please set either "
        "CORPORATE_BASE_URL and CORPORATE_API_KEY, or "
        "OPENROUTER_API_KEY in your .env file."
    )
