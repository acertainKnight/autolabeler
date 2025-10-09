from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from loguru import logger

from ...config import Settings
from .corporate import CorporateOpenAIClient
from .openrouter import OpenRouterClient
from ..utils.budget_tracker import CostTracker

if TYPE_CHECKING:
    from ..configs import LabelingConfig


def get_llm_client(
    settings: Settings, config: "LabelingConfig", cost_tracker: CostTracker | None = None
) -> BaseChatModel:
    """
    Factory function to get the appropriate LLM client based on settings.

    Args:
        settings: The application settings from the environment.
        config: The labeling-specific configuration.
        cost_tracker: Optional cost tracker for budget management.

    Returns:
        An instance of a LangChain BaseChatModel with budget tracking enabled.

    Note:
        If cost_tracker is not provided but a budget is set in config or settings,
        a new CostTracker will be created automatically.
    """
    # Create cost tracker if budget is specified but tracker not provided
    if cost_tracker is None:
        budget = config.budget or settings.llm_budget
        if budget is not None:
            cost_tracker = CostTracker(budget=budget)
            logger.info(f"Created cost tracker with budget: ${budget:.2f}")

    if settings.corporate_base_url and settings.corporate_api_key:
        logger.info("Using Corporate LLM client.")
        return CorporateOpenAIClient(
            api_key=settings.corporate_api_key,
            base_url=settings.corporate_base_url,
            model=config.model_name or "gpt-3.5-turbo",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            cost_tracker=cost_tracker,
        )

    if settings.openrouter_api_key:
        logger.info("Using OpenRouter LLM client.")
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            model=config.model_name or settings.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            cost_tracker=cost_tracker,
        )

    raise ValueError(
        "No LLM provider configured. Please set either "
        "CORPORATE_BASE_URL and CORPORATE_API_KEY, or "
        "OPENROUTER_API_KEY in your .env file."
    )
