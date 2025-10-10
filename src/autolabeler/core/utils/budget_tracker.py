"""Budget tracking utilities for LLM API calls.

This module provides functionality to track costs across different LLM providers
(OpenRouter, OpenAI, Corporate) and enforce budget limits during labeling operations.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from loguru import logger


@dataclass
class CostTracker:
    """Thread-safe cost tracking for LLM API calls.

    Tracks cumulative costs and checks against budget limits to enable
    graceful shutdown when budget is exceeded.

    Attributes:
        budget: Maximum allowed spending in USD (None = unlimited)
        total_cost: Cumulative cost in USD across all API calls
        call_count: Total number of API calls made
        _lock: Thread lock for thread-safe operations
        _budget_exceeded: Flag indicating if budget has been exceeded
    """

    budget: float | None = None
    total_cost: float = 0.0
    call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _budget_exceeded: bool = field(default=False, init=False, repr=False)

    def add_cost(self, cost: float) -> bool:
        """Add cost from an API call and check budget.

        Thread-safe method to accumulate costs and check if budget has been exceeded.

        Args:
            cost: Cost of the API call in USD

        Returns:
            bool: True if within budget, False if budget exceeded
        """
        with self._lock:
            self.total_cost += cost
            self.call_count += 1

            # Check if budget exceeded
            if self.budget is not None and self.total_cost >= self.budget:
                if not self._budget_exceeded:
                    self._budget_exceeded = True
                    logger.warning(
                        f"Budget limit reached: ${self.total_cost:.4f} >= ${self.budget:.4f} "
                        f"after {self.call_count} calls"
                    )
                return False
            return True

    def is_budget_exceeded(self) -> bool:
        """Check if budget has been exceeded.

        Returns:
            bool: True if budget exceeded, False otherwise
        """
        with self._lock:
            return self._budget_exceeded

    def get_stats(self) -> dict[str, Any]:
        """Get current cost statistics.

        Returns:
            dict: Statistics including total_cost, call_count, budget, and remaining budget
        """
        with self._lock:
            remaining = None
            if self.budget is not None:
                remaining = max(0.0, self.budget - self.total_cost)

            return {
                "total_cost": self.total_cost,
                "call_count": self.call_count,
                "budget": self.budget,
                "remaining_budget": remaining,
                "budget_exceeded": self._budget_exceeded,
            }

    def reset(self) -> None:
        """Reset cost tracking (for testing or new runs)."""
        with self._lock:
            self.total_cost = 0.0
            self.call_count = 0
            self._budget_exceeded = False


class BudgetExceededError(Exception):
    """Exception raised when LLM budget is exceeded."""

    def __init__(self, total_cost: float, budget: float) -> None:
        """Initialize with cost information.

        Args:
            total_cost: Total cost accumulated in USD
            budget: Budget limit in USD
        """
        self.total_cost = total_cost
        self.budget = budget
        super().__init__(
            f"Budget exceeded: ${total_cost:.4f} >= ${budget:.4f}"
        )


def extract_openrouter_cost(result: LLMResult) -> float:
    """Extract cost information from OpenRouter API response.

    OpenRouter returns cost information in multiple possible locations:
    1. response_metadata (in the AIMessage) - most common for direct cost
    2. llm_output (in the LLMResult) - contains usage and cost details
    3. Usage-based calculation as fallback

    Args:
        result: LangChain LLMResult from OpenRouter API call

    Returns:
        float: Total cost in USD, or 0.0 if cost information not available

    Example response data structure:
        response_metadata: {
            "total_cost": 0.000234,
            "usage": {...}
        }
        OR
        llm_output: {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "total_tokens": 200
            },
            "total_cost": 0.000234,
            "native_tokens_prompt": 150,
            "native_tokens_completion": 50
        }
    """
    try:
        # Check if we have generations
        if not result.generations or not result.generations[0]:
            logger.debug("No generations in LLMResult")
            return 0.0

        # Get first generation
        generation = result.generations[0][0]
        if not isinstance(generation, ChatGeneration):
            logger.debug("Generation is not ChatGeneration")
            return 0.0

        # Strategy 1: Check response_metadata in the AIMessage
        message = generation.message
        if isinstance(message, AIMessage):
            response_metadata = getattr(message, "response_metadata", {})

            # Try to extract cost from response_metadata
            total_cost = (
                response_metadata.get("total_cost") or
                response_metadata.get("cost") or
                0.0
            )

            if total_cost > 0:
                logger.debug(f"Extracted OpenRouter cost from response_metadata: ${total_cost:.6f}")
                return float(total_cost)

            # Log usage from response_metadata for debugging
            usage = response_metadata.get("usage", {})
            if usage:
                logger.debug(
                    f"OpenRouter usage in response_metadata: "
                    f"prompt={usage.get('prompt_tokens')}, "
                    f"completion={usage.get('completion_tokens')}, "
                    f"total={usage.get('total_tokens')}"
                )

        # Strategy 2: Check llm_output in the LLMResult (primary location for OpenRouter)
        llm_output = result.llm_output or {}

        # Extract cost from llm_output
        total_cost = (
            llm_output.get("total_cost") or
            llm_output.get("cost") or
            0.0
        )

        if total_cost > 0:
            logger.debug(f"Extracted OpenRouter cost from llm_output: ${total_cost:.6f}")
            return float(total_cost)

        # Strategy 3: Check usage in llm_output
        usage = llm_output.get("usage", {})
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            logger.debug(
                f"OpenRouter usage in llm_output: "
                f"prompt={prompt_tokens}, "
                f"completion={completion_tokens}, "
                f"total={usage.get('total_tokens', 0)}"
            )

            # Check if there's cost data embedded in usage
            usage_cost = usage.get("total_cost") or usage.get("cost")
            if usage_cost and usage_cost > 0:
                logger.debug(f"Extracted OpenRouter cost from usage field: ${usage_cost:.6f}")
                return float(usage_cost)

        # Strategy 4: Check for native_tokens fields with cost
        native_prompt = llm_output.get("native_tokens_prompt")
        native_completion = llm_output.get("native_tokens_completion")
        if native_prompt or native_completion:
            logger.debug(
                f"OpenRouter native tokens: "
                f"prompt={native_prompt}, completion={native_completion}"
            )

        # If we got here, no cost was found
        logger.debug("No cost information found in OpenRouter response")
        return 0.0

    except Exception as e:
        logger.warning(f"Error extracting OpenRouter cost: {e}")
        return 0.0


def extract_openai_cost(result: LLMResult, model: str) -> float:
    """Extract and calculate cost from OpenAI API response.

    OpenAI returns token usage but not direct cost. We calculate cost based on
    the pricing for the specific model used.

    Pricing (as of 2025-01):
        gpt-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
        gpt-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
        gpt-3.5-turbo: $0.50 per 1M input tokens, $1.50 per 1M output tokens

    Args:
        result: LangChain LLMResult from OpenAI API call
        model: Model name used for the API call

    Returns:
        float: Estimated cost in USD, or 0.0 if usage information not available
    """
    # Define pricing per 1M tokens (input, output)
    PRICING = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-3.5-turbo": (0.50, 1.50),
        "gpt-3.5-turbo-16k": (0.50, 1.50),
        "gpt-4": (30.00, 60.00),
        "gpt-4-32k": (60.00, 120.00),
        "gpt-4-turbo": (10.00, 30.00),
    }

    try:
        # Get token usage from llm_output
        llm_output = result.llm_output or {}
        token_usage = llm_output.get("token_usage", {})

        if not token_usage:
            logger.debug("No token_usage in LLMResult")
            return 0.0

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)

        if prompt_tokens == 0 and completion_tokens == 0:
            logger.debug("Token counts are zero")
            return 0.0

        # Find pricing for the model
        # Try exact match first, then prefix match
        pricing = None
        for model_key, model_pricing in PRICING.items():
            if model.startswith(model_key):
                pricing = model_pricing
                break

        if pricing is None:
            logger.warning(
                f"Unknown OpenAI model for cost calculation: {model}. "
                f"Using gpt-3.5-turbo pricing as fallback."
            )
            pricing = PRICING["gpt-3.5-turbo"]

        # Calculate cost: (tokens / 1,000,000) * price_per_million
        input_cost = (prompt_tokens / 1_000_000) * pricing[0]
        output_cost = (completion_tokens / 1_000_000) * pricing[1]
        total_cost = input_cost + output_cost

        logger.debug(
            f"Calculated OpenAI cost for {model}: ${total_cost:.6f} "
            f"(input: {prompt_tokens} tokens @ ${pricing[0]}/1M, "
            f"output: {completion_tokens} tokens @ ${pricing[1]}/1M)"
        )

        return total_cost

    except Exception as e:
        logger.warning(f"Error calculating OpenAI cost: {e}")
        return 0.0


def extract_cost_from_result(result: LLMResult, provider: str, model: str) -> float:
    """Extract cost from LLM result based on provider.

    Routes to the appropriate cost extraction function based on the provider.
    For corporate/internal endpoints, returns 0.0 as they typically don't charge.

    Args:
        result: LangChain LLMResult from API call
        provider: Provider name ("openrouter", "openai", "corporate")
        model: Model name used for the API call

    Returns:
        float: Cost in USD, or 0.0 if not applicable/available
    """
    provider_lower = provider.lower()

    if "openrouter" in provider_lower:
        return extract_openrouter_cost(result)
    elif "openai" in provider_lower:
        return extract_openai_cost(result, model)
    elif "corporate" in provider_lower:
        # Corporate/internal endpoints typically don't charge
        logger.debug("Corporate endpoint - no cost tracking")
        return 0.0
    else:
        logger.warning(f"Unknown provider for cost tracking: {provider}")
        return 0.0
