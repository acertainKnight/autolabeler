"""OpenRouter client for autolabeler with rate limiting support.

Adapted from project-thoth implementation to support high-throughput labeling.
Supports 500 req/sec with automatic credit-based rate limiting.
"""

import os
import time
from typing import Any

import requests
from loguru import logger


class OpenRouterRateLimiter:
    """Rate limiter for OpenRouter API based on available credits.

    OpenRouter allows 1 request per credit per second up to a surge limit.
    This class checks available credits and configures rate limiting accordingly.

    Args:
        api_key: OpenRouter API key
        max_surge_limit: Maximum requests per second (default: 500)
        min_requests_per_second: Minimum rate when credits < 1 (default: 1.0)
    """

    def __init__(
        self,
        api_key: str,
        max_surge_limit: int = 500,
        min_requests_per_second: float = 1.0,
    ):
        self.api_key = api_key
        self.max_surge_limit = max_surge_limit
        self.min_requests_per_second = min_requests_per_second
        self.credits = None
        self.requests_per_second = None
        self.last_request_time = 0
        self.request_interval = 0

    def _get_credits(self) -> float | None:
        """Get available credits for the API key."""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                usage = data.get("usage", 0)
                limit = data.get("limit")

                if limit is not None:
                    return max(0, limit - usage)
                # Unlimited credits
                return float(self.max_surge_limit)
            else:
                logger.error(
                    f"Failed to get credits: {response.status_code} {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error getting credits: {e}")
            return None

    def setup(self):
        """Set up rate limiter based on available credits."""
        self.credits = self._get_credits()

        if self.credits is None:
            requests_per_second = self.min_requests_per_second
            logger.warning(
                f"Unable to determine credits. Setting rate limit to {requests_per_second} req/s"
            )
        else:
            if self.credits < 1:
                requests_per_second = self.min_requests_per_second
            else:
                requests_per_second = min(
                    float(self.credits), float(self.max_surge_limit)
                )

            logger.info(
                f"Available credits: {self.credits:.2f}, rate limit: {requests_per_second} req/s"
            )

        self.requests_per_second = requests_per_second
        self.request_interval = 1.0 / requests_per_second

    def acquire(self):
        """Acquire permission to make a request (blocks until allowed)."""
        if self.requests_per_second is None:
            self.setup()

        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_interval:
            sleep_time = self.request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class OpenRouterClient:
    """Client for OpenRouter API with automatic rate limiting.

    Provides OpenAI-compatible interface for multiple LLM providers through OpenRouter.

    Args:
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        model: Model identifier (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4")
        temperature: Sampling temperature (default: 0.1)
        max_tokens: Maximum tokens to generate
        use_rate_limiter: Enable automatic rate limiting (default: True)
        site_url: Your site URL for OpenRouter rankings
        site_name: Your site name for OpenRouter rankings
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.1,
        max_tokens: int | None = None,
        use_rate_limiter: bool = True,
        site_url: str | None = None,
        site_name: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY or pass api_key parameter."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://openrouter.ai/api/v1"
        self.site_url = site_url or "http://localhost:8000"
        self.site_name = site_name or "AutoLabeler"

        self.rate_limiter = None
        if use_rate_limiter:
            self.rate_limiter = OpenRouterRateLimiter(api_key=self.api_key)
            self.rate_limiter.setup()

    def create(self, messages: list[dict], **kwargs) -> dict:
        """Create a chat completion (OpenAI-compatible interface).

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            OpenAI-compatible response dict
        """
        if self.rate_limiter:
            self.rate_limiter.acquire()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if self.max_tokens:
            payload["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API error: {e}")
            raise


def get_openrouter_models() -> list[dict[str, Any]]:
    """Fetch list of available models from OpenRouter."""
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        logger.info(f"Fetched {len(models)} models from OpenRouter")
        return models
    except requests.RequestException as e:
        logger.error(f"Failed to fetch models: {e}")
        return []
