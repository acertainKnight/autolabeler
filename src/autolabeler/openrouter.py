"""Client for the OpenRouter API leveraging the OpenAI API."""

from __future__ import annotations

import os
import time
from typing import Any, ClassVar

import requests
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from loguru import logger


class OpenRouterError(Exception):
    """Exception raised for errors in the OpenRouter API."""

    pass


class OpenRouterRateLimiter:
    """Rate limiter for the OpenRouter API based on available credits."""

    def __init__(
        self,
        api_key: str,
        max_surge_limit: int = 500,
        min_requests_per_second: float = 1.0,
        check_interval: float = 0.1,
    ) -> None:
        """Initialize the OpenRouter rate limiter with the specified parameters."""
        self.api_key = api_key
        self.max_surge_limit = max_surge_limit
        self.min_requests_per_second = min_requests_per_second
        self.check_interval = check_interval
        self.credits: float | None = None
        self.rate_limiter: InMemoryRateLimiter | None = None

    def _get_credits(self) -> float | None:
        """Get the available credits for the API key."""
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
                return float(self.max_surge_limit)
            logger.error(
                f"Failed to get credits: {response.status_code} {response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Error getting credits: {e}")
            return None

    def setup(self) -> None:
        """Set up the rate limiter based on the available credits."""
        self.credits = self._get_credits()

        if self.credits is None:
            requests_per_second = self.min_requests_per_second
            logger.warning(
                "Unable to determine credits. Setting rate limit to %s req/s",
                requests_per_second,
            )
        else:
            if self.credits < 1:
                requests_per_second = self.min_requests_per_second
            else:
                requests_per_second = min(
                    float(self.credits), float(self.max_surge_limit)
                )

            logger.info(
                "Available credits: %s, setting rate limit to %s req/s",
                self.credits,
                requests_per_second,
            )

        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=self.check_interval,
            max_bucket_size=min(10, self.max_surge_limit),
        )

    def acquire(self) -> None:
        """Acquire permission to make a request."""
        if self.rate_limiter is None:
            self.setup()

        if self.rate_limiter:
            self.rate_limiter.acquire()
        else:
            time.sleep(1.0 / self.min_requests_per_second)

    def get_langchain_limiter(self) -> InMemoryRateLimiter | None:
        """Return the underlying LangChain rate limiter."""
        if self.rate_limiter is None:
            self.setup()
        return self.rate_limiter


class OpenRouterClient(ChatOpenAI):
    """Client for the OpenRouter API leveraging the OpenAI API."""

    custom_attributes: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        streaming: bool = False,
        use_rate_limiter: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenRouter client."""
        api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("API_OPENROUTER_KEY")
        )
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY, "
                "API_OPENROUTER_KEY, or pass api_key."
            )

        if use_rate_limiter:
            rate_limiter = OpenRouterRateLimiter(api_key=api_key)
            rate_limiter.setup()
            kwargs["rate_limiter"] = rate_limiter.get_langchain_limiter()

        extra_headers = {
            "HTTP-Referer": site_url or "http://localhost:8000",
            "X-Title": site_name or "Thoth Research Assistant",
        }

        super().__init__(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            extra_headers=extra_headers,
            **kwargs,
        )

        instance_id = id(self)
        if instance_id not in self.custom_attributes:
            self.custom_attributes[instance_id] = {}

        self.custom_attributes[instance_id]["use_rate_limiter"] = use_rate_limiter
        self.custom_attributes[instance_id]["rate_limiter"] = None

        if use_rate_limiter and api_key:
            self.custom_attributes[instance_id]["rate_limiter"] = OpenRouterRateLimiter(
                api_key=api_key
            )

    def _generate(self, *args: Any, **kwargs: Any):
        """Synchronous method for generating completions with rate limiting."""
        instance_id = id(self)
        use_rate_limiter = self.custom_attributes.get(instance_id, {}).get(
            "use_rate_limiter", False
        )
        rate_limiter = self.custom_attributes.get(instance_id, {}).get("rate_limiter")

        if use_rate_limiter and rate_limiter:
            rate_limiter.acquire()

        return super()._generate(*args, **kwargs)
