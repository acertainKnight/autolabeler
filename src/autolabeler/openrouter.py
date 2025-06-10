"""Client for the OpenRouter API leveraging the OpenAI API."""

from __future__ import annotations

import os
import time
import weakref
from typing import Any

import requests
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLMResult
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
        """Initialize the OpenRouter rate limiter with the specified parameters.

        Args:
            api_key (str): OpenRouter API key.
            max_surge_limit (int): Maximum requests per second when credits are high.
            min_requests_per_second (float): Minimum requests per second when credits are low.
            check_interval (float): How often to check the rate limit.
        """
        self.api_key = api_key
        self.max_surge_limit = max_surge_limit
        self.min_requests_per_second = min_requests_per_second
        self.check_interval = check_interval
        self.credits: float | None = None
        self.rate_limiter: InMemoryRateLimiter | None = None

    def _get_credits(self) -> float | None:
        """Get the available credits for the API key.

        Returns:
            float | None: Available credits or None if unable to determine.
        """
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json().get("data", {})

                # OpenRouter API returns balance info in 'data' object
                # Check for credit_balance first (newer API format)
                if "credit_balance" in data:
                    return max(0.0, float(data["credit_balance"]))

                # Fallback to usage/limit calculation (older format)
                if "usage" in data and "limit" in data:
                    usage = float(data["usage"])
                    limit = float(data["limit"])
                    return max(0.0, limit - usage)

                # If no balance info available, use conservative default
                logger.warning("No credit balance information available in API response")
                return float(self.min_requests_per_second)

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
                # Scale rate limiting based on available credits
                # More credits = higher rate limit (up to max_surge_limit)
                requests_per_second = min(
                    float(self.credits) * 0.1,  # Conservative scaling factor
                    float(self.max_surge_limit)
                )

            logger.info(
                "Available credits: %s, setting rate limit to %s req/s",
                self.credits,
                requests_per_second,
            )

        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=self.check_interval,
            max_bucket_size=min(10, int(requests_per_second * 2)),
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
        """Return the underlying LangChain rate limiter.

        Returns:
            InMemoryRateLimiter | None: The rate limiter instance or None if not set up.
        """
        if self.rate_limiter is None:
            self.setup()
        return self.rate_limiter


class OpenRouterClient(ChatOpenAI):
    """Client for the OpenRouter API leveraging the OpenAI API.

    This client extends ChatOpenAI to work with OpenRouter's API, providing
    automatic rate limiting based on available credits and proper header management.

    Args:
        api_key (str | None): OpenRouter API key. If None, will try environment variables.
        model (str): Model name to use (default: "meta-llama/llama-3.1-8b-instruct:free").
        temperature (float): Temperature for generation (default: 0.7).
        max_tokens (int | None): Maximum tokens to generate.
        site_url (str | None): URL for HTTP-Referer header (your site's URL).
        site_name (str | None): Name for X-Title header (your app's name).
        streaming (bool): Whether to use streaming responses.
        use_rate_limiter (bool): Whether to enable rate limiting based on credits.
        **kwargs: Additional arguments passed to ChatOpenAI.

    Example:
        >>> client = OpenRouterClient(
        ...     api_key="your-key",
        ...     model="meta-llama/llama-3.1-8b-instruct:free",
        ...     temperature=0.7,
        ...     site_url="https://yourapp.com",
        ...     site_name="Your App Name"
        ... )
        >>> response = client.invoke("Hello, world!")
    """

    # Use WeakValueDictionary to prevent memory leaks
    _instance_data: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
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

        # Create a single rate limiter instance and use it for both purposes
        rate_limiter_instance: OpenRouterRateLimiter | None = None
        if use_rate_limiter:
            rate_limiter_instance = OpenRouterRateLimiter(api_key=api_key)
            rate_limiter_instance.setup()
            kwargs["rate_limiter"] = rate_limiter_instance.get_langchain_limiter()

        # OpenRouter-specific headers as per official documentation
        # These headers help with tracking and rate limiting on OpenRouter's side
        extra_headers = {
            "HTTP-Referer": site_url or "http://localhost:8000",
            "X-Title": site_name or "AutoLabeler",
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

        # Store instance data using a more memory-safe approach
        class InstanceData:
            def __init__(self, use_rate_limiter: bool, rate_limiter: OpenRouterRateLimiter | None):
                self.use_rate_limiter = use_rate_limiter
                self.rate_limiter = rate_limiter

        self._instance_data[id(self)] = InstanceData(use_rate_limiter, rate_limiter_instance)

    def _generate(self, *args: Any, **kwargs: Any) -> LLMResult:
        """Synchronous method for generating completions with rate limiting.

        Args:
            *args: Arguments passed to the parent _generate method.
            **kwargs: Keyword arguments passed to the parent _generate method.

        Returns:
            LLMResult: The result from the parent _generate method.
        """
        instance_data = self._instance_data.get(id(self))

        if instance_data and instance_data.use_rate_limiter and instance_data.rate_limiter:
            instance_data.rate_limiter.acquire()

        return super()._generate(*args, **kwargs)
