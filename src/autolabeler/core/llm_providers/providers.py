"""LLM provider abstractions for unified multi-provider access.

Provides a Protocol-based interface for calling any LLM provider (OpenAI,
Anthropic, Google, OpenRouter) with a consistent API. Includes rate limiting,
cost tracking, and JSON parsing.

Custom providers can be created by subclassing ``BaseLLMProvider`` and
registering them via ``register_provider()``. This allows corporate proxy
systems or any custom endpoint to integrate without modifying this file.

Example:
    >>> from autolabeler.core.llm_providers import BaseLLMProvider, register_provider, LLMResponse
    >>> class MyCorporateProxy(BaseLLMProvider):
    ...     def __init__(self, model: str, **kwargs):
    ...         self.model = model
    ...     async def call(self, system, user, temperature, logprobs=False, response_schema=None):
    ...         # Call your internal endpoint here
    ...         text = my_proxy_call(system, user)
    ...         return LLMResponse(
    ...             text=text,
    ...             parsed_json=self.parse_json_response(text),
    ...             logprobs=None,
    ...             cost=0.0,
    ...             model_name=self.model,
    ...         )
    >>> register_provider("corporate_proxy", MyCorporateProxy)
"""

import asyncio
import importlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from loguru import logger


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Standard response from any LLM provider.

    Attributes:
        text: Raw text response
        parsed_json: Parsed JSON if response is valid JSON, else None
        logprobs: Dictionary of label token logprobs if available, else None
        cost: Cost in USD for this call
        model_name: Model identifier used
        raw_response: Original provider response object for debugging
    """
    text: str
    parsed_json: dict[str, Any] | None
    logprobs: dict[str, float] | None
    cost: float
    model_name: str
    raw_response: Any = None


class LLMProvider(Protocol):
    """Unified interface for any LLM provider.

    All providers (Anthropic, OpenAI, Google, OpenRouter) implement this protocol.
    """

    async def call(
        self,
        system: str,
        user: str,
        temperature: float,
        logprobs: bool = False,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call the LLM with a system and user prompt.

        Parameters:
            system: System prompt defining role and context
            user: User prompt with task and input
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to request token logprobs (only OpenAI supports this)
            response_schema: Optional JSON schema to constrain output format

        Returns:
            LLMResponse with text, parsed JSON, optional logprobs, and cost
        """
        ...


class BaseLLMProvider(ABC):
    """Abstract base class for custom LLM provider integrations.

    Subclass this to integrate a corporate proxy, internal API, or any
    non-standard LLM endpoint. Register your subclass with
    ``register_provider()`` so the pipeline can instantiate it by name.

    The constructor must accept ``model`` as its first positional argument.
    Any additional keyword arguments are passed through from ``get_provider()``.

    Attributes:
        model: Model identifier used by this provider instance.

    Example:
        >>> class MyCorporateProxy(BaseLLMProvider):
        ...     def __init__(self, model: str, base_url: str = "https://proxy.example.com"):
        ...         self.model = model
        ...         self.base_url = base_url
        ...
        ...     async def call(self, system, user, temperature, logprobs=False, response_schema=None):
        ...         text = await my_internal_call(self.base_url, system, user)
        ...         return LLMResponse(
        ...             text=text,
        ...             parsed_json=self.parse_json_response(text),
        ...             logprobs=None,
        ...             cost=0.0,
        ...             model_name=self.model,
        ...         )
        ...
        >>> register_provider("corporate_proxy", MyCorporateProxy)
    """

    @abstractmethod
    async def call(
        self,
        system: str,
        user: str,
        temperature: float,
        logprobs: bool = False,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call the LLM with a system and user prompt.

        Parameters:
            system: System prompt defining role and context.
            user: User prompt with task and input.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            logprobs: Whether to request token logprobs (provider-dependent).
            response_schema: Optional JSON schema to constrain output format.

        Returns:
            LLMResponse with text, parsed JSON, optional logprobs, and cost.
        """

    @staticmethod
    def parse_json_response(text: str) -> dict[str, Any] | None:
        """Parse JSON from an LLM response string.

        Handles markdown-fenced code blocks and embedded JSON objects.
        A convenience wrapper around the module-level ``_parse_json`` helper.

        Parameters:
            text: Raw LLM output text.

        Returns:
            Parsed dict if successful, else None.

        Example:
            >>> BaseLLMProvider.parse_json_response('```json\\n{"label": "1"}\\n```')
            {'label': '1'}
        """
        return _parse_json(text)


# ---------------------------------------------------------------------------
# Cost tracker -- lightweight, per-provider running total
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    """Tracks cumulative API costs and enforces a budget ceiling.

    Attributes:
        budget: Maximum allowed spend in USD (0 = unlimited)
        total_cost: Running total of costs accumulated so far
        call_count: Number of API calls made
    """
    budget: float = 0.0
    total_cost: float = 0.0
    call_count: int = 0

    def add(self, cost: float) -> None:
        """Record a cost increment.

        Parameters:
            cost: Cost in USD for the latest call.
        """
        self.total_cost += cost
        self.call_count += 1

    @property
    def budget_exceeded(self) -> bool:
        """Whether the budget has been exceeded."""
        return self.budget > 0 and self.total_cost >= self.budget

    @property
    def remaining(self) -> float:
        """Remaining budget in USD (inf if no budget set)."""
        if self.budget <= 0:
            return float("inf")
        return max(0.0, self.budget - self.total_cost)


# ---------------------------------------------------------------------------
# Rate limiter -- async token-bucket shared across providers
# ---------------------------------------------------------------------------

class AsyncRateLimiter:
    """Simple async token-bucket rate limiter.

    Parameters:
        requests_per_second: Target throughput ceiling.
        max_concurrent: Maximum number of in-flight requests.
    """

    def __init__(
        self,
        requests_per_second: float = 50.0,
        max_concurrent: int = 50,
    ) -> None:
        self._rps = requests_per_second
        self._interval = 1.0 / requests_per_second
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    def release(self) -> None:
        """Release a request slot."""
        self._semaphore.release()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict[str, Any] | None:
    """Try to parse JSON from an LLM response string.

    Handles markdown-fenced code blocks and embedded JSON objects.

    Parameters:
        text: Raw LLM output text.

    Returns:
        Parsed dict if successful, else None.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return None


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """OpenAI provider with logprobs support.

    Supports: gpt-4o, gpt-4o-mini, gpt-3.5-turbo

    Example:
        >>> provider = OpenAIProvider(model="gpt-4o")
        >>> response = await provider.call(system="...", user="...", temperature=0.1, logprobs=True)
        >>> print(response.logprobs)  # {"0": 0.95, "1": 0.03, "-1": 0.02}
    """

    # Approximate per-1k-token pricing (update as OpenAI changes)
    _PRICING: dict[str, tuple[float, float]] = {
        "gpt-4o":      (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
    }
    _DEFAULT_PRICING = (0.005, 0.015)

    def __init__(self, model: str, api_key: str | None = None):
        """Initialize OpenAI provider.

        Parameters:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: Optional API key (reads from env if not provided)
        """
        import openai

        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def call(
        self,
        system: str,
        user: str,
        temperature: float,
        logprobs: bool = False,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call OpenAI API with optional structured output."""
        try:
            params: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": 2048,
                "logprobs": logprobs,
                "top_logprobs": 10 if logprobs else None,
            }

            # Add structured output if schema provided
            if response_schema:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "labeling_response",
                        "strict": True,
                        "schema": response_schema,
                    }
                }

            response = await self.client.chat.completions.create(**params)

            text = response.choices[0].message.content or ""
            parsed = _parse_json(text)

            logprob_dict = None
            if logprobs and response.choices[0].logprobs:
                logprob_dict = self._extract_label_logprobs(response.choices[0].logprobs)

            cost = self._calculate_cost(response.usage)

            return LLMResponse(
                text=text,
                parsed_json=parsed,
                logprobs=logprob_dict,
                cost=cost,
                model_name=self.model,
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            raise

    def _extract_label_logprobs(self, logprobs_obj: Any) -> dict[str, float]:
        """Extract label token logprobs from OpenAI response."""
        import math

        label_probs: dict[str, float] = {}
        for token_info in logprobs_obj.content:
            if token_info.top_logprobs:
                for lp in token_info.top_logprobs:
                    token_clean = lp.token.strip().strip('"\'')
                    if token_clean.lstrip('-').isdigit():
                        prob = math.exp(lp.logprob)
                        if token_clean not in label_probs or prob > label_probs[token_clean]:
                            label_probs[token_clean] = prob

        return label_probs

    def _calculate_cost(self, usage: Any) -> float:
        """Calculate cost based on token usage."""
        prompt_k, completion_k = self._PRICING.get(self.model, self._DEFAULT_PRICING)
        prompt_cost = (usage.prompt_tokens / 1000) * prompt_k
        completion_cost = (usage.completion_tokens / 1000) * completion_k
        return prompt_cost + completion_cost


class AnthropicProvider:
    """Anthropic provider.

    Supports: claude-sonnet-4-5, claude-opus-4, claude-haiku-4

    Note: Does not support logprobs. Use self-consistency sampling for confidence.
    """

    _PRICING: dict[str, tuple[float, float]] = {
        "claude-sonnet-4-5-20250929": (0.003, 0.015),
        "claude-opus-4-20250514":     (0.015, 0.075),
        "claude-haiku-4-20250514":    (0.0008, 0.004),
    }
    _DEFAULT_PRICING = (0.003, 0.015)

    def __init__(self, model: str, api_key: str | None = None):
        """Initialize Anthropic provider.

        Parameters:
            model: Model name (e.g., "claude-sonnet-4-5-20250929")
            api_key: Optional API key (reads from env if not provided)
        """
        import anthropic

        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def call(
        self,
        system: str,
        user: str,
        temperature: float,
        logprobs: bool = False,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call Anthropic API with optional structured output via enhanced prompt."""
        if logprobs:
            logger.warning("Anthropic does not support logprobs. Ignoring logprobs parameter.")

        # Enhance system prompt with schema if provided
        if response_schema:
            schema_str = json.dumps(response_schema, indent=2)
            system = f"{system}\n\nYou MUST respond with valid JSON matching this exact schema:\n{schema_str}\n\nDo not include any text outside the JSON object."

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )

            text = response.content[0].text
            parsed = _parse_json(text)
            cost = self._calculate_cost(response.usage)

            return LLMResponse(
                text=text,
                parsed_json=parsed,
                logprobs=None,
                cost=cost,
                model_name=self.model,
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            raise

    def _calculate_cost(self, usage: Any) -> float:
        """Calculate cost based on token usage."""
        prompt_k, completion_k = self._PRICING.get(self.model, self._DEFAULT_PRICING)
        return (usage.input_tokens / 1000) * prompt_k + (usage.output_tokens / 1000) * completion_k


class GoogleProvider:
    """Google Generative AI provider.

    Supports: gemini-2.5-pro, gemini-2.5-flash

    Note: Does not support logprobs. Use self-consistency sampling for confidence.
    """

    # Per-1k-token pricing (input, output) -- Google AI Studio / Vertex rates
    # Source: https://ai.google.dev/pricing (as of 2026-02)
    _PRICING: dict[str, tuple[float, float]] = {
        "gemini-2.5-pro":   (0.00125, 0.005),
        "gemini-2.5-flash": (0.000075, 0.0003),
        "gemini-2.0-flash": (0.0001, 0.0004),
        "gemini-1.5-pro":   (0.00125, 0.005),
        "gemini-1.5-flash": (0.000075, 0.0003),
    }
    _DEFAULT_PRICING = (0.00025, 0.001)

    def __init__(self, model: str, api_key: str | None = None):
        """Initialize Google provider.

        Parameters:
            model: Model name (e.g., "gemini-2.5-pro")
            api_key: Optional API key (reads from env if not provided)
        """
        import google.generativeai as genai

        self.model = model
        if api_key:
            genai.configure(api_key=api_key)
        elif 'GOOGLE_API_KEY' in os.environ:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

        self.client = genai.GenerativeModel(model)

    async def call(
        self,
        system: str,
        user: str,
        temperature: float,
        logprobs: bool = False,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call Google Generative AI API with optional structured output."""
        if logprobs:
            logger.warning("Google does not support logprobs. Ignoring logprobs parameter.")

        try:
            full_prompt = f"{system}\n\n{user}"
            
            generation_config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": 2048,
            }
            
            # Add structured output configuration if schema provided
            if response_schema:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_schema"] = response_schema
            
            response = await self.client.generate_content_async(
                full_prompt,
                generation_config=generation_config,
            )

            text = response.text
            parsed = _parse_json(text)
            cost = self._calculate_cost(response)

            return LLMResponse(
                text=text,
                parsed_json=parsed,
                logprobs=None,
                cost=cost,
                model_name=self.model,
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"Google call failed: {e}")
            raise

    def _calculate_cost(self, response: Any) -> float:
        """Calculate cost from token usage metadata on the Google response.

        Parameters:
            response: GenerateContentResponse from the Google SDK.

        Returns:
            Estimated cost in USD.
        """
        prompt_k, completion_k = self._PRICING.get(self.model, self._DEFAULT_PRICING)

        try:
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
            completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
        except (AttributeError, TypeError):
            return 0.0

        return (prompt_tokens / 1000) * prompt_k + (completion_tokens / 1000) * completion_k


class OpenRouterProvider:
    """OpenRouter provider with rate limiting and cost tracking.

    OpenRouter is a unified API that provides access to multiple LLM providers
    through an OpenAI-compatible interface. This provider handles:

    - **Rate limiting**: Async token-bucket to stay within OpenRouter's limits
    - **Cost tracking**: Extracts cost from usage data returned per request
    - **Credit monitoring**: Optionally queries /api/v1/credits on init
    - **Budget enforcement**: Raises if cumulative spend exceeds the configured budget

    Parameters:
        model: Model identifier (e.g., "anthropic/claude-sonnet-4")
        api_key: Optional API key (reads from env if not provided)
        rate_limit: Requests per second ceiling (default 50)
        max_concurrent: Max in-flight requests (default 50)
        budget: Max USD spend -- 0 means unlimited (default 0)
        check_credits: Whether to query credit balance on init (default False)

    Example:
        >>> provider = OpenRouterProvider("google/gemini-2.5-pro", budget=5.0)
        >>> response = await provider.call(system="...", user="...", temperature=0.1)
        >>> print(provider.cost_tracker.total_cost)
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        rate_limit: float = 50.0,
        max_concurrent: int = 50,
        budget: float = 0.0,
        check_credits: bool = False,
    ) -> None:
        self.model = model
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("API_OPENROUTER_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key."
            )

        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.AsyncClient(timeout=120.0)

        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(
            requests_per_second=rate_limit,
            max_concurrent=max_concurrent,
        )

        # Cost tracking
        self.cost_tracker = CostTracker(budget=budget)

        # Optionally check credit balance at startup
        if check_credits:
            self._log_credit_balance()

    # -- Public API ----------------------------------------------------------

    async def call(
        self,
        system: str,
        user: str,
        temperature: float,
        logprobs: bool = False,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call OpenRouter API with rate limiting and cost tracking.

        Parameters:
            system: System prompt.
            user: User prompt.
            temperature: Sampling temperature.
            logprobs: Ignored (OpenRouter doesn't expose logprobs).
            response_schema: Optional JSON schema to constrain output.

        Returns:
            LLMResponse with text, parsed JSON, and cost.

        Raises:
            RuntimeError: If the budget has been exceeded.
            httpx.HTTPStatusError: On HTTP errors from OpenRouter.
        """
        if logprobs:
            logger.debug("OpenRouter does not expose logprobs; parameter ignored.")

        if self.cost_tracker.budget_exceeded:
            raise RuntimeError(
                f"Budget exceeded: ${self.cost_tracker.total_cost:.4f} "
                f"spent of ${self.cost_tracker.budget:.2f} budget"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/autolabeler",
            "X-Title": "AutoLabeler",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            # Ask OpenRouter to include usage data for cost calculation
            "usage": {"include": True},
        }
        
        # Add structured output if schema provided (OpenRouter forwards to upstream)
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "labeling_response",
                    "strict": True,
                    "schema": response_schema,
                }
            }

        await self.rate_limiter.acquire()
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            text = data["choices"][0]["message"]["content"]
            parsed = _parse_json(text)
            cost = self._extract_cost(data)

            self.cost_tracker.add(cost)

            return LLMResponse(
                text=text,
                parsed_json=parsed,
                logprobs=None,
                cost=cost,
                model_name=self.model,
                raw_response=data,
            )
        except Exception as e:
            logger.error(f"OpenRouter call failed ({self.model}): {e}")
            raise
        finally:
            self.rate_limiter.release()

    # -- Cost helpers --------------------------------------------------------

    @staticmethod
    def _extract_cost(data: dict[str, Any]) -> float:
        """Extract cost from OpenRouter response.

        OpenRouter may include cost directly in the usage object
        or in x-openrouter headers. Falls back to 0.0 if unavailable.

        Parameters:
            data: Parsed JSON response from OpenRouter.

        Returns:
            Cost in USD for this single call.
        """
        usage = data.get("usage", {})

        # OpenRouter sometimes includes cost directly
        if "cost" in usage:
            try:
                return float(usage["cost"])
            except (ValueError, TypeError):
                pass

        # Alternatively in a top-level field
        if "cost" in data:
            try:
                return float(data["cost"])
            except (ValueError, TypeError):
                pass

        # Estimate from native_tokens_* if OpenRouter provides them
        # (native_tokens_prompt, native_tokens_completion)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        if prompt_tokens or completion_tokens:
            # Very rough fallback: $1/1M prompt, $3/1M completion (cheap model average)
            return (prompt_tokens * 1e-6) + (completion_tokens * 3e-6)

        return 0.0

    def _log_credit_balance(self) -> None:
        """Query OpenRouter credit balance and log it (synchronous, init-only)."""
        import requests as sync_requests

        try:
            resp = sync_requests.get(
                f"{self.base_url}/credits",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                info = resp.json().get("data", {})
                total = info.get("total_credits", "?")
                used = info.get("total_usage", "?")
                logger.info(f"OpenRouter credits: ${total} total, ${used} used")
            else:
                logger.debug(f"Could not fetch credit balance: HTTP {resp.status_code}")
        except Exception as e:
            logger.debug(f"Credit check skipped: {e}")

    # -- Context manager -----------------------------------------------------

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()


# ---------------------------------------------------------------------------
# Custom provider registry
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {}

_BUILTIN_PROVIDERS: dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "openrouter": OpenRouterProvider,
}


def register_provider(name: str, provider_class: type[BaseLLMProvider]) -> None:
    """Register a custom LLM provider class by name.

    Once registered, the provider can be referenced in YAML config files via
    ``provider: <name>`` and the pipeline will instantiate it automatically.

    Parameters:
        name: Unique string identifier for this provider (e.g., "corporate_proxy").
            Must not clash with built-in names ("openai", "anthropic", "google",
            "openrouter").
        provider_class: A subclass of ``BaseLLMProvider``.

    Raises:
        TypeError: If ``provider_class`` is not a subclass of ``BaseLLMProvider``.
        ValueError: If ``name`` conflicts with a built-in provider name.

    Example:
        >>> register_provider("corporate_proxy", MyCorporateProxy)
    """
    if not (isinstance(provider_class, type) and issubclass(provider_class, BaseLLMProvider)):
        raise TypeError(
            f"provider_class must be a subclass of BaseLLMProvider, got {provider_class!r}"
        )
    if name in _BUILTIN_PROVIDERS:
        raise ValueError(
            f"Cannot register '{name}': conflicts with built-in provider. "
            f"Built-in names: {list(_BUILTIN_PROVIDERS.keys())}"
        )
    if name in _PROVIDER_REGISTRY:
        logger.warning(f"Overwriting existing custom provider registration for '{name}'")
    _PROVIDER_REGISTRY[name] = provider_class
    logger.info(f"Registered custom LLM provider: '{name}' -> {provider_class.__name__}")


def get_registered_providers() -> dict[str, type[BaseLLMProvider]]:
    """Return a copy of the current custom provider registry.

    Returns:
        Dict mapping provider name to provider class for all registered custom providers.

    Example:
        >>> get_registered_providers()
        {'corporate_proxy': <class 'my_company.llm.MyCorporateProxy'>}
    """
    return dict(_PROVIDER_REGISTRY)


def load_provider_module(dotted_path: str) -> None:
    """Import a Python module by dotted path to trigger provider registration.

    The module is expected to call ``register_provider()`` at import time.
    This is the mechanism used by the ``--provider-module`` CLI flag.

    Parameters:
        dotted_path: Fully-qualified module path (e.g., ``"my_company.llm_proxy"``).

    Raises:
        ImportError: If the module cannot be found or imported.

    Example:
        >>> load_provider_module("my_company.llm_proxy")
    """
    try:
        importlib.import_module(dotted_path)
        logger.info(f"Loaded provider module: {dotted_path}")
    except ImportError as e:
        raise ImportError(
            f"Could not import provider module '{dotted_path}': {e}. "
            "Ensure the module is on sys.path and calls register_provider() at import time."
        ) from e


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(provider_name: str, model: str, api_key: str | None = None) -> LLMProvider:
    """Factory function to create provider instances.

    Checks the custom registry first, then falls back to built-in providers.

    Parameters:
        provider_name: Provider name. Can be a custom-registered name or one of
            the built-ins: "anthropic", "openai", "google", "openrouter".
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-5-20250929").
        api_key: Optional API key (reads from env if not provided). Passed as
            ``api_key`` kwarg to built-in providers; custom providers receive it
            only if their constructor accepts it.

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If ``provider_name`` is not registered and not a built-in.

    Example:
        >>> provider = get_provider("openai", "gpt-4o")
        >>> response = await provider.call(system="...", user="...", temperature=0.1)
    """
    # Custom registry takes priority so users can shadow built-ins if needed
    if provider_name in _PROVIDER_REGISTRY:
        custom_class = _PROVIDER_REGISTRY[provider_name]
        logger.debug(f"Using custom provider '{provider_name}': {custom_class.__name__}")
        return custom_class(model=model)

    if provider_name not in _BUILTIN_PROVIDERS:
        all_known = sorted(list(_BUILTIN_PROVIDERS.keys()) + list(_PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Available: {all_known}. "
            "To add a custom provider, subclass BaseLLMProvider and call register_provider()."
        )

    return _BUILTIN_PROVIDERS[provider_name](model=model, api_key=api_key)
