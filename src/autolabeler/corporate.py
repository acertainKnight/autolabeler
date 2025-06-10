from __future__ import annotations

import os
import ssl
from typing import Any, TypeVar
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from loguru import logger

T = TypeVar("T", bound=BaseModel)


class CorporateOpenAIClient(ChatOpenAI):
    """
    Client for internal corporate LLM using a private API URL.

    This client extends ChatOpenAI to work with corporate/internal LLM deployments
    that use OpenAI-compatible APIs but with custom endpoints and authentication.
    Includes security features for corporate environments.

    Args:
        api_key (str | None): Corporate API key. Falls back to CORPORATE_API_KEY
            environment variable only (not OPENAI_API_KEY for security).
        base_url (str | None): Corporate API base URL. Falls back to
            CORPORATE_BASE_URL environment variable. Must be internal/corporate domain.
        model (str): Model name to use. Defaults to "gpt-3.5-turbo".
        verify_ssl (bool): Whether to verify SSL certificates. Defaults to True.
        extra_headers (dict[str, str] | None): Additional headers for corporate authentication.
        timeout (float): Request timeout in seconds. Defaults to 60.0.
        **kwargs: Additional arguments passed to ChatOpenAI.

    Raises:
        ValueError: If required parameters are missing or invalid.
        SecurityError: If base_url appears to be external rather than corporate.

    Returns:
        CorporateOpenAIClient: Configured client instance.

    Example:
        >>> client = CorporateOpenAIClient(
        ...     api_key="corp-key-123",
        ...     base_url="https://llm.internal.company.com/v1",
        ...     extra_headers={"X-Corporate-ID": "division-a"}
        ... )
        >>> response = client.invoke("Hello, world!")

        >>> # Using structured output (recommended)
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> structured_client = client.with_structured_output(Person)
        >>> result = structured_client.invoke("John is 25 years old")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-3.5-turbo",
        verify_ssl: bool = True,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the Corporate OpenAI client with security validations."""

        # Get API key - only from corporate environment variables for security
        api_key = api_key or os.getenv("CORPORATE_API_KEY")
        if not api_key:
            raise ValueError(
                "Corporate API key not provided. Set CORPORATE_API_KEY environment variable "
                "or pass api_key parameter. For security, OPENAI_API_KEY is not used."
            )

        # Get base URL
        base_url = base_url or os.getenv("CORPORATE_BASE_URL")
        if not base_url:
            raise ValueError(
                "Corporate base URL not provided. Set CORPORATE_BASE_URL environment variable "
                "or pass base_url parameter."
            )

        # Validate base_url for security (should be internal/corporate)
        self._validate_corporate_url(base_url)

        # Set up corporate-specific headers
        corporate_headers = {
            "User-Agent": "CorporateOpenAIClient/1.0",
            "X-Client-Type": "corporate-internal",
        }
        if extra_headers:
            corporate_headers.update(extra_headers)

        # Configure SSL context for corporate environments
        if not verify_ssl:
            logger.warning("SSL verification disabled for corporate client")
            # Note: ChatOpenAI uses httpx internally, so we pass verify parameter
            kwargs["http_client_kwargs"] = kwargs.get("http_client_kwargs", {})
            kwargs["http_client_kwargs"]["verify"] = False

        # Set timeout
        kwargs["timeout"] = timeout

        try:
            super().__init__(
                api_key=api_key,
                base_url=base_url,
                model=model,
                extra_headers=corporate_headers,
                **kwargs
            )

            logger.info(f"Initialized CorporateOpenAIClient for {self._get_domain(base_url)}")

        except Exception as e:
            logger.error(f"Failed to initialize CorporateOpenAIClient: {e}")
            raise

    def _validate_corporate_url(self, base_url: str) -> None:
        """Validate that the base URL appears to be a corporate/internal endpoint.

        Args:
            base_url (str): The base URL to validate.

        Raises:
            ValueError: If the URL format is invalid or appears to be external.
        """
        try:
            parsed = urlparse(base_url)

            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid base_url format: {base_url}")

            # Check for common external LLM providers (security check)
            external_domains = [
                "openai.com",
                "api.openai.com",
                "openrouter.ai",
                "anthropic.com",
                "cohere.ai",
                "huggingface.co",
                "replicate.com",
                "together.ai",
            ]

            domain = parsed.netloc.lower()
            for external_domain in external_domains:
                if external_domain in domain:
                    raise ValueError(
                        f"Base URL appears to be external provider ({external_domain}). "
                        f"Corporate client should only connect to internal endpoints."
                    )

            # Warn if using non-HTTPS (but allow for internal development)
            if parsed.scheme != "https":
                logger.warning(f"Non-HTTPS base_url detected: {base_url}")

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error validating base_url: {e}")

    def _get_domain(self, base_url: str) -> str:
        """Extract domain from base URL for logging."""
        try:
            return urlparse(base_url).netloc
        except Exception:
            return "unknown"

    def with_structured_output(self, schema: type[T], **kwargs: Any) -> Any:
        """
        Create a structured output version of this client.

        This is the modern recommended way to get structured output from LLMs,
        replacing the older PydanticOutputParser pattern.

        Args:
            schema (type[T]): Pydantic model class defining the expected output structure.
            **kwargs: Additional arguments passed to the underlying method.

        Returns:
            Any: Configured client that returns instances of the schema type.

        Raises:
            Exception: If structured output configuration fails.

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> structured_client = client.with_structured_output(Person)
            >>> person = structured_client.invoke("Alice is 30 years old")
            >>> assert isinstance(person, Person)
        """
        try:
            return super().with_structured_output(schema=schema, **kwargs)
        except Exception as e:
            logger.error(f"Failed to configure structured output for schema {schema.__name__}: {e}")
            raise
