from __future__ import annotations

import os
from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class CorporateOpenAIClient(ChatOpenAI):
    """
    Client for internal corporate LLM using a private API URL.

    This client extends ChatOpenAI to work with corporate/internal LLM deployments
    that use OpenAI-compatible APIs but with custom endpoints and authentication.

    Args:
        api_key (str | None): Corporate API key. Falls back to CORPORATE_API_KEY
            or OPENAI_API_KEY environment variables.
        base_url (str | None): Corporate API base URL. Falls back to
            CORPORATE_BASE_URL environment variable.
        model (str): Model name to use. Defaults to "gpt-3.5-turbo".
        **kwargs: Additional arguments passed to ChatOpenAI.

    Returns:
        CorporateOpenAIClient: Configured client instance.

    Example:
        >>> client = CorporateOpenAIClient(
        ...     api_key="your-key",
        ...     base_url="https://corporate-llm.company.com/v1"
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
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.getenv("CORPORATE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Corporate API key not provided. Set CORPORATE_API_KEY or OPENAI_API_KEY."
            )
        base_url = base_url or os.getenv("CORPORATE_BASE_URL")
        if not base_url:
            raise ValueError(
                "Corporate base URL not provided. Set CORPORATE_BASE_URL or pass base_url."
            )
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

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

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> structured_client = client.with_structured_output(Person)
            >>> person = structured_client.invoke("Alice is 30 years old")
            >>> assert isinstance(person, Person)
        """
        return super().with_structured_output(schema=schema, **kwargs)
