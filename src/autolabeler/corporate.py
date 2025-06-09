from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI


class CorporateOpenAIClient(ChatOpenAI):
    """Client for internal corporate LLM using a private API URL."""

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
