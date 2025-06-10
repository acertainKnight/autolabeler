from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LabelResponse(BaseModel):
    """Structured output from the LLM."""

    label: str
    metadata: dict[str, Any] | None = None
