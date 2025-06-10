from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LabelResponse(BaseModel):
    """
    Structured output from the LLM for automated labeling.

    Contains the predicted label, confidence score, and optional metadata
    about the labeling decision process.
    """

    label: str = Field(description="The predicted label for the text")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1 for the label prediction"
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation of why this label was chosen"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the labeling decision"
    )
