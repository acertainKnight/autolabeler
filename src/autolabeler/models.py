from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LabelResponse(BaseModel):
    """
    Structured output from the LLM for automated labeling.

    Contains the predicted label, confidence score, and optional metadata
    about the labeling decision process, including context awareness.
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
    context_influence: str | None = Field(
        default=None,
        description="How the provided context influenced the labeling decision"
    )
    similar_examples_used: list[str] | None = Field(
        default=None,
        description="References to similar examples that influenced the decision"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the labeling decision"
    )
    uncertainty_factors: list[str] | None = Field(
        default=None,
        description="Factors that contributed to uncertainty in the labeling decision"
    )
    alternative_labels: list[dict[str, Any]] | None = Field(
        default=None,
        description="Alternative labels considered with their confidence scores"
    )


class SyntheticExample(BaseModel):
    """
    Structured output from the LLM for synthetic data generation.

    Contains the generated text, target label, and metadata about the generation process.
    """

    text: str = Field(description="The synthetically generated text")
    label: str = Field(description="The target label for this synthetic text")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the quality of the synthetic example"
    )
    reasoning: str | None = Field(
        default=None,
        description="Explanation of the generation strategy used"
    )
    generation_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata about the generation process and source examples"
    )


class SyntheticBatch(BaseModel):
    """
    Structured output for batch synthetic data generation.

    Contains multiple synthetic examples and batch-level metadata.
    """

    examples: list[SyntheticExample] = Field(description="List of generated synthetic examples")
    generation_strategy: str = Field(description="The strategy used for this batch generation")
    source_patterns: list[str] | None = Field(
        default=None,
        description="Patterns or themes identified in source data"
    )
    diversity_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Measure of diversity within the generated batch"
    )
    batch_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the batch generation process"
    )


class MultiFieldLabelResponse(BaseModel):
    """
    Structured output for multi-field extraction tasks.

    Supports extracting multiple pieces of information from a single text
    with individual confidence scores for each field.
    """

    speaker: str | None = Field(
        default=None,
        description="The speaker or source of the headline"
    )
    speaker_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for speaker identification"
    )

    relevance: str | None = Field(
        default=None,
        description="Relevance level to the target question/topic"
    )
    relevance_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for relevance assessment"
    )

    sentiment: str | None = Field(
        default=None,
        description="Sentiment regarding the question/guidelines"
    )
    sentiment_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for sentiment classification"
    )

    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence for the complete extraction"
    )

    reasoning: str | None = Field(
        default=None,
        description="Explanation of the extraction decisions"
    )

    context_influence: str | None = Field(
        default=None,
        description="How context from past headlines influenced the extraction"
    )

    extraction_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the extraction process"
    )
