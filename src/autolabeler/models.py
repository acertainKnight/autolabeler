from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class LabelResponse(BaseModel):
    """
    Structured output from the LLM for automated labeling.

    Contains the predicted label, confidence score, and optional metadata
    about the labeling decision process, including context awareness.
    """

    model_config = ConfigDict(extra="forbid")

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
    similar_examples_used: list[str]| str | None = Field(
        default=None,
        description="References to similar examples that influenced the decision"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the labeling decision as string key-value pairs"
    )
    uncertainty_factors: list[str] = Field(
        default_factory=list,
        description="Factors that contributed to uncertainty in the labeling decision"
    )
    alternative_labels: list[str] = Field(
        default_factory=list,
        description="Alternative labels considered during the decision process"
    )


class SyntheticExample(BaseModel):
    """
    Structured output from the LLM for synthetic data generation.

    Contains the generated text, target label, and metadata about the generation process.
    """

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    extraction_metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata about the extraction process as string key-value pairs"
    )


class LabelingRule(BaseModel):
    """
    Represents a single labeling rule derived from training data.

    Contains patterns, conditions, and examples that define when to apply specific labels.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(description="Unique identifier for this rule")
    label: str = Field(description="The label this rule applies to")
    pattern_description: str = Field(description="Human-readable description of the pattern")
    conditions: list[str] = Field(description="Specific conditions that trigger this rule")
    indicators: list[str] = Field(description="Key words, phrases, or patterns that indicate this label")
    examples: list[str] = Field(description="Representative examples that demonstrate this rule")
    counter_examples: list[str] | None = Field(
        default=None,
        description="Examples that might seem to fit but don't apply to this rule"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the reliability of this rule"
    )
    frequency: int = Field(
        default=0,
        description="Number of training examples that support this rule"
    )
    edge_cases: list[str] | None = Field(
        default=None,
        description="Known edge cases or exceptions to this rule"
    )
    creation_timestamp: str | None = Field(default=None, description="When this rule was created")
    last_updated: str | None = Field(default=None, description="When this rule was last updated")
    source_data_hash: str | None = Field(
        default=None,
        description="Hash of source data used to create this rule"
    )


class RuleSet(BaseModel):
    """
    Collection of labeling rules for a specific dataset and task.

    Represents the complete annotation guidelines derived from training data.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(description="Name of the dataset this ruleset applies to")
    task_description: str = Field(description="Description of the labeling task")
    label_categories: list[str] = Field(description="All possible labels in this task")
    rules: list[LabelingRule] = Field(description="Individual labeling rules")
    general_guidelines: list[str] = Field(description="General annotation principles")
    disambiguation_rules: list[str] | None = Field(
        default=None,
        description="Rules for handling ambiguous cases"
    )
    quality_checks: list[str] | None = Field(
        default=None,
        description="Quality assurance guidelines for annotators"
    )
    version: str = Field(description="Version of this ruleset")
    creation_timestamp: str = Field(description="When this ruleset was created")
    last_updated: str = Field(description="When this ruleset was last updated")
    statistics: dict[str, Any] | None = Field(
        default=None,
        description="Statistics about rule coverage and performance"
    )


class RuleGenerationResult(BaseModel):
    """
    Result of the rule generation process.

    Contains the generated ruleset along with metadata about the generation process.
    """

    model_config = ConfigDict(extra="forbid")

    ruleset: RuleSet = Field(description="The generated labeling ruleset")
    generation_metadata: dict[str, Any] | None = Field(default=None, description="Metadata about the rule generation process")
    data_analysis: dict[str, Any] | None = Field(default=None, description="Analysis of the input data used for rule generation")
    rule_conflicts: list[str] | None = Field(
        default=None,
        description="Identified conflicts or inconsistencies between rules"
    )
    coverage_analysis: dict[str, Any] | None = Field(
        default=None,
        description="Analysis of how well the rules cover the training data"
    )
    recommendations: list[str] | None = Field(
        default=None,
        description="Recommendations for improving the ruleset"
    )


class RuleUpdateResult(BaseModel):
    """
    Result of updating an existing ruleset with new data.

    Contains information about what changed and the updated ruleset.
    """

    model_config = ConfigDict(extra="forbid")

    updated_ruleset: RuleSet = Field(description="The updated labeling ruleset")
    changes_made: list[str] = Field(description="List of changes made to the ruleset")
    new_rules_added: int = Field(default=0, description="Number of new rules added")
    rules_modified: int = Field(default=0, description="Number of existing rules modified")
    rules_removed: int = Field(default=0, description="Number of rules removed")
    confidence_changes: dict[str, float] | None = Field(
        default=None,
        description="Changes in rule confidence scores"
    )
    update_metadata: dict[str, Any] = Field(description="Metadata about the update process")
    validation_results: dict[str, Any] | None = Field(
        default=None,
        description="Validation results for the updated ruleset"
    )
