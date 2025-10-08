"""Configuration classes for AutoLabeler components."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LabelingConfig(BaseModel):
    """Configuration for labeling operations."""

    use_rag: bool = Field(True, description="Whether to use RAG for example retrieval")
    k_examples: int | None = Field(None, description="Number of examples to retrieve")
    prefer_human_examples: bool = Field(True, description="Prefer human-labeled examples")
    confidence_threshold: float = Field(0.0, description="Minimum confidence for knowledge base updates")
    save_to_knowledge_base: bool = Field(True, description="Save high-confidence predictions")
    metadata_columns: list[str] | None = Field(None, description="Additional metadata columns to use")
    model_name: str | None = Field(None, description="Model name/identifier")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int | None = Field(None, description="Maximum tokens in response")
    max_examples: int = Field(5, description="Maximum examples to retrieve")
    use_validation: bool = Field(True, description="Use structured output validation with retry")
    validation_max_retries: int = Field(3, description="Maximum validation retry attempts")
    allowed_labels: list[str] | None = Field(None, description="Allowed label values for validation")


class BatchConfig(BaseModel):
    """Configuration for batch processing."""

    batch_size: int = Field(50, description="Size of each batch")
    max_concurrency: int | None = Field(None, description="Maximum concurrent operations")
    save_interval: int | None = Field(10, description="Save progress every N items/batches")
    resume: bool = Field(True, description="Resume from existing progress if available")


class GenerationConfig(BaseModel):
    """Configuration for synthetic data generation."""

    strategy: str = Field("mixed", description="Generation strategy")
    diversity_focus: str = Field("high", description="Level of diversity emphasis")
    num_examples: int = Field(5, description="Number of examples to generate")
    confidence_threshold: float = Field(0.7, description="Minimum confidence for generated examples")
    add_to_knowledge_base: bool = Field(True, description="Add generated examples to knowledge base")
    batch_constraints: list[str] | None = Field(None, description="Additional constraints for batch generation")
    model_name: str | None = Field(None, description="Model name/identifier for generation")
    temperature: float = Field(0.7, description="Sampling temperature for generation")
    max_tokens: int | None = Field(None, description="Maximum tokens in response for generation")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation operations."""

    test_size: float = Field(0.2, description="Fraction of data for testing")
    stratify_column: str | None = Field(None, description="Column to stratify split on")
    save_results: bool = Field(True, description="Save evaluation results")
    create_report: bool = Field(True, description="Create human-readable report")
    confidence_analysis: bool = Field(True, description="Analyze confidence vs performance")


class RuleGenerationConfig(BaseModel):
    """Configuration for rule generation."""

    batch_size: int = Field(50, description="Examples per batch for rule generation")
    min_examples_per_rule: int = Field(3, description="Minimum examples needed for a rule")
    task_description: str | None = Field(None, description="Description of the labeling task")
    export_format: str = Field("markdown", description="Format for human-readable export")
    consolidate_similar: bool = Field(True, description="Merge similar rules")


class EnsembleConfig(BaseModel):
    """Configuration for ensemble labeling."""

    method: str = Field("majority_vote", description="Ensemble consolidation method")
    min_confidence: float = Field(0.0, description="Minimum confidence threshold for predictions")
    agreement_threshold: float = Field(0.7, description="Minimum agreement for high_agreement method")
    max_models: int | None = Field(None, description="Maximum models to consider")
    weight_by_performance: bool = Field(False, description="Weight models by past performance")


class PromptConfig(BaseModel):
    """Configuration for prompt generation and tracking."""

    template_path: str | None = Field(None, description="Path to the prompt template")
    track_prompts: bool = Field(True, description="Whether to track prompts")
    include_examples: bool = Field(True, description="Whether to include examples in prompts")
    max_prompt_length: int | None = Field(None, description="Maximum length of the prompt")
    temperature: float = Field(0.1, description="Sampling temperature for the LLM")
    tags: list[str] = Field(default_factory=list, description="Tags for prompt management")


class KnowledgeBaseConfig(BaseModel):
    """Configuration for knowledge base operations."""

    embedding_model: str = Field("text-embedding-3-small", description="Embedding model for the knowledge base")
    similarity_threshold: float = Field(0.8, description="Similarity threshold for retrieving documents")
    chunk_size: int = Field(1000, description="Chunk size for splitting documents")
    chunk_overlap: int = Field(200, description="Chunk overlap for splitting documents")
    filter_source: str | None = Field(None, description="Source to filter documents from")
    include_metadata: bool = Field(True, description="Whether to include metadata in the knowledge base")


class DataSplitConfig(BaseModel):
    """Configuration for train/test splits."""

    test_size: float = Field(0.2, description="Proportion of the dataset to include in the test split")
    validation_size: float | None = Field(None, description="Proportion of the dataset to include in the validation split")
    stratify_column: str | None = Field(None, description="Column to use for stratified sampling")
    random_state: int = Field(42, description="Seed for the random number generator")
    exclude_from_training: list[Any] = Field(
        default_factory=list, description="Items to exclude from the training set"
    )


class ComponentConfig(BaseModel):
    """Base configuration for all components."""

    dataset_name: str = Field(..., description="Name of the dataset")
    component_name: str = Field(..., description="Name of the component")
    storage_path: str | None = Field(None, description="Path for storing component data")
    log_level: str = Field("INFO", description="Logging level")

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow additional fields
