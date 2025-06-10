from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """
    Configuration for a specific model setup in ensemble labeling.

    Tracks model identity, parameters, and metadata for reproducible
    multi-model experiments and ensemble consolidation.
    """

    # Model Identity
    model_id: str = Field(default="", description="Unique identifier for this model configuration")
    model_name: str = Field(description="Base model name (e.g., 'gpt-3.5-turbo')")
    provider: str = Field(description="Model provider ('openrouter', 'corporate', etc.)")

    # Model Parameters
    temperature: float = Field(default=0.1, description="Sampling temperature")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter")

    # RAG Configuration
    use_rag: bool = Field(default=True, description="Whether to use RAG examples")
    max_examples: int = Field(default=5, description="Maximum RAG examples to use")
    prefer_human_examples: bool = Field(default=True, description="Prefer human over model examples")
    confidence_threshold: float = Field(default=0.0, description="Minimum confidence for KB updates")

    # Metadata
    description: str | None = Field(default=None, description="Human-readable description")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

    @field_validator('model_id')
    @classmethod
    def generate_model_id_if_empty(cls, v: str, info) -> str:
        """Generate model_id if not provided."""
        if not v:
            # Create a temporary instance to call generate_id
            temp_values = info.data.copy()
            temp_values['model_id'] = 'temp'  # Avoid recursion
            temp_config = cls.model_construct(**temp_values)
            return temp_config.generate_id()
        return v

    def generate_id(self) -> str:
        """
        Generate a unique ID based on model configuration.

        Returns:
            str: Unique model configuration ID.
        """
        config_dict = {
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "use_rag": self.use_rag,
            "max_examples": self.max_examples,
            "prefer_human_examples": self.prefer_human_examples,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage and tracking."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from dictionary."""
        return cls(**data)

    def __str__(self) -> str:
        return f"ModelConfig({self.model_id}: {self.model_name} T={self.temperature})"


class EnsembleMethod(BaseModel):
    """
    Configuration for ensemble consolidation methods.

    Defines how multiple model predictions should be combined
    into a single final prediction.
    """

    method_name: str = Field(description="Name of the ensemble method")
    description: str = Field(description="Description of how the method works")

    # Method-specific parameters
    weight_by_confidence: bool = Field(default=True, description="Weight predictions by confidence")
    min_agreement: float = Field(default=0.5, description="Minimum agreement threshold")
    require_human_baseline: bool = Field(default=False, description="Require human examples for validation")

    # Quality filters
    min_confidence_threshold: float = Field(default=0.3, description="Minimum confidence to include prediction")
    max_models_to_consider: int | None = Field(default=None, description="Maximum models to consider")

    @classmethod
    def majority_vote(cls) -> EnsembleMethod:
        """Simple majority voting ensemble method."""
        return cls(
            method_name="majority_vote",
            description="Select the label with the most votes across models",
            weight_by_confidence=False,
            min_agreement=0.5
        )

    @classmethod
    def confidence_weighted(cls) -> EnsembleMethod:
        """Confidence-weighted ensemble method."""
        return cls(
            method_name="confidence_weighted",
            description="Weight each prediction by its confidence score",
            weight_by_confidence=True,
            min_agreement=0.3
        )

    @classmethod
    def high_agreement(cls) -> EnsembleMethod:
        """High agreement ensemble method."""
        return cls(
            method_name="high_agreement",
            description="Only predictions with high model agreement",
            weight_by_confidence=True,
            min_agreement=0.7,
            min_confidence_threshold=0.5
        )

    @classmethod
    def human_validated(cls) -> EnsembleMethod:
        """Human-validated ensemble method."""
        return cls(
            method_name="human_validated",
            description="Ensemble with human baseline validation",
            weight_by_confidence=True,
            require_human_baseline=True,
            min_confidence_threshold=0.6
        )


class ModelRun(BaseModel):
    """
    Records of a specific model run on a dataset.

    Tracks the execution of a model configuration on a dataset,
    including timing, results, and performance metrics.
    """

    run_id: str = Field(description="Unique identifier for this run")
    model_config_id: str = Field(description="ID of the model configuration used")
    dataset_name: str = Field(description="Name of the dataset labeled")

    # Execution details
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = Field(default=None)
    status: str = Field(default="running", description="Status: running, completed, failed")

    # Results summary
    total_texts: int = Field(default=0, description="Total number of texts processed")
    successful_predictions: int = Field(default=0, description="Number of successful predictions")
    failed_predictions: int = Field(default=0, description="Number of failed predictions")
    avg_confidence: float | None = Field(default=None, description="Average confidence score")

    # Performance metrics
    processing_time_seconds: float | None = Field(default=None)
    predictions_per_second: float | None = Field(default=None)

    # Error tracking
    error_messages: list[str] = Field(default_factory=list, description="List of error messages")

    def mark_completed(self) -> None:
        """Mark the run as completed and calculate final metrics."""
        self.completed_at = datetime.now().isoformat()
        self.status = "completed"

        if self.started_at and self.completed_at:
            start_time = datetime.fromisoformat(self.started_at)
            end_time = datetime.fromisoformat(self.completed_at)
            self.processing_time_seconds = (end_time - start_time).total_seconds()

            if self.processing_time_seconds > 0 and self.successful_predictions > 0:
                self.predictions_per_second = self.successful_predictions / self.processing_time_seconds

    def add_error(self, error_message: str) -> None:
        """Add an error message to the run."""
        self.error_messages.append(f"{datetime.now().isoformat()}: {error_message}")
        if len(self.error_messages) > 10:  # Keep only recent errors
            self.error_messages = self.error_messages[-10:]
