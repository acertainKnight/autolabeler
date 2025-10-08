"""Configuration classes for AutoLabeler components."""

from __future__ import annotations

from typing import Any, Literal

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


class DSPyOptimizationConfig(BaseModel):
    """Configuration for DSPy prompt optimization."""

    enabled: bool = Field(False, description="Enable DSPy optimization")
    model_name: str = Field('gpt-4o-mini', description="Model for optimization")
    num_candidates: int = Field(10, description="Number of prompt candidates")
    num_trials: int = Field(20, description="Number of optimization trials")
    max_bootstrapped_demos: int = Field(4, description="Max bootstrapped demos")
    max_labeled_demos: int = Field(8, description="Max labeled demos")
    init_temperature: float = Field(1.0, description="Initial temperature")
    metric_threshold: float = Field(0.8, description="Success threshold")
    cache_optimized_prompts: bool = Field(True, description="Cache optimized prompts")


class AdvancedRAGConfig(BaseModel):
    """Configuration for advanced RAG methods."""

    rag_mode: str = Field('traditional', description="RAG mode: traditional, graph, or raptor")

    # GraphRAG settings
    graph_similarity_threshold: float = Field(0.7, description="Similarity threshold for graph edges")
    graph_max_neighbors: int = Field(10, description="Max neighbors per graph node")
    graph_use_communities: bool = Field(True, description="Use community detection")
    graph_pagerank_alpha: float = Field(0.85, description="PageRank damping factor")

    # RAPTOR settings
    raptor_max_tree_depth: int = Field(3, description="Maximum tree depth for RAPTOR")
    raptor_clustering_threshold: float = Field(0.5, description="Clustering threshold")
    raptor_min_cluster_size: int = Field(3, description="Minimum cluster size")
    raptor_summary_length: int = Field(100, description="Summary length in tokens")
    raptor_use_multi_level: bool = Field(True, description="Use multi-level retrieval")

    # Auto-build settings
    auto_build_on_startup: bool = Field(False, description="Auto-build advanced indices on startup")
    rebuild_interval_hours: int | None = Field(None, description="Auto-rebuild interval")


class ActiveLearningConfig(BaseModel):
    """Configuration for active learning."""

    # Strategy selection
    strategy: Literal["uncertainty", "diversity", "committee", "hybrid"] = Field(
        "hybrid",
        description="Active learning sampling strategy"
    )

    # Uncertainty parameters
    uncertainty_method: Literal["least_confident", "margin", "entropy"] = Field(
        "least_confident",
        description="Method for calculating uncertainty"
    )

    # Diversity parameters
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="Embedding model for diversity sampling"
    )

    # Hybrid parameters
    hybrid_alpha: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Weight for uncertainty in hybrid strategy (1-alpha for diversity)"
    )

    # Sampling parameters
    batch_size: int = Field(
        50,
        gt=0,
        description="Number of samples to select per iteration"
    )
    initial_seed_size: int = Field(
        100,
        gt=0,
        description="Initial seed dataset size"
    )
    text_column: str = Field(
        "text",
        description="Column containing text to label"
    )

    # Stopping criteria
    max_iterations: int = Field(
        20,
        gt=0,
        description="Maximum active learning iterations"
    )
    max_budget: float = Field(
        100.0,
        gt=0,
        description="Maximum budget in USD"
    )
    target_accuracy: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Target accuracy to reach"
    )
    patience: int = Field(
        3,
        gt=0,
        description="Iterations without improvement before stopping"
    )
    improvement_threshold: float = Field(
        0.01,
        gt=0,
        description="Minimum improvement considered significant"
    )
    uncertainty_threshold: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Stop if pool uncertainty below this threshold"
    )

    # Human-in-the-loop
    enable_human_review: bool = Field(
        False,
        description="Enable human review for selected samples"
    )
    human_review_confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Send samples below this confidence to human review"
    )


class WeakSupervisionConfig(BaseModel):
    """Configuration for weak supervision."""

    # Aggregation method
    aggregation_method: Literal["majority", "snorkel", "flyingsquid"] = Field(
        "snorkel",
        description="Label aggregation method"
    )

    # Snorkel parameters
    n_epochs: int = Field(
        500,
        gt=0,
        description="Training epochs for Snorkel label model"
    )
    learning_rate: float = Field(
        0.01,
        gt=0,
        description="Learning rate for label model"
    )

    # LF generation parameters
    enable_lf_generation: bool = Field(
        True,
        description="Enable LLM-based LF generation"
    )
    num_generated_lfs: int = Field(
        10,
        gt=0,
        description="Number of LFs to generate with LLM"
    )
    lf_generation_model: str = Field(
        "gpt-4o-mini",
        description="Model for LF generation"
    )

    # Quality thresholds
    min_lf_coverage: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Minimum coverage for LF to be kept"
    )
    min_lf_accuracy: float = Field(
        0.55,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy for LF to be kept (if dev set available)"
    )
    max_lf_conflicts: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Maximum conflict rate for LF"
    )

    # Processing
    batch_size: int = Field(
        1000,
        gt=0,
        description="Batch size for applying LFs"
    )
    text_column: str = Field(
        "text",
        description="Column containing text"
    )

    # Output
    save_label_matrix: bool = Field(
        True,
        description="Save label matrix for debugging"
    )
    save_lf_analysis: bool = Field(
        True,
        description="Save LF analysis report"
    )


# Phase 3: Advanced Features Configurations


class MultiAgentConfig(BaseModel):
    """Configuration for multi-agent system."""

    coordinator_id: str = Field(
        "coordinator_main",
        description="Coordinator identifier"
    )
    routing_strategy: str = Field(
        "performance_based",
        description="Agent routing strategy (performance_based, round_robin, random)"
    )
    enable_parallel: bool = Field(
        True,
        description="Enable parallel agent execution"
    )
    max_concurrent_agents: int = Field(
        5,
        gt=0,
        description="Maximum concurrent agent tasks"
    )
    performance_window: int = Field(
        100,
        gt=0,
        description="Window size for performance tracking"
    )


class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""

    psi_threshold: float = Field(
        0.2,
        ge=0.0,
        description="PSI threshold for significant drift"
    )
    statistical_alpha: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Significance level for statistical tests"
    )
    domain_classifier_threshold: float = Field(
        0.75,
        ge=0.5,
        le=1.0,
        description="AUC threshold for domain classifier drift detection"
    )
    min_samples: int = Field(
        100,
        gt=0,
        description="Minimum samples for reliable drift detection"
    )
    num_bins: int = Field(
        10,
        gt=1,
        description="Number of bins for PSI calculation"
    )
    test_size: float = Field(
        0.3,
        ge=0.1,
        le=0.5,
        description="Test size for domain classifier"
    )


class DPOAlignmentConfig(BaseModel):
    """Configuration for Direct Preference Optimization."""

    base_model: str = Field(
        description="Base model name for fine-tuning"
    )
    output_dir: str = Field(
        description="Directory for saving aligned model"
    )
    num_epochs: int = Field(
        3,
        gt=0,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        5e-5,
        gt=0,
        description="Learning rate"
    )
    batch_size: int = Field(
        4,
        gt=0,
        description="Training batch size"
    )
    gradient_accumulation_steps: int = Field(
        4,
        gt=0,
        description="Gradient accumulation steps"
    )
    beta: float = Field(
        0.1,
        gt=0,
        description="DPO regularization parameter"
    )
    max_length: int = Field(
        512,
        gt=0,
        description="Maximum sequence length"
    )
    warmup_steps: int = Field(
        100,
        gt=0,
        description="Warmup steps"
    )
    save_strategy: str = Field(
        "epoch",
        description="Model save strategy (epoch, steps)"
    )
    logging_steps: int = Field(
        10,
        gt=0,
        description="Logging frequency"
    )


class ConstitutionalConfig(BaseModel):
    """Configuration for Constitutional AI."""

    constitution_path: str = Field(
        description="Path to constitution JSON file"
    )
    max_revisions: int = Field(
        2,
        gt=0,
        description="Maximum revision iterations"
    )
    critique_temperature: float = Field(
        0.2,
        ge=0,
        le=2,
        description="Temperature for critique generation"
    )
    revision_temperature: float = Field(
        0.3,
        ge=0,
        le=2,
        description="Temperature for revision generation"
    )
    require_unanimous_compliance: bool = Field(
        False,
        description="Require all principles to pass"
    )


class STAPLEConfig(BaseModel):
    """Configuration for STAPLE ensemble algorithm."""

    num_classes: int = Field(
        gt=1,
        description="Number of label classes"
    )
    max_iterations: int = Field(
        50,
        gt=0,
        description="Maximum EM iterations"
    )
    convergence_threshold: float = Field(
        1e-5,
        gt=0,
        description="Convergence threshold for ground truth changes"
    )
    min_annotators: int = Field(
        2,
        gt=1,
        description="Minimum number of annotators required"
    )
