"""Dataset configuration for labeling pipelines."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sibyls.core.diagnostics.config import DiagnosticsConfig


@dataclass
class ModelConfig:
    """Configuration for a single LLM in the jury.
    
    Attributes:
        provider: Provider name ("anthropic", "openai", "google", "openrouter")
        model: Model identifier (e.g., "claude-sonnet-4-5-20250929", "gpt-4o")
        name: Human-readable name for this model
        has_logprobs: Whether this provider supports logprobs
        self_consistency_samples: Number of samples for self-consistency confidence
        cost_tier: Relative cost ordering for cascade mode (1 = cheapest).
            Models with the same tier are treated as equally expensive.
    """
    provider: str
    model: str
    name: str
    has_logprobs: bool = False
    self_consistency_samples: int = 0
    cost_tier: int = 1


@dataclass
class DatasetConfig:
    """Complete configuration for a labeling task.
    
    This dataclass contains everything the pipeline needs to know about
    a specific dataset and how to label it.
    
    Attributes:
        name: Dataset identifier (e.g., "fed_headlines", "tpu")
        labels: List of valid labels (e.g., ["-99","-2","-1","0","1","2"] or ["0","1"])
        text_column: Name of column containing text to label
        input_format: File format ("csv" or "jsonl")
        use_relevancy_gate: Whether to use Stage 1 relevancy filter
        use_candidate_annotation: Whether to use Stage 4 candidate annotation for disagreements
        use_typed_rag: Whether to use type-organized RAG retrieval
        use_cross_verification: Whether to use Stage 5 cross-verification for uncertain labels
        use_structured_output: Whether to use JSON schema constraints for structured output
        jury_models: List of models for the jury
        gate_model: Optional model for relevancy gate
        candidate_model: Optional model for candidate annotation
        verification_model: Optional model for cross-verification (must be different family than jury)
        verification_threshold: Confidence below which triggers verification
        jury_temperature: Temperature for jury voting
        sc_temperature: Temperature for self-consistency sampling
        candidate_temperature: Temperature for candidate annotation
        budget_per_model: Budget in USD per model
        batch_size: Concurrent batch size for processing
        use_cascade: Whether to use cascaded model escalation (cheap models first,
            escalate only when confidence is low). Default False = run all jury models.
        cascade_confidence_threshold: Minimum confidence from a single model to
            accept its label without escalating to more models (default 0.85)
        cascade_agreement_threshold: Minimum confidence when two models agree
            to accept without escalating further (default 0.80)
        jury_weights_path: Optional path to learned jury weights JSON file
        program_sample_size: Number of items to sample for program generation
        program_confidence_threshold: Confidence threshold for program labeling
        
    Example:
        >>> config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
        >>> print(config.name)  # "fed_headlines"
        >>> print(config.labels)  # ["-99", "-2", "-1", "0", "1", "2"]
    """
    
    # Dataset identification
    name: str
    labels: list[str]
    text_column: str = "text"
    input_format: str = "csv"  # "csv" or "jsonl"
    
    # Pipeline stage toggles
    use_relevancy_gate: bool = False
    use_candidate_annotation: bool = True
    use_typed_rag: bool = False
    use_cross_verification: bool = False
    use_structured_output: bool = True
    
    # Model configuration
    jury_models: list[ModelConfig] = field(default_factory=list)
    gate_model: ModelConfig | None = None
    candidate_model: ModelConfig | None = None
    verification_model: ModelConfig | None = None
    
    # Verification configuration
    verification_threshold: float = 0.6
    
    # Temperature settings
    jury_temperature: float = 0.1
    sc_temperature: float = 0.3  # Self-consistency sampling
    candidate_temperature: float = 0.2
    
    # Resource constraints
    budget_per_model: float = 10.0
    batch_size: int = 10
    
    # Cascade mode (Trust or Escalate)
    use_cascade: bool = False
    cascade_confidence_threshold: float = 0.85
    cascade_agreement_threshold: float = 0.80
    
    # Advanced features
    jury_weights_path: str | None = None
    program_sample_size: int = 500
    program_confidence_threshold: float = 0.8

    # Post-labeling diagnostics (optional)
    diagnostics: DiagnosticsConfig | None = None

    # LIT integration defaults (optional -- all values can be overridden via CLI)
    lit_config: dict | None = None
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> 'DatasetConfig':
        """Load configuration from YAML file.
        
        Parameters:
            path: Path to YAML config file
            
        Returns:
            DatasetConfig instance
            
        Example:
            >>> config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Convert jury_models list[dict] -> list[ModelConfig]
        jury_models = []
        for m in data.get('jury_models', []):
            jury_models.append(ModelConfig(**m))
        
        # Convert gate_model dict -> ModelConfig
        gate_model = None
        if 'gate_model' in data and data['gate_model']:
            gate_model = ModelConfig(**data['gate_model'])
        
        # Convert candidate_model dict -> ModelConfig
        candidate_model = None
        if 'candidate_model' in data and data['candidate_model']:
            candidate_model = ModelConfig(**data['candidate_model'])
        
        # Convert verification_model dict -> ModelConfig
        verification_model = None
        if 'verification_model' in data and data['verification_model']:
            verification_model = ModelConfig(**data['verification_model'])
        
        # Build config
        return cls(
            name=data['name'],
            labels=data['labels'],
            text_column=data.get('text_column', 'text'),
            input_format=data.get('input_format', 'csv'),
            use_relevancy_gate=data.get('use_relevancy_gate', False),
            use_candidate_annotation=data.get('use_candidate_annotation', True),
            use_typed_rag=data.get('use_typed_rag', False),
            use_cross_verification=data.get('use_cross_verification', False),
            use_structured_output=data.get('use_structured_output', True),
            jury_models=jury_models,
            gate_model=gate_model,
            candidate_model=candidate_model,
            verification_model=verification_model,
            verification_threshold=data.get('verification_threshold', 0.6),
            jury_temperature=data.get('jury_temperature', 0.1),
            sc_temperature=data.get('sc_temperature', 0.3),
            candidate_temperature=data.get('candidate_temperature', 0.2),
            budget_per_model=data.get('budget_per_model', 10.0),
            batch_size=data.get('batch_size', 10),
            use_cascade=data.get('use_cascade', False),
            cascade_confidence_threshold=data.get('cascade_confidence_threshold', 0.85),
            cascade_agreement_threshold=data.get('cascade_agreement_threshold', 0.80),
            jury_weights_path=data.get('jury_weights_path'),
            program_sample_size=data.get('program_sample_size', 500),
            program_confidence_threshold=data.get('program_confidence_threshold', 0.8),
            diagnostics=(
                DiagnosticsConfig.from_dict(data['diagnostics'])
                if data.get('diagnostics')
                else None
            ),
            lit_config=data.get('lit') or None,
        )
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.
        
        Parameters:
            path: Path to save YAML config file
        """
        path = Path(path)
        
        # Convert to dict
        data = {
            'name': self.name,
            'labels': self.labels,
            'text_column': self.text_column,
            'input_format': self.input_format,
            'use_relevancy_gate': self.use_relevancy_gate,
            'use_candidate_annotation': self.use_candidate_annotation,
            'use_typed_rag': self.use_typed_rag,
            'jury_models': [
                {
                    'provider': m.provider,
                    'model': m.model,
                    'name': m.name,
                    'has_logprobs': m.has_logprobs,
                    'self_consistency_samples': m.self_consistency_samples,
                }
                for m in self.jury_models
            ],
            'jury_temperature': self.jury_temperature,
            'sc_temperature': self.sc_temperature,
            'candidate_temperature': self.candidate_temperature,
            'budget_per_model': self.budget_per_model,
            'batch_size': self.batch_size,
        }
        
        if self.gate_model:
            data['gate_model'] = {
                'provider': self.gate_model.provider,
                'model': self.gate_model.model,
                'name': self.gate_model.name,
            }
        
        if self.candidate_model:
            data['candidate_model'] = {
                'provider': self.candidate_model.provider,
                'model': self.candidate_model.model,
                'name': self.candidate_model.name,
            }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DatasetConfig(name='{self.name}', "
            f"labels={len(self.labels)}, "
            f"jury_models={len(self.jury_models)})"
        )
