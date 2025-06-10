# Multi-Model Ensemble Labeling System

The AutoLabeler package now includes a powerful ensemble system that allows you to:

1. **Configure multiple models** with different parameters
2. **Run systematic experiments** across model variants
3. **Consolidate predictions** using various ensemble methods
4. **Track performance** and compare model effectiveness
5. **Maintain full provenance** of all model runs and predictions

## Quick Start

```python
from autolabeler import EnsembleLabeler, ModelConfig, EnsembleMethod, Settings

# Initialize ensemble system
settings = Settings()
ensemble = EnsembleLabeler("my_dataset", settings)

# Add model configurations
conservative = ModelConfig(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    description="Conservative model"
)
creative = ModelConfig(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    description="Creative model"
)

ensemble.add_model_config(conservative)
ensemble.add_model_config(creative)

# Run ensemble labeling
results = ensemble.label_dataframe_ensemble(
    df,
    text_column="text",
    ensemble_method=EnsembleMethod.confidence_weighted()
)
```

## Core Components

### 1. ModelConfig

Defines a specific model configuration with parameters:

```python
config = ModelConfig(
    model_name="gpt-3.5-turbo",
    provider="openrouter",
    temperature=0.3,
    seed=42,
    max_tokens=150,
    use_rag=True,
    max_examples=5,
    description="Balanced configuration",
    tags=["production", "balanced"]
)
```

**Key Parameters:**
- `model_name`: Base model identifier
- `temperature`: Sampling temperature (0.0-1.0)
- `seed`: Random seed for reproducibility
- `use_rag`: Whether to use RAG examples
- `max_examples`: Number of RAG examples to retrieve
- `confidence_threshold`: Minimum confidence for knowledge base updates

### 2. EnsembleMethod

Defines how multiple predictions are consolidated:

```python
# Built-in methods
majority = EnsembleMethod.majority_vote()
weighted = EnsembleMethod.confidence_weighted()
high_agreement = EnsembleMethod.high_agreement()
human_validated = EnsembleMethod.human_validated()

# Custom method
custom = EnsembleMethod(
    method_name="custom",
    description="Custom ensemble logic",
    weight_by_confidence=True,
    min_agreement=0.6,
    min_confidence_threshold=0.4
)
```

**Ensemble Methods:**
- **Majority Vote**: Simple voting, most common label wins
- **Confidence Weighted**: Weight predictions by model confidence
- **High Agreement**: Only return predictions with strong model consensus
- **Human Validated**: Incorporate human validation (future feature)

### 3. EnsembleLabeler

Main class that orchestrates the ensemble process:

```python
ensemble = EnsembleLabeler("dataset_name", settings)

# Add configurations
model_id = ensemble.add_model_config(config)

# Create systematic variants
variant_ids = ensemble.create_model_config_variants(
    base_model="gpt-3.5-turbo",
    temperature_range=[0.1, 0.3, 0.7],
    seed_range=[42, 123, 456]
)

# Single text prediction
result = ensemble.label_text_ensemble(
    "Sample text",
    model_ids=variant_ids[:3],
    ensemble_method=EnsembleMethod.confidence_weighted()
)

# Batch prediction
df_results = ensemble.label_dataframe_ensemble(
    dataframe,
    text_column="content",
    ensemble_method=EnsembleMethod.majority_vote()
)
```

## Advanced Features

### Systematic Experimentation

Create multiple model variants for comprehensive testing:

```python
# Create temperature sweep
temp_variants = ensemble.create_model_config_variants(
    base_model="gpt-3.5-turbo",
    temperature_range=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# Create seed variations
seed_variants = ensemble.create_model_config_variants(
    base_model="gpt-4",
    temperature_range=[0.3],
    seed_range=[42, 123, 456, 789, 101112]
)
```

### Performance Comparison

Analyze model performance across configurations:

```python
# Get performance metrics
performance_df = ensemble.compare_model_performance()
print(performance_df[['model_name', 'temperature', 'avg_confidence', 'success_rate']])

# Get detailed summary
summary = ensemble.get_ensemble_summary()
print(f"Completed runs: {summary['num_completed_runs']}")
print(f"Model configs: {summary['num_model_configs']}")
```

### Result Analysis

Examine ensemble predictions in detail:

```python
# Single prediction with full metadata
result = ensemble.label_text_ensemble("Sample text")
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence}")
print(f"Agreement: {result.model_agreement}")
print(f"Method: {result.ensemble_method}")

# Individual model predictions
for pred in result.individual_predictions:
    print(f"{pred['model_name']}: {pred['label']} ({pred['confidence']:.3f})")
```

## File Organization

The ensemble system creates organized output:

```
ensemble_results/
└── dataset_name/
    ├── ensemble_state.json          # Configuration and run metadata
    ├── individual_model_123.csv     # Individual model results
    ├── individual_model_456.csv
    ├── ensemble_majority_vote.csv   # Consolidated results
    └── ensemble_confidence_weighted.csv
```

## Best Practices

### 1. Model Configuration Strategy

**Start Simple:**
```python
# Begin with basic temperature variations
configs = [
    ModelConfig(model_name="gpt-3.5-turbo", temperature=0.1),  # Conservative
    ModelConfig(model_name="gpt-3.5-turbo", temperature=0.5),  # Balanced
    ModelConfig(model_name="gpt-3.5-turbo", temperature=0.9)   # Creative
]
```

**Add Complexity Gradually:**
```python
# Add seeds for robustness
for temp in [0.1, 0.5, 0.9]:
    for seed in [42, 123, 456]:
        config = ModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=temp,
            seed=seed,
            description=f"T={temp}_seed={seed}"
        )
        ensemble.add_model_config(config)
```

### 2. Ensemble Method Selection

**For High Accuracy Tasks:**
```python
method = EnsembleMethod.high_agreement()  # Only high-consensus predictions
```

**For Balanced Performance:**
```python
method = EnsembleMethod.confidence_weighted()  # Weight by confidence
```

**For Simple Interpretability:**
```python
method = EnsembleMethod.majority_vote()  # Simple voting
```

### 3. Resource Management

**Limit Active Models:**
```python
# Use subset for quick experiments
selected_models = model_ids[:3]
result = ensemble.label_dataframe_ensemble(
    df, "text", model_ids=selected_models
)
```

**Batch Processing:**
```python
# Process in chunks for large datasets
chunk_size = 100
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk_results = ensemble.label_dataframe_ensemble(chunk, "text")
```

### 4. Quality Control

**Filter by Confidence:**
```python
method = EnsembleMethod(
    method_name="high_quality",
    min_confidence_threshold=0.7,  # Only high-confidence predictions
    min_agreement=0.8              # Require strong model agreement
)
```

**Monitor Disagreement:**
```python
# Flag cases where models disagree
high_disagreement = results[results['model_agreement'] < 0.5]
print(f"Found {len(high_disagreement)} cases with low agreement")
```

## Integration with Knowledge Base

The ensemble system integrates seamlessly with the knowledge base:

```python
# Each model uses the same dataset-specific knowledge base
ensemble = EnsembleLabeler("sentiment_analysis", settings)

# High-confidence ensemble predictions are added back to knowledge base
ensemble_df = ensemble.label_dataframe_ensemble(
    df,
    "text",
    save_individual_results=True
)

# Individual AutoLabelers share the same knowledge base
labeler = AutoLabeler("sentiment_analysis", settings)
stats = labeler.get_knowledge_base_stats()
print(f"KB contains: {stats['total_examples']} examples")
```

## Complete Example

See `examples/ensemble_labeling_example.py` for a comprehensive demonstration of:

- Creating multiple model configurations
- Running systematic experiments
- Comparing ensemble methods
- Analyzing performance metrics
- Exporting results for further analysis

The ensemble system provides a robust framework for production-scale auto-labeling with multiple models, complete provenance tracking, and sophisticated consolidation methods.
