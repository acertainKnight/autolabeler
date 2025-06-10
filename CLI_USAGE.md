# AutoLabeler CLI Usage Guide

The AutoLabeler CLI provides a powerful command-line interface for text labeling, synthetic data generation, and ensemble methods. It supports JSON configuration files that can specify multiple model configurations for comprehensive experimentation.

## Installation

```bash
pip install -e .
```

This will install the `autolabeler` command-line tool.

## Quick Start

1. **Create a configuration file:**
   ```bash
   autolabeler create-config
   ```
   This creates `autolabeler_config.json` with sample configurations.

2. **Edit the configuration file** to add your API keys and adjust model parameters.

3. **Label your data:**
   ```bash
   autolabeler label config.json input.csv output.csv --text-column "text" --dataset-name "my_dataset"
   ```

## Configuration Format

The CLI uses JSON configuration files with the following structure:

```json
{
  "settings": {
    "openrouter_api_key": "your-api-key",
    "corporate_api_key": "",
    "corporate_base_url": null,
    "llm_model": "openai/gpt-3.5-turbo",
    "embedding_model": "all-MiniLM-L6-v2",
    "max_examples_per_query": 5,
    "similarity_threshold": 0.8,
    "knowledge_base_dir": "knowledge_bases"
  },
  "models": [
    {
      "model_name": "openai/gpt-3.5-turbo",
      "provider": "openrouter",
      "temperature": 0.1,
      "seed": 42,
      "description": "Conservative model",
      "tags": ["conservative", "low-temp"],
      "custom_params": {}
    },
    {
      "model_name": "anthropic/claude-3-haiku",
      "provider": "openrouter",
      "temperature": 0.7,
      "seed": 123,
      "description": "Creative model",
      "tags": ["creative", "high-temp"],
      "custom_params": {}
    }
  ]
}
```

## Commands

### `label` - Label Text Data

Label text data using multiple model configurations.

```bash
autolabeler label CONFIG_FILE INPUT_FILE OUTPUT_FILE [OPTIONS]
```

**Arguments:**
- `CONFIG_FILE`: Path to JSON configuration file
- `INPUT_FILE`: Path to input CSV file
- `OUTPUT_FILE`: Path to output CSV file

**Options:**
- `--text-column TEXT`: Name of column containing text to label (required)
- `--dataset-name TEXT`: Name of the dataset for knowledge base (required)
- `--label-column TEXT`: Name of column to store predictions (default: "predicted_label")
- `--use-rag/--no-rag`: Whether to use RAG examples (default: True)
- `--save-to-kb/--no-save-to-kb`: Whether to save predictions to knowledge base (default: True)
- `--confidence-threshold FLOAT`: Minimum confidence to save to KB (default: 0.0)

**Example:**
```bash
autolabeler label config.json reviews.csv labeled_reviews.csv \
  --text-column "review_text" \
  --dataset-name "product_reviews" \
  --label-column "sentiment" \
  --confidence-threshold 0.8
```

### `ensemble` - Ensemble Labeling

Run ensemble labeling using multiple models with consolidation.

```bash
autolabeler ensemble CONFIG_FILE INPUT_FILE OUTPUT_FILE [OPTIONS]
```

**Options:**
- `--ensemble-method CHOICE`: Consolidation method (majority_vote, confidence_weighted, high_agreement)
- `--save-individual/--no-save-individual`: Whether to save individual model results (default: True)

**Example:**
```bash
autolabeler ensemble config.json data.csv ensemble_results.csv \
  --text-column "text" \
  --dataset-name "sentiment_analysis" \
  --ensemble-method "confidence_weighted"
```

### `generate` - Generate Synthetic Examples

Generate synthetic examples for a specific label.

```bash
autolabeler generate CONFIG_FILE [OPTIONS]
```

**Options:**
- `--dataset-name TEXT`: Name of the dataset (required)
- `--target-label TEXT`: Target label for synthetic examples (required)
- `--num-examples INTEGER`: Number of examples to generate (default: 5)
- `--strategy CHOICE`: Generation strategy (paraphrase, interpolate, extrapolate, transform, mixed)
- `--output-file PATH`: Output file for synthetic examples
- `--add-to-kb/--no-add-to-kb`: Whether to add to knowledge base (default: True)
- `--confidence-threshold FLOAT`: Minimum confidence to add to KB (default: 0.7)

**Example:**
```bash
autolabeler generate config.json \
  --dataset-name "sentiment_analysis" \
  --target-label "positive" \
  --num-examples 10 \
  --strategy "mixed" \
  --output-file "synthetic_positive.csv"
```

### `balance` - Balance Dataset

Generate synthetic examples to balance dataset labels.

```bash
autolabeler balance CONFIG_FILE [OPTIONS]
```

**Options:**
- `--dataset-name TEXT`: Name of the dataset (required)
- `--target-balance TEXT`: Target balance: "equal" or JSON dict like `{"pos":100,"neg":100}`
- `--max-per-label INTEGER`: Maximum synthetic examples per label (default: 50)
- `--confidence-threshold FLOAT`: Minimum confidence threshold (default: 0.7)
- `--output-file PATH`: Output file for synthetic examples

**Examples:**
```bash
# Balance to equal distribution
autolabeler balance config.json \
  --dataset-name "reviews" \
  --target-balance "equal" \
  --output-file "balanced_data.csv"

# Balance to specific counts
autolabeler balance config.json \
  --dataset-name "reviews" \
  --target-balance '{"positive": 1000, "negative": 1000, "neutral": 500}' \
  --max-per-label 100
```

### `stats` - Show Dataset Statistics

Display statistics for a dataset's knowledge base.

```bash
autolabeler stats DATASET_NAME [OPTIONS]
```

**Options:**
- `--config-file PATH`: Configuration file for settings

**Example:**
```bash
autolabeler stats "product_reviews" --config-file config.json
```

### `create-config` - Create Sample Configuration

Create a sample configuration file with multiple model setups.

```bash
autolabeler create-config
```

This creates `autolabeler_config.json` with example configurations for different models and parameters.

## Multiple Model Workflows

### Systematic Model Comparison

Compare different models on the same dataset:

```json
{
  "models": [
    {
      "model_name": "openai/gpt-3.5-turbo",
      "temperature": 0.1,
      "description": "GPT-3.5 Conservative"
    },
    {
      "model_name": "openai/gpt-4",
      "temperature": 0.1,
      "description": "GPT-4 Conservative"
    },
    {
      "model_name": "anthropic/claude-3-haiku",
      "temperature": 0.1,
      "description": "Claude Haiku"
    },
    {
      "model_name": "anthropic/claude-3-sonnet",
      "temperature": 0.1,
      "description": "Claude Sonnet"
    }
  ]
}
```

### Parameter Exploration

Test different parameters for the same model:

```json
{
  "models": [
    {
      "model_name": "openai/gpt-3.5-turbo",
      "temperature": 0.0,
      "seed": 42,
      "description": "Deterministic"
    },
    {
      "model_name": "openai/gpt-3.5-turbo",
      "temperature": 0.3,
      "seed": 42,
      "description": "Low creativity"
    },
    {
      "model_name": "openai/gpt-3.5-turbo",
      "temperature": 0.7,
      "seed": 42,
      "description": "High creativity"
    },
    {
      "model_name": "openai/gpt-3.5-turbo",
      "temperature": 1.0,
      "seed": 42,
      "description": "Maximum creativity"
    }
  ]
}
```

## Complete Workflow Example

Here's a complete workflow from data preparation to ensemble results:

```bash
# 1. Create configuration
autolabeler create-config
# Edit autolabeler_config.json with your API keys

# 2. Check initial dataset stats
autolabeler stats "my_dataset" --config-file autolabeler_config.json

# 3. Label data with multiple models
autolabeler label autolabeler_config.json raw_data.csv individual_results.csv \
  --text-column "text" \
  --dataset-name "my_dataset" \
  --use-rag

# 4. Generate ensemble predictions
autolabeler ensemble autolabeler_config.json raw_data.csv ensemble_results.csv \
  --text-column "text" \
  --dataset-name "my_dataset" \
  --ensemble-method "confidence_weighted"

# 5. Analyze class imbalance and generate synthetic data
autolabeler balance autolabeler_config.json \
  --dataset-name "my_dataset" \
  --target-balance "equal" \
  --output-file "synthetic_balance.csv"

# 6. Check final stats
autolabeler stats "my_dataset" --config-file autolabeler_config.json
```

## Output Files

### Label Command Output
The `label` command creates a CSV file with columns for each model configuration:
- Original data columns
- `{label_column}_{model_id}`: Predicted label for each model
- `{label_column}_{model_id}_confidence`: Confidence score for each model
- `{label_column}_{model_id}_model`: Model name used
- `{label_column}_{model_id}_temp`: Temperature parameter used
- `{label_column}_{model_id}_config_id`: Model configuration ID

### Ensemble Command Output
The `ensemble` command creates a CSV file with:
- Original data columns
- `ensemble_label`: Final ensemble prediction
- `ensemble_confidence`: Ensemble confidence score
- `ensemble_method`: Method used for consolidation
- `model_agreement`: Agreement level between models
- Individual model predictions (if `--save-individual` is True)

### Generate/Balance Command Output
Synthetic generation commands create CSV files with:
- `text`: Generated text
- `label`: Target label
- `confidence`: Generation confidence
- `reasoning`: Generation reasoning
- `strategy`: Generation strategy used
- `model_config_id`: Model configuration ID
- `meta_*`: Additional generation metadata

## Tips and Best Practices

1. **Start Small**: Begin with 2-3 model configurations to test your workflow
2. **Use Ensemble Methods**: Ensemble predictions are typically more robust than individual models
3. **Monitor Costs**: Different models have different costs - use cheaper models for experimentation
4. **Leverage RAG**: Enable RAG to improve predictions with similar examples
5. **Balance Your Data**: Use synthetic generation to address class imbalance
6. **Track Experiments**: Use descriptive names in your model configurations
7. **Version Control**: Keep your configuration files in version control
8. **Validate Results**: Always manually check a sample of predictions for quality

## Error Handling

The CLI includes comprehensive error handling:
- Configuration validation with helpful error messages
- Graceful handling of API failures
- Partial results when some models fail
- Clear logging and progress indicators

## Integration with Python

You can also use the CLI programmatically:

```python
import subprocess
import json

# Create configuration
config = {
    "settings": {"openrouter_api_key": "your-key"},
    "models": [{"model_name": "openai/gpt-3.5-turbo", "temperature": 0.1}]
}

with open("config.json", "w") as f:
    json.dump(config, f)

# Run labeling
subprocess.run([
    "autolabeler", "label", "config.json", "input.csv", "output.csv",
    "--text-column", "text", "--dataset-name", "my_dataset"
])
```

For more advanced usage, see the `examples/cli_usage_example.py` script.
