# Multi-Field Extraction with Ensemble Modeling - Usage Guide

This guide shows you how to extract multiple pieces of information from headlines using ensemble modeling with multiple LLMs, seeds, and temperatures.

## Overview

Your pipeline will:
1. **Extract 3 fields**: Speaker, Relevance, Sentiment
2. **Use 27 models**: 3 LLMs × 3 seeds × 3 temperatures
3. **Leverage context**: Past headlines from the same speech
4. **Provide uncertainty**: Track disagreement between models
5. **Ensemble predictions**: Combine all 27 predictions intelligently

## Quick Start

### 1. Basic Setup

```python
from autolabeler.config import Settings
from examples.multi_extraction_ensemble import HeadlineMultiExtractor
import pandas as pd

# Configure your settings
settings = Settings(
    openrouter_api_key="your-api-key-here",
    max_examples_per_query=5,
    similarity_threshold=0.8
)

# Define your research question
target_question = "How does this headline relate to climate change policy?"

# Initialize the extractor
extractor = HeadlineMultiExtractor(
    dataset_name="my_headlines",
    target_question=target_question,
    base_settings=settings
)
```

### 2. Prepare Your Training Data

You need two datasets: one for relevance labels, one for sentiment labels.

```python
# Relevance training data
relevance_df = pd.DataFrame({
    "headline": [
        "Biden announces new climate regulations",
        "Stock market hits new highs",
        "EPA proposes emissions standards"
    ],
    "relevance": [
        "highly_relevant",
        "not_relevant",
        "highly_relevant"
    ]
})

# Sentiment training data
sentiment_df = pd.DataFrame({
    "headline": [
        "Environmental groups praise new initiatives",
        "Industry leaders slam costly regulations",
        "Bipartisan support grows for legislation"
    ],
    "sentiment": [
        "positive",
        "negative",
        "positive"
    ]
})

# Add training data to all 27 models
extractor.prepare_training_data(relevance_df, sentiment_df)
```

### 3. Extract from Your Headlines

```python
# Your headlines to analyze
headlines_df = pd.DataFrame({
    "headline": [
        "President calls for immediate action on emissions",
        "Senator criticizes green energy subsidies",
        "Tech companies pledge net-zero by 2030"
    ]
})

# Context from past headlines (optional but recommended)
past_headlines = [
    "President addresses environmental priorities",
    "Administration unveils climate strategy",
    "New funding for renewable energy projects"
]

# Run the extraction (this generates 27 predictions per headline)
results = extractor.extract_from_headlines(
    headlines_df,
    headline_column="headline",
    past_headlines=past_headlines
)
```

### 4. Analyze Results

```python
# Get comprehensive statistics
analysis = extractor.analyze_results(results)

print(f"Success rate: {analysis['success_rate']:.1%}")
print(f"Average confidence: {analysis['ensemble_overall_confidence_mean']:.3f}")

# View results
display_cols = [
    "headline",
    "ensemble_speaker",
    "ensemble_relevance",
    "ensemble_sentiment",
    "ensemble_overall_confidence",
    "speaker_uncertainty",
    "num_models_succeeded"
]

print(results[display_cols])
```

## Advanced Usage

### Custom Model Configurations

If you want different LLMs or parameters, modify the `HeadlineMultiExtractor` class:

```python
# In the __init__ method, customize these:
self.llm_models = [
    "anthropic/claude-3-sonnet",
    "openai/gpt-4o-mini",
    "google/gemini-pro",
    "meta-llama/llama-2-70b-chat"  # Add more models
]
self.seeds = [42, 123, 789, 456]  # Add more seeds
self.temperatures = [0.0, 0.3, 0.7, 1.0]  # Add more temperatures
```

### Adding More Context

```python
# Enhanced context with additional metadata
headlines_df = pd.DataFrame({
    "headline": ["Your headlines..."],
    "speaker_type": ["politician", "activist", "scientist"],
    "publication": ["CNN", "Fox News", "Reuters"],
    "date": ["2024-01-15", "2024-01-16", "2024-01-17"]
})

results = extractor.extract_from_headlines(
    headlines_df,
    headline_column="headline",
    context_columns=["speaker_type", "publication", "date"],  # Include extra context
    past_headlines=past_headlines
)
```

### Analyzing Individual Model Predictions

```python
import json

# Extract individual predictions for detailed analysis
for _, row in results.iterrows():
    individual_preds = json.loads(row["individual_predictions"])

    print(f"\nHeadline: {row['headline']}")
    print(f"Ensemble: {row['ensemble_speaker']} | {row['ensemble_relevance']} | {row['ensemble_sentiment']}")

    # See how different models disagreed
    for pred in individual_preds:
        print(f"  {pred['model_config']}: {pred['speaker']} | {pred['relevance']} | {pred['sentiment']} (conf: {pred['overall_confidence']:.2f})")
```

## Understanding the Results

### Key Output Columns

- **`ensemble_speaker`**: Final speaker prediction (majority vote)
- **`ensemble_relevance`**: Final relevance prediction (highly_relevant, moderately_relevant, etc.)
- **`ensemble_sentiment`**: Final sentiment prediction (positive, negative, neutral, mixed)
- **`ensemble_*_confidence`**: Confidence scores for each field (0.0-1.0)
- **`*_uncertainty`**: Disagreement between models (0.0-1.0, higher = more disagreement)
- **`num_models_succeeded`**: How many of the 27 models completed successfully

### Interpreting Uncertainty

- **Low uncertainty (< 0.3)**: Models mostly agree, high confidence
- **Medium uncertainty (0.3-0.7)**: Some disagreement, moderate confidence
- **High uncertainty (> 0.7)**: Significant disagreement, low confidence

### Quality Control

```python
# Filter high-confidence predictions
high_confidence = results[results["ensemble_overall_confidence"] > 0.8]

# Identify uncertain cases for manual review
uncertain_cases = results[
    (results["speaker_uncertainty"] > 0.5) |
    (results["relevance_uncertainty"] > 0.5) |
    (results["sentiment_uncertainty"] > 0.5)
]

print(f"High confidence cases: {len(high_confidence)}")
print(f"Uncertain cases for review: {len(uncertain_cases)}")
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: The pipeline makes many API calls (27 per headline). Consider:
   - Using smaller batches
   - Adding delays between requests
   - Using multiple API keys

2. **Model Failures**: Some models might fail. Check `num_models_succeeded`:
   ```python
   failed_cases = results[results["num_models_succeeded"] < 20]  # Less than 20/27 models
   ```

3. **Low Training Data**: Add more examples if confidence is consistently low:
   ```python
   # Add more training data
   extractor.prepare_training_data(more_relevance_data, more_sentiment_data)
   ```

### Performance Optimization

```python
# Reduce model configurations for faster processing
extractor.llm_models = ["openai/gpt-4o-mini"]  # Use just one model
extractor.seeds = [42]  # Use just one seed
extractor.temperatures = [0.1, 0.5, 0.9]  # Keep temperature variation

# This reduces from 27 to 3 models per headline
extractor.model_configs = extractor._generate_model_configs()
extractor.labelers = extractor._initialize_labelers()
```

## Export and Analysis

```python
# Export detailed results
output_dir = Path("my_results")
output_dir.mkdir(exist_ok=True)

# Main results
results.to_csv(output_dir / "headline_extractions.csv", index=False)

# Analysis summary
analysis = extractor.analyze_results(results)
with open(output_dir / "analysis_summary.json", "w") as f:
    json.dump(analysis, f, indent=2, default=str)

# Individual model breakdown
individual_data = []
for _, row in results.iterrows():
    preds = json.loads(row["individual_predictions"])
    for pred in preds:
        pred["headline_index"] = row["headline_index"]
        pred["headline"] = row["headline"]
        individual_data.append(pred)

pd.DataFrame(individual_data).to_csv(output_dir / "individual_predictions.csv", index=False)
```

## Next Steps

1. **Analyze patterns**: Look for systematic biases across models/temperatures
2. **Improve training data**: Add examples where models disagree
3. **Fine-tune thresholds**: Adjust confidence and uncertainty thresholds based on your needs
4. **Scale up**: Process larger datasets with the optimized pipeline
5. **Domain adaptation**: Customize for your specific domain (finance, politics, etc.)

## Example Output

```
headline                                    ensemble_speaker  ensemble_relevance  ensemble_sentiment  ensemble_overall_confidence  speaker_uncertainty  num_models_succeeded
President calls for immediate action on emissions    President Biden   highly_relevant     positive                    0.87                    0.12                    25
Senator criticizes green energy subsidies           Senator Johnson   highly_relevant     negative                    0.82                    0.15                    24
Tech companies pledge net-zero by 2030             Tech companies    moderately_relevant  positive                    0.79                    0.28                    26
```

This gives you robust, uncertainty-quantified predictions for each headline with full traceability back to individual model decisions.
