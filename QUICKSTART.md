# AutoLabeler Quick Start Guide

Get up and running with AutoLabeler in 5 minutes!

## 1. Installation

```bash
# Clone and install
git clone https://github.com/yourusername/autolabeler.git
cd autolabeler
pip install -e .
```

## 2. Set Up API Keys

Create a `.env` file:
```env
OPENROUTER_API_KEY=your_openrouter_key  # Get from https://openrouter.ai
```

## 3. Your First Labeling Task

### Option A: Python Script

```python
from autolabeler import AutoLabeler, Settings
import pandas as pd

# Initialize
settings = Settings()  # Reads from .env
labeler = AutoLabeler("my_first_project", settings)

# Add some training examples
train_data = pd.DataFrame({
    "text": [
        "This product is amazing! Best purchase ever.",
        "Terrible quality, complete waste of money.",
        "It's okay, nothing special but does the job.",
    ],
    "label": ["positive", "negative", "neutral"]
})
labeler.add_training_data(train_data, "text", "label")

# Label new data
new_data = pd.DataFrame({
    "text": [
        "Fantastic service, highly recommend!",
        "Disappointed with this purchase.",
        "Average product, fair price.",
    ]
})

# Get predictions
results = labeler.label_dataframe(new_data, "text")
print(results[["text", "predicted_label", "predicted_label_confidence"]])
```

### Option B: Command Line

1. Create a config file `config.json`:

```json
{
  "project": {
    "name": "sentiment_analysis",
    "description": "Customer review sentiment"
  },
  "data": {
    "input_file": "reviews.csv",
    "text_column": "review",
    "label_column": "sentiment"
  },
  "models": [{
    "model_name": "meta-llama/llama-3.1-8b-instruct:free",
    "temperature": 0.1,
    "save_interval": 50,
    "workers": 1
  }]
}
```

2. Prepare your data `reviews.csv`:
```csv
review,sentiment
"Great product!",positive
"Not worth it.",negative
"It's fine.",neutral
```

3. Run labeling:
```bash
python -m autolabeler.cli label \
    --config config.json \
    --input unlabeled_reviews.csv \
    --output labeled_reviews.csv
```

## 4. Using Batch Processing (Faster!)

For larger datasets, use batch processing:

```python
# Process 100 texts at a time with 5 concurrent API calls
results = labeler.label_dataframe_batch(
    large_df,
    text_column="text",
    batch_size=100,
    max_concurrency=5
)
```

## 5. Async for Maximum Speed

```python
import asyncio

async def label_fast():
    results = await labeler.label_dataframe_batch_async(
        huge_df,
        text_column="text",
        batch_size=200,
        max_concurrency=10
    )
    return results

# Run it
labeled_df = asyncio.run(label_fast())
```

## 6. Generate Synthetic Data

Balance your dataset automatically:

```python
from autolabeler import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    "my_project",
    settings,
    knowledge_base=labeler.knowledge_base
)

# Generate equal examples for each label
balanced = generator.balance_dataset("equal")
```

## 7. Extract Labeling Rules

```python
from autolabeler import RuleGenerator

rule_gen = RuleGenerator("my_project", settings)

# Generate rules from your labeled data
result = rule_gen.generate_rules_from_data(
    labeled_df,
    text_column="text",
    label_column="label"
)

# Export human-readable guidelines
rule_gen.export_ruleset_for_humans(
    result.ruleset,
    Path("labeling_guidelines.md")
)
```

## Common Use Cases

### Customer Support Tickets
```python
labeler = AutoLabeler("support_tickets", settings)
categories = ["billing", "technical", "feature_request", "bug_report"]
```

### Content Moderation
```python
labeler = AutoLabeler("content_moderation", settings)
labels = ["safe", "needs_review", "inappropriate"]
```

### Sentiment Analysis
```python
labeler = AutoLabeler("sentiment", settings)
labels = ["positive", "negative", "neutral"]
```

## Tips for Best Results

1. **Start with 10-20 examples per label** for initial training
2. **Use low temperature (0.1-0.3)** for consistent labeling
3. **Enable RAG** (on by default) for better accuracy
4. **Review low-confidence predictions** (< 0.7)
5. **Use batch processing** for datasets > 1000 items

## Next Steps

- Read the [full documentation](README.md)
- Check out [example notebooks](examples/)
- Join our [Discord community](https://discord.gg/autolabeler)

## Need Help?

- 📚 [Full Documentation](README.md)
- 💬 [Discord Community](https://discord.gg/autolabeler)
- 🐛 [Report Issues](https://github.com/yourusername/autolabeler/issues)
- 📧 Email: support@autolabeler.com
