# Rule Generation Feature

The AutoLabeler now includes a powerful rule generation feature that automatically creates comprehensive labeling guidelines from your existing labeled data. This feature helps maintain consistency across human annotators and provides clear, evidence-based rules for labeling tasks.

## Overview

The rule generation system analyzes patterns in your labeled training data and creates:

- **Specific labeling rules** with patterns, conditions, and indicators
- **Human-readable annotation guidelines** in multiple formats
- **Quality assurance recommendations** for maintaining consistency
- **Coverage analysis** to identify gaps in your rules
- **Automatic rule updates** when new data becomes available

## Key Features

### 🧠 Intelligent Pattern Recognition
- Analyzes linguistic patterns and indicators for each label category
- Identifies key words, phrases, and structural patterns
- Creates specific, actionable rules that capture labeling logic

### 📚 Comprehensive Guidelines Generation
- Generates complete annotation guidelines for human annotators
- Includes examples, counter-examples, and edge cases
- Provides disambiguation rules for handling ambiguous cases

### 🔄 Dynamic Rule Updates
- Updates existing rules when new labeled data is added
- Maintains rule versioning and change tracking
- Preserves institutional knowledge while adapting to new patterns

### 📊 Quality Assurance
- Analyzes rule coverage across your training data
- Identifies potential conflicts or inconsistencies
- Provides recommendations for improving annotation quality

## Quick Start

### 1. Generate Rules from Labeled Data

```python
from autolabeler import Settings, RuleGenerator
import pandas as pd

# Load your labeled data
df = pd.read_csv("labeled_data.csv")

# Initialize rule generator
settings = Settings(openrouter_api_key="your-key")
generator = RuleGenerator("my_dataset", settings)

# Generate rules
result = generator.generate_rules_from_data(
    df=df,
    text_column="text",
    label_column="label",
    task_description="Classify customer feedback sentiment",
    batch_size=50,
    min_examples_per_rule=3
)

print(f"Generated {len(result.ruleset.rules)} rules")
```

### 2. Export Human-Readable Guidelines

```python
# Export as Markdown for annotators
generator.export_ruleset_for_humans(
    result.ruleset,
    Path("annotation_guidelines.md"),
    format="markdown"
)

# Also available in HTML and JSON formats
generator.export_ruleset_for_humans(
    result.ruleset,
    Path("annotation_guidelines.html"),
    format="html"
)
```

### 3. Update Rules with New Data

```python
# Load new labeled examples
new_df = pd.read_csv("new_labeled_data.csv")

# Update existing rules
update_result = generator.update_rules_with_new_data(
    new_df=new_df,
    text_column="text",
    label_column="label"
)

print(f"Added {update_result.new_rules_added} new rules")
print(f"Modified {update_result.rules_modified} existing rules")
```

## CLI Usage

The rule generation feature is also available through the command line:

### Generate Rules
```bash
autolabeler generate-rules config.json labeled_data.csv \
    --text-column "text" \
    --label-column "sentiment" \
    --dataset-name "customer_reviews" \
    --task-description "Classify customer review sentiment" \
    --guidelines-file "guidelines.md"
```

### Update Existing Rules
```bash
autolabeler update-rules config.json new_data.csv \
    --text-column "text" \
    --label-column "sentiment" \
    --dataset-name "customer_reviews" \
    --guidelines-file "updated_guidelines.md"
```

### Export Guidelines
```bash
autolabeler export-rules customer_reviews \
    --output-file "guidelines.md" \
    --output-format "markdown"
```

## Rule Structure

Each generated rule contains:

- **Pattern Description**: Human-readable explanation of when the rule applies
- **Conditions**: Specific conditions that must be met
- **Indicators**: Key words, phrases, or patterns that signal the label
- **Examples**: Representative examples that demonstrate the rule
- **Counter-examples**: Examples that might seem similar but don't apply
- **Confidence Score**: Reliability measure for the rule
- **Frequency**: Number of training examples supporting the rule

Example rule:
```json
{
  "rule_id": "positive_sentiment_enthusiastic",
  "label": "positive",
  "pattern_description": "Enthusiastic positive language with strong emotional indicators",
  "conditions": [
    "Contains superlative adjectives (best, amazing, incredible)",
    "Uses exclamation marks or enthusiastic punctuation",
    "Expresses strong satisfaction or recommendation"
  ],
  "indicators": ["love", "amazing", "best", "excellent", "perfect"],
  "examples": [
    "I absolutely love this product!",
    "This is the best purchase I've made!"
  ],
  "confidence": 0.92,
  "frequency": 15
}
```

## Output Formats

### Markdown Guidelines
Perfect for sharing with annotation teams:
- Clear hierarchical structure
- Examples and counter-examples
- Quality assurance guidelines
- Easy to read and reference

### HTML Guidelines
Professional-looking documentation:
- Styled for better readability
- Interactive elements
- Suitable for web deployment

### JSON Format
Machine-readable for programmatic use:
- Complete rule definitions
- Metadata and statistics
- Integration with other tools

## Best Practices

### 1. Data Quality
- Ensure your training data is high-quality and consistent
- Include diverse examples for each label category
- Have at least 10-20 examples per label for meaningful rules

### 2. Rule Refinement
- Review generated rules for accuracy and completeness
- Test rules with new examples before deployment
- Regularly update rules as your data grows

### 3. Human Review
- Have domain experts review the generated guidelines
- Validate edge cases and disambiguation rules
- Incorporate feedback into rule updates

### 4. Iterative Improvement
- Start with initial rules and refine based on usage
- Monitor annotation consistency and update rules accordingly
- Use coverage analysis to identify areas needing more examples

## Integration with AutoLabeler

The generated rules integrate seamlessly with the AutoLabeler's existing features:

### RAG-Enhanced Labeling
```python
# The AutoLabeler can use both generated rules and RAG examples
labeler = AutoLabeler("my_dataset", settings)

# Add training data to knowledge base
labeler.add_training_data(df, "text", "label", source="human")

# Label new text with RAG examples
result = labeler.label_text("New text to label", use_rag=True)
```

### Ensemble Methods
```python
# Combine rule-based consistency with ensemble approaches
from autolabeler import EnsembleLabeler

ensemble = EnsembleLabeler("my_dataset", settings)
# Rules help ensure consistent patterns across all models
```

## Advanced Features

### Custom Rule Templates
Customize the rule generation process by modifying templates:
```python
# Use custom Jinja2 templates for rule generation
generator = RuleGenerator(
    "my_dataset",
    settings,
    template_path=Path("custom_rule_template.j2")
)
```

### Metadata Integration
Include additional context in rule generation:
```python
result = generator.generate_rules_from_data(
    df=df_with_metadata,
    text_column="text",
    label_column="label",
    # Additional metadata can inform rule generation
    task_description="Domain-specific sentiment analysis for product reviews"
)
```

### Rule Versioning
Track changes over time:
```python
# Rules are automatically versioned
current_rules = generator.load_latest_ruleset()
print(f"Current version: {current_rules.version}")

# Update creates new version
update_result = generator.update_rules_with_new_data(new_df, "text", "label")
print(f"Updated to version: {update_result.updated_ruleset.version}")
```

## Troubleshooting

### Common Issues

**Not enough examples for rule generation**
- Ensure you have at least `min_examples_per_rule` examples per label
- Consider lowering the threshold for initial exploration
- Add more diverse examples for underrepresented labels

**Rules too specific or too general**
- Adjust the `batch_size` parameter to control granularity
- Review and manually refine generated rules
- Use domain expertise to validate rule appropriateness

**Low confidence scores**
- Indicates inconsistent patterns in training data
- Review data quality and labeling consistency
- Consider additional training examples for problematic categories

### Performance Optimization

- Use larger batch sizes for faster processing of large datasets
- Set appropriate `min_examples_per_rule` to balance quality vs coverage
- Consider using corporate LLM endpoints for faster generation

## Examples

See the complete example in [`examples/rule_generation_example.py`](examples/rule_generation_example.py) for a full demonstration of the rule generation workflow.

The example shows:
- Generating rules from sample sentiment data
- Exporting guidelines for human annotators
- Updating rules with new examples
- Integration with the AutoLabeler for consistent labeling

## Future Enhancements

- **Active Learning Integration**: Identify examples that would most improve rules
- **Cross-Dataset Rule Transfer**: Apply rules across similar datasets
- **Conflict Resolution**: Automated resolution of conflicting rules
- **Performance Metrics**: Track annotation consistency improvements
- **Multi-language Support**: Generate rules for non-English datasets
