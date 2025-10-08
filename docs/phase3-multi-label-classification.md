# Phase 3: Multi-Label Classification with Rule Evolution

## Overview

Phase 3 implements multi-label classification where each observation receives multiple simultaneous classifications (e.g., relevancy AND sentiment together) with automatic rule improvement through active learning.

## Key Features

- **Multi-Label Classification**: Apply multiple classification tasks to each observation in a single pass
- **Single LLM Call Optimization**: All labels determined in one API call with structured JSON output
- **Constitutional AI**: Rule-based constraints and principles enforcement
- **Rule Evolution**: Automatic improvement of classification rules based on uncertain predictions
- **Batch Processing**: Configurable batch sizes with progress tracking
- **Confidence-Based Learning**: Identifies low-confidence predictions for rule refinement

## Architecture

### Core Services

#### 1. MultiAgentService (`src/autolabeler/agents.py`)

Coordinates multiple classification tasks simultaneously with a single LLM call.

**Key Methods:**
- `label_single(text, tasks)`: Classify single text with multiple tasks
- `label_with_agents(df, text_column, tasks)`: Batch classification for DataFrame
- `update_task_configs(updated_rules)`: Dynamic rule updates during processing

**Output Format:**
For each task, creates three columns:
- `label_{task_name}`: Predicted label
- `confidence_{task_name}`: Confidence score (0.0-1.0)
- `reasoning_{task_name}`: Explanation of classification decision

#### 2. ConstitutionalService (`src/autolabeler/constitutional.py`)

Manages and enforces labeling principles across classification tasks.

**Enforcement Levels:**
- **Strict**: Warns if confidence < 0.9
- **Moderate**: Warns if confidence < 0.7
- **Lenient**: Warns if confidence < 0.5

**Key Methods:**
- `update_principles(new_principles)`: Update rules for tasks
- `validate_label(task, label, text, confidence)`: Check label against principles
- `export_principles()` / `import_principles()`: Persistence support

#### 3. RuleEvolutionService (`src/autolabeler/active_learning.py`)

Improves classification rules based on feedback and error patterns.

**Error Pattern Types:**
- **low_confidence**: Predictions with confidence < 0.7
- **edge_case**: Borderline predictions (0.5-0.7 confidence)

**Key Methods:**
- `identify_error_patterns(feedback_data, task_names)`: Analyze uncertain predictions
- `improve_rules(current_rules, feedback_data)`: Generate improved rules via LLM
- `get_improvement_stats()`: Track improvement metrics

## CLI Usage

### Command: `label-multi`

```bash
autolabeler label-multi \
    --dataset-name "customer_feedback" \
    --input-file data.csv \
    --output-file labeled.csv \
    --text-column "feedback_text" \
    --tasks "relevancy,sentiment,urgency" \
    --task-configs task_configs.json \
    --enable-rule-evolution \
    --batch-size 50 \
    --confidence-threshold 0.7 \
    --enforcement-level strict
```

### Required Options

- `--dataset-name`: Unique project identifier
- `--input-file`: Path to input CSV or Parquet file
- `--output-file`: Path for labeled output
- `--text-column`: Column containing text to classify
- `--tasks`: Comma-separated task names (e.g., "relevancy,sentiment")
- `--task-configs`: JSON file with task configurations

### Optional Options

- `--enable-rule-evolution`: Enable automatic rule improvement (flag)
- `--batch-size`: Rows per batch (default: 50)
- `--confidence-threshold`: Threshold for uncertain predictions (default: 0.7)
- `--enforcement-level`: Principle enforcement (strict/moderate/lenient, default: strict)
- `--model-name`: Override LLM model
- `--temperature`: LLM temperature (default: 0.1)

## Task Configuration File

The `--task-configs` JSON file defines labels and initial principles for each task:

```json
{
  "relevancy": {
    "labels": ["relevant", "not_relevant", "partially_relevant"],
    "principles": [
      "Text must directly address the main topic to be relevant",
      "Tangential mentions are partially_relevant",
      "Off-topic content is not_relevant"
    ]
  },
  "sentiment": {
    "labels": ["positive", "negative", "neutral"],
    "principles": [
      "Consider overall tone and emotional content",
      "Look for explicit sentiment indicators",
      "Neutral applies when no clear sentiment is present"
    ]
  },
  "urgency": {
    "labels": ["urgent", "normal", "low_priority"],
    "principles": [
      "Urgent requires immediate action or time-sensitive content",
      "Normal is standard business priority",
      "Low_priority can be addressed later"
    ]
  }
}
```

## Workflow

### Basic Multi-Label Classification

1. Load data and task configurations
2. Initialize MultiAgentService with task configs
3. Process all observations with single LLM call per text
4. Output DataFrame with label/confidence/reasoning columns for each task

### With Rule Evolution Enabled

1. Load data and task configurations
2. Initialize all three services (MultiAgent, Constitutional, RuleEvolution)
3. **For each batch:**
   - Classify observations using current rules
   - Identify uncertain predictions (confidence < threshold)
   - If uncertain predictions exist:
     - Generate improved rules via LLM
     - Update ConstitutionalService with new rules
     - Update MultiAgentService task configs
   - Continue to next batch with improved rules
4. Output final labeled DataFrame with improvement statistics

## Example Output

Input: 100 customer feedback messages

Output columns for 3 tasks (relevancy, sentiment, urgency):
```
autolabeler_id | feedback_text | label_relevancy | confidence_relevancy | reasoning_relevancy |
               |               | label_sentiment | confidence_sentiment | reasoning_sentiment |
               |               | label_urgency   | confidence_urgency   | reasoning_urgency
```

## Performance Characteristics

### Single LLM Call Optimization

**Traditional Approach** (multiple calls):
- 3 tasks × 100 observations = 300 LLM API calls
- Higher cost, slower processing, separate context per task

**Phase 3 Approach** (single call):
- 100 observations = 100 LLM API calls
- Lower cost, faster processing, shared context across tasks
- Structured JSON response ensures consistent format

### Rule Evolution Impact

- **Initial Batch**: Uses provided principles
- **Subsequent Batches**: Benefits from learned rules
- **Typical Pattern**: Confidence scores improve 5-15% over batches
- **Rule Generation**: ~5 new rules per improvement cycle

## Integration with Existing Features

### Compatible with AutoLabelerV2 Features:

- ✅ UUID-based progress tracking (`autolabeler_id`)
- ✅ CSV and Parquet file formats
- ✅ Batch processing with configurable sizes
- ✅ Settings-based LLM configuration (Anthropic/OpenAI)
- ✅ Comprehensive logging

### Not Compatible:

- ❌ RAG (Retrieval-Augmented Generation) - use `label` command instead
- ❌ Ensemble methods - use `label_ensemble` command instead
- ❌ Single-task classification - use `label` command for simpler cases

## Best Practices

1. **Start Small**: Test with 50-100 rows before processing full datasets
2. **Task Definition**: Keep tasks independent and clearly defined
3. **Initial Principles**: Provide 3-5 clear principles per task
4. **Batch Size**: 50-100 rows optimal for rule evolution balance
5. **Confidence Threshold**: 0.7 is recommended; adjust based on domain requirements
6. **Rule Evolution**: Enable for datasets > 200 rows where patterns may emerge
7. **Enforcement Level**: Start with "strict" for high-stakes applications

## Troubleshooting

### Low Confidence Scores Across Tasks

- Review and refine initial principles
- Lower temperature for more consistent predictions
- Add more specific guidance in principles
- Enable rule evolution to learn from patterns

### Rule Evolution Not Improving

- Check if enough uncertain predictions exist (< threshold)
- Increase batch size for better pattern identification
- Review generated rules in logs
- Consider if tasks are too subjective for automatic improvement

### JSON Parsing Errors

- Check LLM model supports structured output
- Increase max_tokens if responses are truncated
- Review temperature setting (too high = inconsistent format)

## Implementation Details

### File Locations

- CLI command: `src/autolabeler/cli.py` (line 320-544)
- MultiAgentService: `src/autolabeler/agents.py`
- ConstitutionalService: `src/autolabeler/constitutional.py`
- RuleEvolutionService: `src/autolabeler/active_learning.py`

### Dependencies

```python
from .agents import MultiAgentService
from .constitutional import ConstitutionalService
from .active_learning import RuleEvolutionService
```

All services require `Settings` for LLM configuration.

## Future Enhancements

Potential improvements for Phase 4+:

- Async/parallel batch processing
- Real-time rule evolution monitoring dashboard
- Export/import learned rules for reuse
- A/B testing of rule sets
- Multi-model ensemble for uncertainty estimation
- Active learning with human-in-the-loop validation
