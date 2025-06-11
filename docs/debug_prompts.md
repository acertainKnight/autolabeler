# Debug Prompt Storage

The AutoLabeler now includes a debug prompt storage feature that automatically stores the last 10 rendered prompts for debugging purposes. This helps developers understand exactly what prompts are being sent to the LLM and troubleshoot any issues.

## Features

- **Automatic Storage**: The last 10 rendered prompts are automatically stored in memory using a circular buffer (deque)
- **Persistent Storage**: Debug prompts are saved to JSON files for later inspection
- **Rich Metadata**: Each stored prompt includes:
  - Prompt ID
  - Input text
  - Rendered prompt
  - Template source
  - Number of examples used
  - Ruleset (if applicable)
  - Timestamp
  - Additional context-specific metadata

## Usage

### In LabelingService

```python
from autolabeler.core.labeling.labeling_service import LabelingService

# The service automatically stores debug prompts
service = LabelingService(dataset_name, settings)

# Label some text
response = service.label_text("This is a test")

# Get the last 10 debug prompts
debug_prompts = service.get_debug_prompts()

# Save debug prompts to a file
debug_file = service.save_debug_prompts()
print(f"Debug prompts saved to: {debug_file}")
```

### In SyntheticGenerationService

```python
from autolabeler.core.generation.synthetic_service import SyntheticGenerationService

# The service automatically stores debug prompts
service = SyntheticGenerationService(dataset_name, settings)

# Generate synthetic examples
examples = service.generate_for_label("positive", num_examples=5)

# Get and save debug prompts
debug_prompts = service.get_debug_prompts()
debug_file = service.save_debug_prompts()
```

## Debug Prompt Files

Debug prompts are saved to:
- Labeling Service: `results/labeling_service/{dataset_name}/debug_prompts.json`
- Synthetic Generation Service: `results/synthetic_service/{dataset_name}/debug_prompts.json`

## Viewing Debug Prompts

Use the provided utility script to view debug prompts in a formatted way:

```bash
# View all prompts with preview
python scripts/utilities/view_debug_prompts.py results/labeling_service/my_dataset/debug_prompts.json

# View only first 5 prompts
python scripts/utilities/view_debug_prompts.py results/labeling_service/my_dataset/debug_prompts.json -n 5

# View with full prompt text
python scripts/utilities/view_debug_prompts.py results/labeling_service/my_dataset/debug_prompts.json -f
```

## Example Debug Prompt Structure

```json
{
  "dataset_name": "test_debug_prompts",
  "timestamp": "2025-06-10T23:08:49.976421",
  "prompts_count": 5,
  "prompts": [
    {
      "prompt_id": "7fa4eef43b9a297c",
      "text": "This product is absolutely amazing!",
      "rendered_prompt": "You are an expert automated labeling assistant...",
      "template_source": "<template>",
      "examples_count": 0,
      "ruleset": {
        "example_rule": "Rule for text 1"
      },
      "timestamp": "2025-06-10T23:08:49.959413"
    }
  ]
}
```

## Benefits

1. **Debugging**: Easily inspect what prompts are being sent to the LLM
2. **Optimization**: Analyze prompts to improve templates and reduce token usage
3. **Auditing**: Keep a record of prompts for compliance or review
4. **Development**: Test prompt templates without making actual LLM calls

## Implementation Notes

- The feature uses Python's `collections.deque` with `maxlen=10` to automatically maintain only the last 10 prompts
- Debug prompts are saved automatically after each storage operation
- The feature has minimal performance impact as it only stores prompt metadata
- Both synchronous and asynchronous prompt preparation methods are supported
