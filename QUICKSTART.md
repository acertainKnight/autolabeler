# Unified Labeling Pipeline - Quick Start Guide

## Overview

The autolabeler now uses a **unified, dataset-agnostic pipeline** configured via YAML files. One pipeline handles all classification tasks (FedSpeak, TPU, or any new dataset).

## Evidence-Based Architecture

Based on recent research:
- **Heterogeneous Jury** (NeurIPS 2025, A-HMAD): Diverse models improve disagreement resolution
- **Confidence-Weighted Voting** (Amazon 2024): Logprob-based confidence reduces ECE by 46%
- **Candidate Annotation** (ACL 2025): Soft labels for ambiguous cases
- **Tier System**: ACCEPT (unanimous) → ACCEPT-M (majority) → SOFT (ambiguous) → QUARANTINE (failed)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Or with optional dependencies:
pip install -e ".[all]"
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

### 3. Run Labeling

#### Fed Headlines (6-class ordinal)

```bash
python scripts/run_labeling.py \
    --dataset fed_headlines \
    --input datasets/headlines.csv \
    --output outputs/fed_headlines/labeled.csv \
    --limit 100
```

#### TPU (binary classification)

```bash
python scripts/run_labeling.py \
    --dataset tpu \
    --input datasets/tpu_articles.csv \
    --output outputs/tpu/labeled.csv \
    --resume
```

### 4. Review Results

Output CSV contains:
- `label`: Primary label (most likely)
- `label_type`: "hard" or "soft"
- `tier`: ACCEPT, ACCEPT-M, SOFT, or QUARANTINE
- `training_weight`: Weight for training (0-1)
- `agreement`: Agreement type (unanimous, majority_adjacent, etc.)
- `jury_labels`: Individual jury votes (JSON)
- `jury_confidences`: Confidence scores (JSON)
- `soft_label`: Soft label distribution for ambiguous cases (JSON)

## Configuration

### Dataset Config (YAML)

Create `configs/{dataset}.yaml`:

```yaml
name: my_dataset
labels: ["0", "1", "2"]
text_column: text
label_column: label
batch_size: 10

jury_models:
  - name: gpt-4o-mini
    provider: openai
    model: gpt-4o-mini
    has_logprobs: true
    
  - name: claude-3-5-haiku
    provider: anthropic
    model: claude-3-5-haiku-20241022
    has_logprobs: false

jury_temperature: 0.0
use_relevancy_gate: false
use_candidate_annotation: true
candidate_temperature: 0.3
```

### Prompts (Markdown)

Create `prompts/{dataset}/`:
- `system.md` - System prompt (role, expertise, context)
- `rules.md` - Classification rules and framework
- `examples.md` - Calibration examples with reasoning
- `mistakes.md` - Common mistakes to avoid
- `candidate.md` - Candidate annotation prompt (optional)

See `prompts/_template/` for templates.

## Architecture

### 5-Stage Pipeline

1. **Optional Relevancy Gate** - Cheap pre-filter (not implemented in MVP)
2. **Heterogeneous Jury** - Parallel calls to 3-6 diverse models
3. **Confidence-Weighted Aggregation** - Logprobs or self-consistency
4. **Optional Candidate Annotation** - Soft labels for disagreements
5. **Tier Assignment** - ACCEPT/ACCEPT-M/SOFT/QUARANTINE

### Key Components

```python
from autolabeler.core.dataset_config import DatasetConfig
from autolabeler.core.prompts.registry import PromptRegistry
from autolabeler.core.labeling.pipeline import LabelingPipeline
from autolabeler.core.quality.confidence_scorer import ConfidenceScorer

# Load config and prompts
config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
prompts = PromptRegistry("fed_headlines")

# Initialize pipeline
pipeline = LabelingPipeline(config, prompts)

# Label one text
result = await pipeline.label_one("FED SEES RATES RISING")
print(result.label, result.tier)

# Label dataframe
import pandas as pd
df = pd.read_csv("data.csv")
results_df = await pipeline.label_dataframe(df, "output.csv")
```

## Command-Line Options

```bash
python scripts/run_labeling.py \
    --dataset DATASET      # Dataset name (matches config file)
    --input INPUT          # Input CSV file
    --output OUTPUT        # Output CSV file
    --resume               # Resume from existing output
    --batch-size N         # Batch size (overrides config)
    --limit N              # Limit to first N rows (for testing)
    --max-budget AMOUNT    # Max budget in USD (TODO: implement)
    --verbose              # Enable debug logging
```

## Adding a New Dataset

1. **Create config**: `configs/my_dataset.yaml`
2. **Create prompts**: `prompts/my_dataset/system.md`, `rules.md`, `examples.md`, `mistakes.md`
3. **Run labeling**: `python scripts/run_labeling.py --dataset my_dataset ...`

That's it! The unified pipeline handles everything else.

## Tier Definitions

- **ACCEPT** (weight=1.0): Unanimous jury agreement
- **ACCEPT-M** (weight=0.85): Majority with adjacent disagreement
- **SOFT** (weight=0.7): Ambiguous cases with soft labels
- **QUARANTINE** (weight=0.0): Unresolved disagreements or failures

## Confidence Scoring

### Logprobs (OpenAI)
- Extract P(label | text) directly from token probabilities
- Most reliable, lowest ECE (~0.05-0.07)

### Self-Consistency (Claude, Gemini)
- Sample n times with higher temperature
- Agreement rate = confidence
- Reliable fallback

### Verbal Fallback
- Maps "high/medium/low" to 0.9/0.7/0.5
- Known to be miscalibrated (ECE 0.13-0.43)

### Calibration
- Isotonic regression on held-out data
- Improves reliability, reduces ECE by ~46%

## Budget & Cost Tracking

TODO: Integrate with existing cost tracking utilities in `src/autolabeler/core/utils/`.

## Testing

Run tests:

```bash
pytest tests/test_unified_pipeline.py -v
```

## Migration from Old Scripts

Old scripts in `scripts/legacy/` (e.g., `run_tpu_multi_llm_voting.py`, `run_phase0_unified.py`) are preserved for reference but will be deprecated.

Use the new unified entry point:

```bash
# Old way (deprecated)
python run_tpu_multi_llm_voting.py --input data.csv --output out.csv

# New way (recommended)
python scripts/run_labeling.py --dataset tpu --input data.csv --output out.csv
```

## Troubleshooting

### Missing API Keys
Ensure all required API keys are set:
```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

### Config Not Found
Check that `configs/{dataset}.yaml` exists. List available configs:
```bash
ls configs/*.yaml
```

### Prompts Not Loading
Check that `prompts/{dataset}/` directory exists with required files:
- system.md
- rules.md
- examples.md
- mistakes.md

### Import Errors
Install dependencies:
```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Support

See main `README.md` for architecture details and research references.
