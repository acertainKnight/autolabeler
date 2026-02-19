# Autolabeler: Unified Data Labeling Automation Service

A production-ready LLM-powered labeling service with heterogeneous jury voting, confidence calibration, and dataset-agnostic architecture.

## What's New (v2.0 - February 2026)

**Complete architectural rebuild** focused on:
- ✅ **One pipeline, any dataset**: Configure via YAML, not code
- ✅ **Heterogeneous jury voting**: Mix Anthropic, OpenAI, Google models for diversity
- ✅ **Logprob-based confidence**: Extract true confidence from token probabilities (OpenAI)
- ✅ **Structured prompts**: Rules, examples, and mistakes in readable markdown files
- ✅ **Clean codebase**: Archived 8 experimental features, organized 33+ misplaced files

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Set API Keys

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### 3. Label Data

```bash
# Fed headlines (6-class ordinal)
python scripts/run_labeling.py \
    --dataset fed_headlines \
    --input datasets/fed_data.csv \
    --output outputs/labeled.csv \
    --budget 10.0

# Trade Policy Uncertainty (binary)
python scripts/run_labeling.py \
    --dataset tpu \
    --input datasets/articles.jsonl \
    --output outputs/labeled.csv \
    --budget 10.0
```

## Architecture

```
Input Data
    ↓
[Stage 1: Optional Relevancy Gate]
    ↓
[Stage 2: Heterogeneous Jury]
  • Claude Sonnet 4.5
  • GPT-4o (with logprobs)
  • Gemini 2.5 Pro
    ↓
[Stage 3: Confidence-Weighted Vote]
    ↓
[Stage 4: Optional Candidate Annotation]
  (for jury disagreements)
    ↓
[Stage 5: Tier Assignment]
  • ACCEPT (unanimous, weight=1.0)
  • ACCEPT-M (majority, weight=0.85)
  • SOFT (soft labels, weight=0.7)
  • QUARANTINE (unresolved, weight=0.0)
    ↓
Labeled Output
```

## Adding a New Dataset

1. **Create prompt files:**
```bash
mkdir prompts/my_dataset
# Create: system.md, rules.md, examples.md, mistakes.md
```

2. **Create config:**
```yaml
# configs/my_dataset.yaml
name: my_dataset
labels: ["0", "1", "2"]
text_column: text
input_format: csv

jury_models:
  - provider: openai
    model: gpt-4o
    name: GPT-4o
    has_logprobs: true
  - provider: anthropic
    model: claude-sonnet-4-5-20250929
    name: Claude
```

3. **Run:**
```bash
python scripts/run_labeling.py --dataset my_dataset --input data.csv --output labeled.csv
```

## Key Features

### Heterogeneous Jury
- Mix models from different providers for genuine diversity
- Different training data = different error patterns = better ensemble
- Confidence-weighted voting (not simple majority)

### Advanced Confidence Scoring
- **Logprob extraction** (OpenAI): Extract P(label | text) from token probabilities
- **Self-consistency** (Claude, Gemini): Sample n times, measure agreement
- **Isotonic calibration**: Post-hoc calibration for reliable confidence scores

### Structured Prompts
Prompts are **markdown files**, not JSON arrays:
- Easy to read and edit
- Easy to diff in PRs
- Easy for domain experts (not just engineers) to review
- Version-controlled alongside code

Example structure:
```
prompts/fed_headlines/
  ├── system.md       # Domain expertise, role definition
  ├── rules.md        # 10-rule classification framework
  ├── examples.md     # 50+ calibration examples with reasoning
  ├── mistakes.md     # Common errors to avoid
  └── candidate.md    # Prompt for disagreement resolution
```

### Resume and Checkpointing
- `--resume`: Resume from existing output, label only missing rows
- `--resume-until-complete`: Loop until all rows have labels
- Checkpoint after each batch (default: 10 rows)
- Safe budget enforcement per model

## Datasets

### Fed Headlines (Monetary Policy Classification)
- **Task**: Classify Federal Reserve headlines on hawk-dove spectrum
- **Labels**: -2 (explicitly dovish) to +2 (explicitly hawkish), 0 (neutral), -99 (not relevant)
- **Challenges**: ~45-55% should be neutral, adjacent-class ambiguity (0 vs 1, -1 vs 0)
- **Pipeline**: All stages enabled (relevancy gate, jury, candidate annotation)

### Trade Policy Uncertainty (TPU)
- **Task**: Detect uncertainty about trade policy in news articles
- **Labels**: 0 (no TPU), 1 (TPU present)
- **Challenges**: Must identify causal link between trade policy and uncertainty
- **Pipeline**: Jury + aggregation only (simpler for binary task)

## Project Structure

```
autolabeler/
├── configs/              # Dataset YAML configs
│   ├── fed_headlines.yaml
│   └── tpu.yaml
├── prompts/              # Structured prompt files
│   ├── fed_headlines/
│   └── tpu/
├── src/autolabeler/
│   └── core/
│       ├── prompts/         # PromptRegistry
│       ├── llm_providers/   # Multi-provider abstraction
│       ├── labeling/        # LabelingPipeline
│       ├── quality/         # ConfidenceScorer, Calibrator
│       └── dataset_config.py
├── scripts/
│   ├── run_labeling.py      # Main entry point
│   └── legacy/              # Old scripts (reference)
└── tests/

```

## Development Status

**Completed:**
- ✅ Repo cleanup (Phase 1)
- ✅ Prompt/rules management (Phase 2)
- ✅ LLM provider abstraction (Phase 3, partial)
- ✅ Dataset configuration system

**In Progress:**
- ⏳ ConfidenceScorer
- ⏳ LabelingPipeline
- ⏳ run_labeling.py entry point

See [IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md) for details.

## Evidence Base (v4 Architecture)

This architecture is based on recent NLP research:

1. **No debate/adjudicator** (NeurIPS 2025): Multi-round debate offers no accuracy gains over simple majority voting
2. **Heterogeneous models** (A-HMAD): Agent diversity (different training data) yields 4-6% gains over same-model ensembles
3. **Candidate annotation** (CanDist, ACL 2025): Soft labels for ambiguous cases outperform forced hard labels
4. **Logprob confidence** (Amazon 2024): Token logprobs + calibration reduce ECE by ~46% vs verbal confidence

Expected improvements for Fed Headlines:
- Neutral rate: 31% → 45-55%
- Dev accuracy: 74% → 83-88%
- Train/dev gap: 17pts → <5pts

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
