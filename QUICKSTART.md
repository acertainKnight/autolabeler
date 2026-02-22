# Quick Start

Get from zero to labeled data in 5 minutes.

## 1. Install

```bash
pip install -e .

# Optional: probe model support (transformers + torch)
pip install -e '.[probe]'
```

## 2. Set API Keys

```bash
# Option A: One key for everything via OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# Option B: Direct provider keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## 3. Set Up Your Dataset

```bash
# Copy the template config and prompts
cp configs/example.yaml configs/my_dataset.yaml
cp -r prompts/example prompts/my_dataset

# Edit configs/my_dataset.yaml: set name, labels, text_column, jury models
# Edit prompts/my_dataset/system.md and rules.md: replace [PLACEHOLDER] fields
```

## 4. Label Data

```bash
python scripts/run_labeling.py \
    --dataset my_dataset \
    --input datasets/my_data.csv \
    --output outputs/my_dataset/labeled.csv
```

Output CSV includes: `label`, `tier` (ACCEPT/SOFT/QUARANTINE), `training_weight`, `soft_label`, `jury_labels`, `jury_confidences`.

## 5. Run Diagnostics

```bash
python scripts/run_diagnostics.py \
    --dataset my_dataset \
    --labeled-data outputs/my_dataset/labeled.csv \
    --output outputs/my_dataset/diagnostics/
```

Review `quality_report.md` for issues, `top_suspects.csv` for samples to audit.

## 6. Export for Training

```bash
python scripts/export_for_distillation.py \
    --llm-labels outputs/my_dataset/labeled.csv \
    --output outputs/my_dataset/distillation.jsonl
```

## 7. Train a Probe Model (Optional)

Quick local evaluation before full cloud training:

```bash
python scripts/train_probe.py \
    --dataset my_dataset \
    --training-data outputs/my_dataset/distillation.jsonl
```

## What's Next

| I want to... | Read |
|-------------- |------|
| Understand the full pipeline | [README.md](README.md) |
| Tune diagnostics or embedding settings | [docs/configuration.md](docs/configuration.md) |
| Find patterns in classifier errors | [docs/gap-analysis.md](docs/gap-analysis.md) |
| Iterate on data quality with the probe | [docs/iteration-workflow.md](docs/iteration-workflow.md) |
| Add a new dataset | [README.md#adding-a-new-dataset](README.md#adding-a-new-dataset) |
| Improve prompts with DSPy | [README.md#prompt-optimization-dspy](README.md#prompt-optimization-dspy) |
| Scale labeling with heuristics | [README.md#alchemist-program-generation](README.md#alchemist-program-generation) |
