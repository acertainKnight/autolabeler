# Probe Model Guide

The probe model is a lightweight local RoBERTa classifier that you fine-tune on your distillation export to get fast performance feedback. It's not the production model — it's a cheap, quick way to measure training-data quality before you commit to the full cloud training pipeline.

## Why a Probe Model?

The full training pipeline is expensive: cloud GPUs, hours of training time, and a long feedback loop. The probe model gives you a 5-minute answer to "is my training data good enough yet?"

If the probe can't learn your labels, the full model won't either. Fix the data first.

## Prerequisites

Install the probe dependency group:

```bash
pip install -e '.[probe]'
```

This pulls in `transformers`, `torch`, `accelerate`, and `datasets`. These are not needed for the rest of the pipeline and are kept optional.

You also need a distillation JSONL file (from `scripts/export_for_distillation.py`).

## Quick Start

```bash
# Basic training (uses probe: block from config YAML)
python scripts/train_probe.py \
    --dataset fed_headlines \
    --training-data outputs/fed_headlines/distillation.jsonl

# Override model and hyperparams via CLI
python scripts/train_probe.py \
    --dataset fed_headlines \
    --training-data outputs/fed_headlines/distillation.jsonl \
    --model distilroberta-base \
    --epochs 3 \
    --batch-size 64 \
    --output outputs/fed_headlines/probe_v2/
```

Output:
```
============================================================
PROBE MODEL EVALUATION -- FED_HEADLINES
============================================================
  Accuracy                       0.7823
  Macro F1                       0.6145
  Weighted F1                    0.7801
  Cohen's Kappa                  0.6892
  Mean Abs Error (ordinal)       0.3120
  Spearman ρ (ordinal)           0.8234
============================================================
```

## Configuration

The `probe:` block in your dataset YAML defines defaults:

```yaml
probe:
  model: "roberta-base"        # HuggingFace model ID
  epochs: 5                    # training epochs
  batch_size: 32               # per-device batch size
  learning_rate: 2.0e-5        # AdamW peak LR
  val_split: 0.2               # stratified validation fraction
  max_length: 128              # tokenizer truncation length
  warmup_ratio: 0.1            # LR warmup fraction
  weight_decay: 0.01           # L2 regularisation
  use_training_weights: true   # use distillation tier weights in loss
```

All values can be overridden from the CLI. See `python scripts/train_probe.py --help`.

### Model Selection

| Model | Params | Size | Training time (20k samples) | When to use |
|-------|--------|------|-----------------------------|-------------|
| `distilroberta-base` | ~82M | ~330 MB | ~3 min CPU / ~30s GPU | Quick smoke tests |
| `roberta-base` | ~125M | ~500 MB | ~5 min CPU / ~1 min GPU | Default, good balance |
| `roberta-large` | ~355M | ~1.3 GB | ~15 min CPU / ~3 min GPU | Higher ceiling, needs more VRAM |

### Training Weights

When `use_training_weights: true` (the default), the probe uses the `training_weight` column from the distillation JSONL as per-sample loss weights:

| Tier | Training weight | Effect |
|------|----------------|--------|
| ACCEPT (verified) | 1.0 | Full influence |
| ACCEPT (unverified) | 0.9 | Near-full |
| ACCEPT-M (candidate) | 0.7 | Moderate |
| SOFT | 0.5 | Reduced |
| QUARANTINE | 0.0 | Excluded at export |
| Human | 1.2 | Highest influence |

Disable with `--no-weights` to treat all samples equally.

## Evaluating a Saved Model

Re-evaluate a previously trained probe on new or updated data without retraining:

```bash
python scripts/train_probe.py \
    --dataset fed_headlines \
    --eval-only \
    --model-dir outputs/fed_headlines/probe \
    --training-data outputs/fed_headlines/distillation_v2.jsonl
```

This is useful for comparing how the same model performs before and after data fixes.

## Output Files

Training writes the following to the output directory (default: `outputs/{dataset}/probe/`):

| File | Contents |
|------|----------|
| `config.json` | HuggingFace model config |
| `model.safetensors` | Model weights |
| `tokenizer.json` | Tokenizer |
| `val_predictions.csv` | Per-sample validation predictions (text, true_label, pred_label) |
| `eval_metrics.json` | Full evaluation metrics (accuracy, F1, confusion matrix, ordinal metrics) |
| `probe_summary.json` | Compact summary for quick comparison across runs |

## Metrics

The probe uses the same `EvaluationService` as the rest of the project, so metrics are directly comparable to full cloud model evaluations:

- **Accuracy**: overall correctness
- **Macro F1**: unweighted average across classes (sensitive to rare classes)
- **Weighted F1**: class-frequency-weighted average
- **Cohen's Kappa**: agreement adjusted for chance
- **MAE / Spearman rho** (ordinal tasks): how well the model preserves label ordering
- **3-class metrics** (Fed headlines): accuracy and F1 after collapsing to dove/neutral/hawk
- **Confusion matrix**: per-class breakdown

## Mixed Precision (GPU)

If you have a CUDA GPU, enable fp16 for faster training and lower VRAM usage:

```bash
python scripts/train_probe.py \
    --dataset fed_headlines \
    --training-data outputs/fed_headlines/distillation.jsonl \
    --fp16
```

On an RTX 3050 (4 GB VRAM), `roberta-base` with `batch_size=32` and `fp16` trains comfortably. Lower batch size to 16 without fp16.

## Programmatic Usage

```python
from autolabeler.core.probe import ProbeTrainer, ProbeConfig
from autolabeler.core.dataset_config import DatasetConfig

ds_cfg = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
probe_cfg = ProbeConfig(model_name="roberta-base", epochs=3)

trainer = ProbeTrainer(probe_cfg, ds_cfg)
results = trainer.train("outputs/fed_headlines/distillation.jsonl")

print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"Macro F1: {results['metrics']['f1_macro']:.4f}")
print(f"Model saved to: {results['model_path']}")

# Later: evaluate saved model on new data
metrics = trainer.evaluate_saved(
    "outputs/fed_headlines/probe",
    "outputs/fed_headlines/distillation_v2.jsonl",
)
```

## CLI Reference

```
python scripts/train_probe.py --help

Required:
  --dataset          Dataset name (matches configs/{dataset}.yaml)
  --training-data    Path to distillation JSONL

Training options:
  --model            HuggingFace model ID (overrides config)
  --epochs           Training epochs (overrides config)
  --batch-size       Per-device batch size (overrides config)
  --lr               Learning rate (overrides config)
  --max-length       Tokenizer max length (overrides config)
  --no-weights       Disable per-sample training weights
  --fp16             Mixed-precision training (CUDA only)
  --output           Output directory (default: outputs/{dataset}/probe/)

Evaluation:
  --eval-only        Skip training; evaluate a saved model
  --model-dir        Path to saved model for --eval-only

Other:
  --verbose / -v     DEBUG logging
```

## Tips

- **Iterate fast**: use `distilroberta-base` with 2 epochs for a 2-minute sanity check, then switch to `roberta-base` with 5 epochs for the "real" run.
- **Compare runs**: keep `probe_summary.json` from each run. The key metric to watch is macro F1 — it's the most sensitive to improvements in underrepresented classes.
- **Don't overfit the probe**: if val loss goes up while train loss goes down, reduce epochs. The probe is a thermometer, not the patient.
- **Pair with gap analysis**: the gap report tells you *what* to fix; the probe tells you whether the fix *worked*.
