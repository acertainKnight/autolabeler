# Diagnostics Guide

The diagnostics pipeline detects potential labeling errors and data quality issues without requiring ground-truth labels. It works on both LLM-pipeline output (with jury votes, confidence scores, and tiers) and plain human-labeled CSVs.

## When to Use Diagnostics

- **After labeling**: catch systematic errors before you invest in model training.
- **After human review**: validate that manual corrections didn't introduce new problems.
- **Before distillation export**: identify samples that should be quarantined or relabeled.
- **Iteratively**: run diagnostics, fix the worst issues, re-run, compare reports.

## Quick Start

```bash
# Full diagnostics on pipeline output
python scripts/run_diagnostics.py \
    --dataset fed_headlines \
    --labeled-data outputs/fed_headlines/labeled.csv \
    --output outputs/fed_headlines/diagnostics/

# Human-labeled data with non-default column names
python scripts/run_diagnostics.py \
    --dataset fed_headlines \
    --labeled-data datasets/fedspeak/human_labeled.csv \
    --output outputs/fed_headlines/diagnostics_human/ \
    --text-column headline \
    --label-column hawk_dove

# Selective modules (skip NLI to avoid model download)
python scripts/run_diagnostics.py \
    --dataset fed_headlines \
    --labeled-data outputs/fed_headlines/labeled.csv \
    --output outputs/fed_headlines/diagnostics/ \
    --enable embedding,distribution,report

# Quick test on a small subset
python scripts/run_diagnostics.py \
    --dataset fed_headlines \
    --labeled-data outputs/fed_headlines/labeled.csv \
    --output outputs/fed_headlines/diagnostics_test/ \
    --limit 500 --enable embedding,report
```

## Diagnostic Modules

### Embedding Analysis (`embedding`)

Uses sentence embeddings to detect spatial anomalies in the label space.

**What it finds:**
- **Intra-class outliers**: samples whose embedding is far from their class centroid (z-score above threshold). Often indicates mislabeled samples or edge cases.
- **Centroid violations**: samples that are closer to a different class's centroid than their own. Strong signal for label errors.
- **Cluster fragmentation**: classes that split into multiple disconnected clusters in embedding space. May indicate the label definition covers distinct sub-topics.
- **Near-duplicates**: text pairs with cosine similarity above threshold. Can inflate metrics and bias training.

**Embedding providers** (set in config):

| Provider | Config value | Model example | Notes |
|----------|-------------|---------------|-------|
| Local (sentence-transformers) | `local` | `all-MiniLM-L6-v2` | Free, runs on CPU, ~100ms/batch |
| OpenAI | `openai` | `text-embedding-3-small` | Fast API, good quality |
| OpenRouter | `openrouter` | `openai/text-embedding-3-small` | Same quality, single API key |

**Key config options:**
```yaml
diagnostics:
  embedding_provider: "openrouter"
  embedding_model: "openai/text-embedding-3-small"
  embedding_batch_size: 2048
  outlier_z_threshold: 2.0       # higher = fewer flags
  lof_neighbors: 20
  fragmentation_min_cluster_size: 5
  fragmentation_umap_dim: 10     # UMAP dims before HDBSCAN
  duplicate_similarity_threshold: 0.95
```

**Output files:**
- `suspicion_scores.csv` — every sample with its composite suspicion score
- `top_suspects.csv` — the top-K most suspicious samples
- `umap_clusters.html` — interactive UMAP scatter plot coloured by label

### Distribution Analysis (`distribution`)

Statistical analysis of the label distribution, text length patterns, and potential biases.

**What it finds:**
- Class imbalance (e.g. 55% neutral vs 5% very hawkish)
- Text length bias per class (are short texts disproportionately labeled neutral?)
- Unexpected label co-occurrence patterns

### NLI Scoring (`nli`)

Uses a Natural Language Inference cross-encoder to test whether each text semantically entails its assigned label.

**What it finds:**
- Label-text mismatches: a text that says "rates should remain low" assigned label `2` (very hawkish).
- Systematically weak hypotheses that need rewording.

**Requirements:**
- A `hypotheses.yaml` file mapping each label to a natural-language hypothesis. Located at `prompts/{dataset}/hypotheses.yaml`:

```yaml
labels:
  "-2": "This statement is very dovish and advocates for monetary easing."
  "-1": "This statement is somewhat dovish."
  "0": "This statement is neutral about monetary policy."
  "1": "This statement is somewhat hawkish."
  "2": "This statement is very hawkish and advocates for monetary tightening."
```

**Key config options:**
```yaml
diagnostics:
  nli_model: "cross-encoder/nli-deberta-v3-large"
  nli_batch_size: 256        # lower to 64-128 on CPU
  nli_max_length: 45         # truncation; headlines are ~40 tokens
  nli_entailment_threshold: 0.5
```

### Batch Diagnostics (`batch`)

Detects temporal drift and cascade-tier bias across processing batches.

**What it finds:**
- Label distribution shifts between early and late batches (KL divergence).
- Systematic differences in how cascade tiers assign labels.

**Requirements:** Pipeline output with `jury_labels` column. Skipped automatically for human-labeled data.

### Rationale Analysis (`rationale`)

Checks consistency between model reasoning and assigned labels.

**What it finds:**
- Cases where the reasoning describes one class but the label assigned is different.
- Models that contradict each other's reasoning on the same sample.

**Requirements:** Pipeline output with `reasoning` column. Skipped automatically for human-labeled data.

### Quality Report (`report`)

Aggregates findings from all other modules into a structured report.

**Output files:**
- `quality_report.md` — human-readable Markdown with per-class scorecards and prioritised recommendations.
- `quality_report.json` — machine-readable version for downstream tooling.

## Suspicion Scoring

All diagnostic signals are combined into a single per-sample suspicion score (0-1) using configurable weights:

```yaml
diagnostics:
  suspicion_weights:
    embedding_outlier: 0.25
    nli_mismatch: 0.25
    low_confidence: 0.20
    jury_disagreement: 0.15
    rationale_inconsistency: 0.15
```

The top-K suspects are saved to `top_suspects.csv`. These are the samples most worth reviewing manually.

## UMAP Visualisation

When the embedding module runs, an interactive HTML scatter plot is generated showing all samples in 2D UMAP space, coloured by label. Hover over points to see the text and suspicion score. Outliers and violations are visually apparent as points sitting in the wrong colour region.

## Programmatic Usage

```python
from sibyls.core.diagnostics import run_diagnostics
from sibyls.core.dataset_config import DatasetConfig
import pandas as pd
from pathlib import Path

config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
df = pd.read_csv("outputs/fed_headlines/labeled.csv")

results = run_diagnostics(
    labeled_df=df,
    config=config,
    output_dir=Path("outputs/fed_headlines/diagnostics"),
    enabled_modules=["embedding", "nli", "report"],
)

# Access results
scored_df = results["scored_df"]
top_suspects = scored_df.nlargest(50, "suspicion_score")

report = results["report"]
for rec in report.get("recommendations", []):
    print(f"[{rec['priority']}] {rec['finding']}")
```

## CLI Reference

```
python scripts/run_diagnostics.py --help

Required:
  --dataset          Dataset name (matches configs/{dataset}.yaml)
  --labeled-data     Path to labeled CSV
  --output           Output directory

Optional:
  --enable           Comma-separated modules (default: embedding,distribution,nli,batch,rationale,report)
  --text-column      Text column name (default: from config)
  --label-column     Label column name (default: "label")
  --top-k-suspects   Override number of top suspects (default: from config)
  --hypotheses       Path to hypotheses.yaml (overrides config)
  --limit            Limit to first N rows
  --force-gap-analysis  Force-enable gap analysis even if config says disabled
  --verbose / -v     DEBUG logging
```

## Tips

- **Start with embedding + report** for the fastest feedback. NLI requires downloading a ~1.3 GB model on first run.
- **Use `--limit 500`** during development to iterate quickly on config tuning.
- **Compare reports across runs**: if you fix data and re-run diagnostics, the suspicion score distribution should improve.
- **The UMAP HTML** is the fastest way to visually spot class overlap and outliers.
- **For human-labeled data**, always pass `--text-column` and `--label-column` explicitly. The diagnostics pipeline will skip modules that require pipeline-specific columns (jury_labels, reasoning).
