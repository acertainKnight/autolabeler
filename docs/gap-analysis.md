# Gap Analysis Guide

Gap analysis goes beyond individual error detection. Instead of flagging isolated suspicious samples, it groups errors by topic and asks an LLM to diagnose what the classifier is systematically getting wrong — and what to do about it.

## When to Use

- After running standard diagnostics and reviewing the quality report.
- When you see recurring error patterns (e.g. "conditional rate guidance" headlines keep getting misclassified) and want to understand them at scale.
- When you want synthetic training examples to fill specific data gaps.

## How It Works

The gap analyzer runs as step 10 of the diagnostics pipeline, after suspicion scoring:

1. **Error pool construction** — collects the top-N highest-suspicion samples plus any centroid violations flagged by the embedding analyzer.
2. **Semantic clustering** — UMAP-reduces the cached embeddings for the error pool, then runs HDBSCAN to group errors by topic. Noise samples (label `-1`) are excluded.
3. **Cluster summarisation** — computes TF-IDF topic terms, label distribution, mean suspicion score, and representative texts for each cluster. Clusters are ranked by severity (size x mean suspicion).
4. **LLM diagnosis** — sends each cluster's representative texts and metadata to an LLM, which returns:
   - A concise topic label (e.g. "Conditional rate hike projections")
   - What makes these texts hard to classify
   - What the correct labels likely are
   - What augmentation strategy would help
   - Optionally, synthetic training examples

## Quick Start

### Option A: Enable in config

```yaml
# configs/fed_headlines.yaml
diagnostics:
  gap_analysis:
    enabled: true
    analysis_provider: "google"
    analysis_model: "gemini-2.5-flash"
```

Then run diagnostics as usual — gap analysis runs automatically:

```bash
python scripts/run_diagnostics.py \
    --dataset fed_headlines \
    --labeled-data outputs/fed_headlines/labeled.csv \
    --output outputs/fed_headlines/diagnostics/ \
    --enable embedding,report,gap_analysis
```

### Option B: Force a one-off run

Leave the config unchanged and use the CLI flag:

```bash
python scripts/run_diagnostics.py \
    --dataset fed_headlines \
    --labeled-data outputs/fed_headlines/labeled.csv \
    --output outputs/fed_headlines/diagnostics/ \
    --enable embedding,report,gap_analysis \
    --force-gap-analysis
```

## Configuration

```yaml
diagnostics:
  gap_analysis:
    enabled: false              # master switch
    analysis_provider: "google" # LLM provider: google, openai, anthropic, openrouter
    analysis_model: "gemini-2.5-flash"  # model for cluster diagnosis
    min_cluster_size: 5         # HDBSCAN minimum cluster size
    max_clusters: 20            # prune smallest clusters beyond this cap
    representative_samples: 10  # texts sent to the LLM per cluster
    generate_synthetic: true    # generate training examples per gap
    synthetic_per_cluster: 5    # how many synthetic examples per cluster
    top_n_suspicious: 500       # error pool size (top suspects by suspicion score)
```

**Tuning tips:**
- `top_n_suspicious`: Start with 500. If your dataset is small (<2000 samples), lower to 200.
- `min_cluster_size`: Lower (3) for more granular clusters, higher (10) for broader themes.
- `max_clusters`: 20 is a good default. Review the first report and adjust.
- `representative_samples`: 10 gives good LLM context without excessive token cost. Lower to 5 if you're analyzing many clusters.
- `analysis_model`: Flash-tier models (gemini-2.5-flash, gpt-4o-mini) are cost-effective. Use a stronger model (gemini-2.5-pro, gpt-4o) for deeper analysis if cost isn't a concern.

## Output Files

All outputs are written to the diagnostics output directory.

### `gap_report.md`

Human-readable Markdown. Each cluster gets a section with:
- Severity score, sample count, centroid violations
- Assigned label distribution
- Why it's hard to classify (from LLM)
- Likely correct labels
- Augmentation strategy
- Top keywords and representative texts
- Synthetic examples (if enabled)

### `gap_report.json`

Machine-readable version with the same data. Structure:

```json
{
  "dataset_name": "fed_headlines",
  "n_error_pool": 450,
  "n_clusters": 12,
  "n_synthetic_examples": 60,
  "clusters": [
    {
      "cluster_id": 3,
      "n_samples": 45,
      "n_violations": 8,
      "mean_suspicion": 0.62,
      "severity": 27.9,
      "label_distribution": {"0": 0.45, "1": 0.35, "-1": 0.20},
      "topic_terms": ["rate", "hike", "conditional", "projection", "may"],
      "representative_texts": ["...", "..."],
      "llm_diagnosis": {
        "topic": "Conditional rate hike projections",
        "ambiguity": "These headlines describe potential rate hikes contingent on data. The conditional framing makes them feel neutral, but the rate hike mention pulls toward hawkish.",
        "likely_correct_labels": {"0": 0.4, "1": 0.5, "-1": 0.1},
        "augmentation_strategy": "Add examples of conditional hawkish language that should be labeled 1, and clearly neutral forecasting language that should be labeled 0."
      },
      "synthetic_examples": [
        {"text": "Fed may raise rates if inflation persists above target", "label": "1"},
        {"text": "Economists forecast unchanged rates pending employment data", "label": "0"}
      ]
    }
  ]
}
```

### `synthetic_examples.csv`

Flat CSV with all generated synthetic examples from all clusters, ready for import into your training data:

```csv
text,label
"Fed may raise rates if inflation persists above target",1
"Economists forecast unchanged rates pending employment data",0
```

## Using Synthetic Examples

The generated synthetic examples are a starting point, not production-ready labels. The recommended workflow:

1. Review `gap_report.md` to understand each cluster.
2. Open `synthetic_examples.csv` and validate/edit the generated examples.
3. Merge validated examples into your training data (or distillation JSONL).
4. Re-export for distillation and retrain the probe model.
5. Check if the probe's per-class metrics improve for the affected classes.

## Programmatic Usage

```python
from sibyls.core.diagnostics.gap_analyzer import GapAnalyzer
from sibyls.core.diagnostics.config import DiagnosticsConfig
from pathlib import Path

config = DiagnosticsConfig.from_dict({
    "gap_analysis": {
        "enabled": True,
        "generate_synthetic": True,
        "analysis_provider": "google",
        "analysis_model": "gemini-2.5-flash",
    },
})

analyzer = GapAnalyzer(config)

# diagnostic_results comes from run_diagnostics()
results = analyzer.run_all(
    labeled_df=df,
    diagnostic_results=diagnostic_results,
    output_dir=Path("outputs/gaps"),
    label_definitions={
        "2": "Very hawkish",
        "1": "Somewhat hawkish",
        "0": "Neutral",
        "-1": "Somewhat dovish",
        "-2": "Very dovish",
    },
    dataset_name="fed_headlines",
)

# Access results
print(f"Found {results['gap_report']['n_clusters']} error clusters")
for cluster in results["summaries"]:
    topic = cluster["llm_diagnosis"]["topic"]
    print(f"  {topic}: {cluster['n_samples']} samples, severity {cluster['severity']:.1f}")
```

## Dependencies

Gap analysis uses:
- **UMAP + HDBSCAN** (already in core dependencies) for clustering
- **scikit-learn** (already in core dependencies) for TF-IDF
- **LLM provider** (configured in `analysis_provider` / `analysis_model`) for diagnosis

No additional installation is required beyond the base `pip install -e .`
