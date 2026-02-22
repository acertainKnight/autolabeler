# Configuration Reference

Every dataset is configured through a single YAML file at `configs/{dataset}.yaml`. This document covers every configuration block and field.

## Top-Level Fields

```yaml
name: fed_headlines             # dataset identifier (used in paths and logging)
labels: ["-99", "-2", "-1", "0", "1", "2"]  # valid label vocabulary
text_column: text               # column name in input CSV containing text
input_format: csv               # input file format
```

## Pipeline Stages

Toggle individual stages of the labeling pipeline:

```yaml
use_relevancy_gate: true        # Stage 1: cheap pre-filter for irrelevant items
use_candidate_annotation: true  # Stage 4: soft labels for jury disagreements
use_cross_verification: false   # Stage 5: independent review of uncertain labels
use_structured_output: true     # JSON schema constraints for LLM responses
use_cascade: false              # cost-saving cascade mode (see below)
```

## Jury Models

Define 2-6 models for the heterogeneous jury. Diversity across model families (OpenAI, Anthropic, Google) improves error detection.

```yaml
jury_models:
  - provider: google
    model: gemini-2.5-flash
    name: Gemini-Flash           # display name in logs and reports
    has_logprobs: false          # only OpenAI models support this
    self_consistency_samples: 3  # number of temperature-varied samples
    cost_tier: 1                 # cascade ordering (1=cheapest, called first)

  - provider: openai
    model: gpt-4o
    name: GPT-4o
    has_logprobs: true           # enables logprob-based confidence
    self_consistency_samples: 0  # not needed when logprobs are available
    cost_tier: 2

  - provider: anthropic
    model: claude-sonnet-4-5-20250929
    name: Claude
    has_logprobs: false
    self_consistency_samples: 3
    cost_tier: 2
```

**Supported providers:**

| Provider | Config value | API key env var | Logprobs | Notes |
|----------|-------------|-----------------|----------|-------|
| OpenAI | `openai` | `OPENAI_API_KEY` | Yes | Best for confidence scoring |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | No | Use self-consistency |
| Google | `google` | `GOOGLE_API_KEY` | No | Use self-consistency |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | No | 200+ models, one key |

You can mix providers freely. A common pattern is OpenAI direct (for logprobs) plus OpenRouter for everything else.

## Gate, Candidate, and Verification Models

```yaml
gate_model:
  provider: openai
  model: gpt-4o-mini
  name: Gate

candidate_model:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  name: Claude-Candidate

verification_model:
  provider: google
  model: gemini-2.5-pro
  name: Verifier
verification_threshold: 0.6   # trigger verification below this confidence
```

## Temperature Settings

```yaml
jury_temperature: 0.1          # low for deterministic labels
sc_temperature: 0.3            # higher for self-consistency diversity
candidate_temperature: 0.2     # moderate for soft label generation
```

## Cascade Mode

Call the cheapest model first and only escalate when uncertain. Saves 40-60% API cost on easy items.

```yaml
use_cascade: false
cascade_confidence_threshold: 0.85  # single-model confidence to accept early
cascade_agreement_threshold: 0.80   # two-model agreement to accept
```

`cost_tier` on each jury model controls the order: tier 1 is called first, tier 2 only if tier 1 is uncertain, etc.

## Budget and Batching

```yaml
budget_per_model: 10.0         # max USD per model per run
batch_size: 10                 # texts per batch
```

## LIT Integration

Defaults for the Language Interpretability Tool browser UI (`scripts/run_lit.py`):

```yaml
lit:
  embedding_layer: null         # e.g. "encoder.pooler" for token embeddings
  enable_gradients: false       # token saliency maps
  enable_attention: false       # attention head visualisation
  port: 4321
```

## Diagnostics

Post-labeling error detection. See [diagnostics.md](diagnostics.md) for the full guide.

```yaml
diagnostics:
  enabled: true
  run_post_labeling: false      # auto-run after label_dataframe()

  # Embedding provider
  embedding_provider: "openrouter"  # "local", "openai", or "openrouter"
  embedding_model: "openai/text-embedding-3-small"
  embedding_batch_size: 2048
  # embedding_api_key: null     # falls back to OPENAI_API_KEY / OPENROUTER_API_KEY

  # NLI scoring
  nli_model: "cross-encoder/nli-deberta-v3-large"
  nli_batch_size: 256           # lower to 64-128 on CPU
  nli_max_length: 45            # truncation for short texts
  nli_entailment_threshold: 0.5

  # Embedding analysis thresholds
  outlier_z_threshold: 2.0
  lof_neighbors: 20
  fragmentation_min_cluster_size: 5
  fragmentation_umap_dim: 10    # UMAP dims before HDBSCAN

  # Duplicate detection
  duplicate_similarity_threshold: 0.95

  # Batch drift
  batch_drift_kl_threshold: 0.1

  # Suspicion scoring weights (should sum to ~1.0)
  suspicion_weights:
    embedding_outlier: 0.25
    nli_mismatch: 0.25
    low_confidence: 0.20
    jury_disagreement: 0.15
    rationale_inconsistency: 0.15

  # Audit output
  top_k_suspects: 100

  # Gap analysis (optional, see gap-analysis.md)
  gap_analysis:
    enabled: false
    analysis_provider: "google"
    analysis_model: "gemini-2.5-flash"
    min_cluster_size: 5
    max_clusters: 20
    representative_samples: 10
    generate_synthetic: true
    synthetic_per_cluster: 5
    top_n_suspicious: 500
```

### Embedding Provider Examples

**Local (free, on-device):**
```yaml
embedding_provider: "local"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_batch_size: 64
```

**OpenAI:**
```yaml
embedding_provider: "openai"
embedding_model: "text-embedding-3-small"  # or text-embedding-3-large
embedding_batch_size: 2048
```

**OpenRouter:**
```yaml
embedding_provider: "openrouter"
embedding_model: "openai/text-embedding-3-small"
embedding_batch_size: 2048
```

### NLI Batch Size Guidance

| Hardware | Recommended `nli_batch_size` | Notes |
|----------|------------------------------|-------|
| CPU only | 32-64 | Slow but works |
| GPU, 4 GB VRAM | 128 | Safe with deberta-v3-large |
| GPU, 8+ GB VRAM | 256-512 | Fast |

## Probe Model

Local fine-tuning for fast training-data evaluation. See [probe-model.md](probe-model.md) for the full guide.

```yaml
probe:
  model: "roberta-base"
  epochs: 5
  batch_size: 32
  learning_rate: 2.0e-5
  val_split: 0.2
  max_length: 128
  warmup_ratio: 0.1
  weight_decay: 0.01
  use_training_weights: true
```

All probe values can be overridden via CLI flags in `scripts/train_probe.py`.

## Prompts Directory

Prompts are not in the YAML config. They live in `prompts/{dataset_name}/`:

```
prompts/fed_headlines/
├── system.md          # role and domain expertise
├── rules.md           # classification decision framework
├── examples.md        # calibration examples with reasoning
├── mistakes.md        # common errors to avoid
├── candidate.md       # disagreement resolution prompt
├── verify.md          # cross-verification prompt
├── program_gen.md     # ALCHEmist program generation prompt
└── hypotheses.yaml    # NLI hypothesis definitions per label
```

## Environment Variables

Set in your shell or in a `.env` file:

```bash
# LLM API keys (set whichever providers you use)
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="..."
OPENROUTER_API_KEY="sk-or-..."

# Optional: logging level for loguru
LOG_LEVEL="INFO"
```

## Creating a New Dataset Config

1. Copy the template: `cp configs/example.yaml configs/my_dataset.yaml`
2. Update `name`, `labels`, and `text_column`.
3. Adjust jury models, temperatures, and pipeline toggles.
4. Copy and fill in prompts: `cp -r prompts/example prompts/my_dataset`
5. Replace all `[PLACEHOLDER]` fields in the prompt files.
6. Add `prompts/my_dataset/hypotheses.yaml` for NLI scoring (follow the format in `prompts/example/hypotheses.yaml`).
7. Run labeling: `python scripts/run_labeling.py --dataset my_dataset --input data.csv --output labeled.csv`
