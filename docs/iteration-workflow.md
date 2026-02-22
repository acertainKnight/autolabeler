# Iteration Workflow

This guide walks through the full data-improvement loop: diagnose problems, fix the training data, measure the impact, and repeat until metrics stabilise.

## The Core Loop

```
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
 1. Run diagnostics          ─── find problems         │
    │                                                  │
    ▼                                                  │
 2. Run gap analysis         ─── understand patterns   │
    │                                                  │
    ▼                                                  │
 3. Fix training data        ─── relabel, add, remove  │
    │                                                  │
    ▼                                                  │
 4. Re-export for distillation                         │
    │                                                  │
    ▼                                                  │
 5. Train probe model        ─── measure improvement   │
    │                                                  │
    ▼                                                  │
 6. Compare metrics          ─── better? ──── No ──────┘
    │
    │ Yes
    ▼
 7. Full cloud training
```

Each pass through this loop takes 10-15 minutes on a laptop. You might go around 3-5 times before the data is ready for production training.

## Step-by-Step

### Step 1: Run Diagnostics

Start with the standard modules to get a quality report and suspicion scores:

```bash
python scripts/run_diagnostics.py \
    --dataset fed_headlines \
    --labeled-data outputs/fed_headlines/labeled.csv \
    --output outputs/fed_headlines/diagnostics/ \
    --enable embedding,distribution,nli,report \
    --top-k-suspects 500
```

Review `quality_report.md` for the big picture. Check `top_suspects.csv` for the most suspicious individual samples. Look at the UMAP HTML to visually spot class overlap.

### Step 2: Run Gap Analysis

Once you have a sense of the overall quality, run gap analysis to understand *patterns* of errors:

```bash
python scripts/run_diagnostics.py \
    --dataset fed_headlines \
    --labeled-data outputs/fed_headlines/labeled.csv \
    --output outputs/fed_headlines/diagnostics/ \
    --enable embedding,report,gap_analysis \
    --force-gap-analysis
```

Open `gap_report.md`. Each cluster tells you:
- What topic the errors share
- Why the classifier struggles with it
- What the correct labels probably are
- What new examples would help

### Step 3: Fix the Training Data

Based on the diagnostics and gap report, make targeted fixes:

**Relabel suspicious samples:**
Open `top_suspects.csv`, review the high-suspicion rows, and correct labels in your source data.

**Merge synthetic examples:**
If gap analysis generated `synthetic_examples.csv`, review and edit the examples, then add the good ones to your training data.

**Remove or quarantine bad data:**
Some samples are genuinely ambiguous or mislabeled. Remove them or mark them as quarantine so they get zero training weight.

**Add real examples for weak classes:**
If the gap report identified underrepresented topics (e.g. "conditional rate guidance"), find or annotate real examples for those topics.

### Step 4: Re-Export

After fixing the source data, re-run the distillation export:

```bash
python scripts/export_for_distillation.py \
    --llm-labels outputs/fed_headlines/labeled_v2.csv \
    --human-labels datasets/fedspeak/human_labeled.csv \
    --human-text-column headline \
    --human-label-column hawk_dove \
    --output outputs/fed_headlines/distillation_v2.jsonl
```

### Step 5: Train the Probe

```bash
python scripts/train_probe.py \
    --dataset fed_headlines \
    --training-data outputs/fed_headlines/distillation_v2.jsonl \
    --output outputs/fed_headlines/probe_v2/
```

### Step 6: Compare Metrics

Compare the current run against the previous one:

```bash
# Quick comparison via probe_summary.json
cat outputs/fed_headlines/probe/probe_summary.json
cat outputs/fed_headlines/probe_v2/probe_summary.json
```

Key metrics to track across iterations:

| Metric | What to look for |
|--------|-----------------|
| **Macro F1** | Most sensitive to rare-class improvements |
| **Per-class F1** | Did the specific classes flagged by gap analysis improve? |
| **Confusion matrix** | Are the off-diagonal errors shrinking? |
| **Cohen's Kappa** | Overall agreement quality, adjusted for chance |

If metrics improved, great. If not, look at where the confusion matrix shifted — the fix may have helped one class while hurting another. Re-run diagnostics on the new data to find the new bottleneck.

### Step 7: Full Cloud Training

Once the probe's macro F1 and per-class metrics have stabilised across 2-3 iterations, you have confidence the training data is solid. Run the full cloud training pipeline.

## Example: Fed Headlines Iteration

Here's a concrete example of what one iteration might look like.

**Diagnostics report says:**
- 23% of samples in class `1` (somewhat hawkish) are embedding outliers
- NLI flags 85 samples where the text sounds dovish but is labeled hawkish

**Gap analysis finds:**
- Cluster 3 (45 samples): "Conditional rate hike projections" — mixed between `0` and `1`
- Cluster 7 (28 samples): "Historical dovish references in hawkish context" — labeled `1` but contain dovish language

**Actions taken:**
1. Relabeled 30 of the 45 "conditional" samples from `1` to `0` (they're genuinely neutral).
2. Kept 15 as `1` (they really are hawkish despite conditional language).
3. Added 10 synthetic examples from the gap report for "conditional" language, correctly labeled.
4. Quarantined 8 of the "historical reference" samples that were ambiguous.

**Probe results:**

| Metric | Before | After |
|--------|--------|-------|
| Macro F1 | 0.58 | 0.63 |
| Class `1` F1 | 0.42 | 0.55 |
| Class `0` F1 | 0.71 | 0.73 |

Improvement confirmed. Run diagnostics again on the new data to find the next bottleneck.

## Tracking Progress

Keep a simple log of each iteration:

```
# iteration_log.md

## v1 (baseline)
- Probe: macro_f1=0.58, accuracy=0.74
- Top issues: conditional hawkish language, historical references

## v2
- Fixed 30 conditional samples, added 10 synthetic
- Probe: macro_f1=0.63, accuracy=0.78
- Remaining: class -2 still weak (F1=0.35), need more very-dovish examples

## v3
- Added 20 very-dovish examples from Bloomberg archive
- Probe: macro_f1=0.67, accuracy=0.80
- Stable across 2 runs. Ready for cloud training.
```

## Tips

- **Don't over-iterate.** 3-5 passes is typical. If macro F1 stops improving, the remaining errors are probably genuinely ambiguous.
- **Fix the biggest gap first.** The gap report is sorted by severity — start with cluster 1.
- **Use `distilroberta-base` for intermediate checks** and `roberta-base` for the "real" measurement. Saves time without losing signal.
- **Keep versioned exports.** Name them `distillation_v1.jsonl`, `distillation_v2.jsonl`, etc. so you can always go back.
- **Re-run diagnostics after fixes.** Your fixes might shift error patterns. The second gap report will look different from the first.
