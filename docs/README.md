# Documentation

## Guides

| Document | Description |
|----------|-------------|
| [Configuration Reference](configuration.md) | Complete reference for every YAML config block, environment variables, and provider setup |
| [Diagnostics Guide](diagnostics.md) | Post-labeling error detection: modules, CLI usage, output interpretation |
| [Gap Analysis Guide](gap-analysis.md) | LLM-powered error clustering, synthetic data generation, and output format |
| [Probe Model Guide](probe-model.md) | Local RoBERTa fine-tuning for fast training-data evaluation |
| [Iteration Workflow](iteration-workflow.md) | End-to-end data improvement loop: diagnose, fix, retrain, repeat |

## Getting Started

New to the project? Start with:

1. [Quick Start](../QUICKSTART.md) — get running in 5 minutes
2. [README](../README.md) — full system overview, pipeline architecture, evidence base
3. [Configuration Reference](configuration.md) — understand what every config option does

## Common Workflows

**First-time labeling:**
Quick Start -> README (pipeline section) -> Configuration Reference

**Improving existing labels:**
Diagnostics Guide -> Gap Analysis Guide -> Iteration Workflow

**Training a model:**
Probe Model Guide -> Iteration Workflow -> (then full cloud training)
