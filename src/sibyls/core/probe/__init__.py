"""Lightweight local probe model for fast training-data iteration.

Fine-tunes a small HuggingFace classifier (default: roberta-base) on the
distillation export to get quick performance feedback before running the
full cloud training pipeline.

The intended workflow:
    1. Run the labeling pipeline and export distillation JSONL.
    2. Train a probe model in minutes on your local machine.
    3. Review accuracy / F1 to gauge training-data quality.
    4. Use diagnostics + gap analysis to fix data issues, re-export, retrain.
    5. Once probe metrics stabilise, run the full cloud training pipeline.

Quick start (programmatic):
    >>> from sibyls.core.probe import ProbeTrainer, ProbeConfig
    >>> from sibyls.core.dataset_config import DatasetConfig
    >>> ds_cfg = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
    >>> probe_cfg = ProbeConfig(model_name="roberta-base", epochs=3)
    >>> trainer = ProbeTrainer(probe_cfg, ds_cfg)
    >>> results = trainer.train("outputs/fed_headlines/distillation.jsonl")
    >>> print(results["metrics"]["accuracy"])

Quick start (CLI):
    python scripts/train_probe.py \\
        --dataset fed_headlines \\
        --training-data outputs/fed_headlines/distillation.jsonl
"""

from .config import ProbeConfig
from .trainer import ProbeTrainer

__all__ = ['ProbeConfig', 'ProbeTrainer']
