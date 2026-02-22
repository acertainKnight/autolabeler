"""Configuration for the probe model fine-tuning.

Maps directly to the ``probe:`` block in dataset YAML configs.
All fields have sensible defaults suitable for roberta-base on short texts
(~40 tokens for Fed headlines). Larger models or longer texts may need
adjusted batch_size and max_length values.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProbeConfig:
    """Hyperparameter and runtime configuration for the probe model.

    Args:
        model_name: HuggingFace model identifier to fine-tune. Options:
            - "roberta-base" (~125M params, ~500 MB): recommended default.
              Trains in ~5 min on 20k headlines (CPU) or ~1 min (GPU).
            - "distilroberta-base" (~82M, ~330 MB): faster, slightly less
              accurate. Good for rapid smoke tests.
            - "roberta-large" (~355M, ~1.3 GB): higher ceiling but slow
              on CPU and needs more VRAM.
        epochs: Number of training epochs. 3-5 is typical for short text;
            more epochs risk overfitting on small datasets.
        batch_size: Per-device train and eval batch size. 32 is safe for
            most hardware; lower to 16 if you hit OOM on GPU.
        learning_rate: Peak learning rate for AdamW. 2e-5 is the standard
            starting point for RoBERTa fine-tuning.
        val_split: Fraction of data held out for validation (stratified by
            label to preserve class proportions).
        max_length: Tokenizer truncation length. 128 is sufficient for
            most short-text tasks (Fed headlines are ~40 tokens). Set
            higher (256-512) for paragraph-length inputs.
        warmup_ratio: Fraction of total steps used for linear warmup.
        weight_decay: L2 regularisation coefficient for AdamW.
        use_training_weights: If True, use the training_weight column from
            the distillation JSONL as per-sample loss weights. This gives
            ACCEPT-tier labels more influence than QUARANTINE samples.
        fp16: Enable mixed-precision training on CUDA (ignored on CPU).
            Roughly halves VRAM usage and speeds up training by ~30%.
        seed: Random seed for reproducibility across runs.
        output_dir: Where to save the best checkpoint and tokenizer.
            If None, defaults to ``outputs/{dataset_name}/probe/``.
        logging_steps: Log training loss every N optimizer steps.
        save_total_limit: Keep only the N best checkpoints on disk.
        load_best_model_at_end: Load the checkpoint with best val loss
            after training completes.
        metric_for_best_model: Which eval metric to use for best-model
            selection. "eval_loss" is stable; "eval_accuracy" is intuitive.

    Example:
        >>> cfg = ProbeConfig(model_name="roberta-base", epochs=3)
        >>> cfg.learning_rate
        2e-05

        >>> # Smaller, faster model for quick iteration
        >>> cfg = ProbeConfig(model_name="distilroberta-base", epochs=2, max_length=64)
    """

    model_name: str = 'roberta-base'
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    val_split: float = 0.2
    max_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    use_training_weights: bool = True
    fp16: bool = False
    seed: int = 42
    output_dir: str | None = None
    logging_steps: int = 50
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'eval_loss'

    @classmethod
    def from_dict(cls, data: dict) -> 'ProbeConfig':
        """Build a ProbeConfig from a plain dictionary (e.g. parsed YAML).

        Accepts both the YAML key name ``model`` and the dataclass field name
        ``model_name`` for convenience.

        Args:
            data: Dictionary with probe configuration keys.

        Returns:
            ProbeConfig instance with defaults for any missing keys.

        Example:
            >>> cfg = ProbeConfig.from_dict({"model": "distilroberta-base", "epochs": 3})
            >>> cfg.model_name
            'distilroberta-base'
        """
        return cls(
            model_name=data.get('model', data.get('model_name', 'roberta-base')),
            epochs=data.get('epochs', 5),
            batch_size=data.get('batch_size', 32),
            learning_rate=data.get('learning_rate', 2e-5),
            val_split=data.get('val_split', 0.2),
            max_length=data.get('max_length', 128),
            warmup_ratio=data.get('warmup_ratio', 0.1),
            weight_decay=data.get('weight_decay', 0.01),
            use_training_weights=data.get('use_training_weights', True),
            fp16=data.get('fp16', False),
            seed=data.get('seed', 42),
            output_dir=data.get('output_dir'),
            logging_steps=data.get('logging_steps', 50),
            save_total_limit=data.get('save_total_limit', 2),
            load_best_model_at_end=data.get('load_best_model_at_end', True),
            metric_for_best_model=data.get('metric_for_best_model', 'eval_loss'),
        )

    def resolve_output_dir(self, dataset_name: str) -> Path:
        """Resolve output directory, defaulting to outputs/{dataset_name}/probe/.

        Args:
            dataset_name: Dataset name used to construct default path.

        Returns:
            Resolved Path for model output.

        Example:
            >>> cfg = ProbeConfig()
            >>> str(cfg.resolve_output_dir("fed_headlines"))
            'outputs/fed_headlines/probe'
        """
        if self.output_dir:
            return Path(self.output_dir)
        return Path('outputs') / dataset_name / 'probe'
