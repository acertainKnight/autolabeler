#!/usr/bin/env python3
"""Train a lightweight local RoBERTa probe model for fast training-data iteration.

Fine-tunes a small HuggingFace classifier on the distillation export to get
quick performance feedback before running the full cloud training pipeline.

The typical iteration loop:
    1. Run diagnostics + gap analysis to identify data problems.
    2. Fix training data (relabel, add examples, merge synthetic).
    3. Re-export:  python scripts/export_for_distillation.py ...
    4. Retrain probe:  python scripts/train_probe.py ...
    5. Compare metrics against previous run.
    6. Repeat until satisfied, then run full cloud training.

Requires the optional probe dependency group:
    pip install 'sibyls[probe]'

Usage:
    # Basic: train on distillation JSONL (uses probe: block in config YAML)
    python scripts/train_probe.py \\
        --dataset fed_headlines \\
        --training-data outputs/fed_headlines/distillation.jsonl

    # Override hyperparams via CLI (takes precedence over YAML)
    python scripts/train_probe.py \\
        --dataset fed_headlines \\
        --training-data outputs/fed_headlines/distillation.jsonl \\
        --model distilroberta-base \\
        --epochs 3 \\
        --batch-size 64 \\
        --output outputs/fed_headlines/probe_v2/

    # Faster, smaller model for quick smoke tests
    python scripts/train_probe.py \\
        --dataset fed_headlines \\
        --training-data outputs/fed_headlines/distillation.jsonl \\
        --model distilroberta-base \\
        --epochs 2 --max-length 64

    # Disable per-sample training weights (all samples weighted equally)
    python scripts/train_probe.py \\
        --dataset fed_headlines \\
        --training-data outputs/fed_headlines/distillation.jsonl \\
        --no-weights

    # GPU training with mixed precision
    python scripts/train_probe.py \\
        --dataset fed_headlines \\
        --training-data outputs/fed_headlines/distillation.jsonl \\
        --fp16

    # Evaluate a previously saved model on new/updated data
    python scripts/train_probe.py \\
        --dataset fed_headlines \\
        --eval-only \\
        --model-dir outputs/fed_headlines/probe \\
        --training-data outputs/fed_headlines/distillation_v2.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sibyls.core.dataset_config import DatasetConfig
from src.sibyls.core.probe.config import ProbeConfig
from src.sibyls.core.probe.trainer import ProbeTrainer


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging.

    Args:
        verbose: Enable DEBUG level logging.
    """
    logger.remove()
    log_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | '
        '<level>{message}</level>'
    )
    level = 'DEBUG' if verbose else 'INFO'
    logger.add(sys.stderr, format=log_format, level=level)


def print_metrics_table(metrics: dict, dataset_name: str) -> None:
    """Print a human-readable metrics summary to stdout.

    Args:
        metrics: Metrics dict from ProbeTrainer.train().
        dataset_name: Dataset name for the header.
    """
    print()
    print('=' * 60)
    print(f'PROBE MODEL EVALUATION -- {dataset_name.upper()}')
    print('=' * 60)

    core_keys = [
        ('accuracy', 'Accuracy'),
        ('f1_macro', 'Macro F1'),
        ('f1_weighted', 'Weighted F1'),
        ('macro_f1', 'Macro F1'),
        ('weighted_f1', 'Weighted F1'),
        ('cohen_kappa', "Cohen's Kappa"),
        ('mae', 'Mean Abs Error (ordinal)'),
        ('spearman_rho', 'Spearman ρ (ordinal)'),
    ]

    printed = set()
    for key, label in core_keys:
        if key in metrics and key not in printed:
            val = metrics[key]
            if isinstance(val, float):
                print(f'  {label:<30} {val:.4f}')
            else:
                print(f'  {label:<30} {val}')
            printed.add(key)

    # 3-class metrics if present (fed_headlines)
    three_class_keys = [
        ('3class_accuracy', '3-class Accuracy'),
        ('3class_f1_macro', '3-class Macro F1'),
    ]
    three_class_printed = False
    for key, label in three_class_keys:
        if key in metrics:
            if not three_class_printed:
                print()
                print('  --- 3-class (dove/neutral/hawk) ---')
                three_class_printed = True
            print(f'  {label:<30} {metrics[key]:.4f}')

    print('=' * 60)
    print()


def main() -> int:
    """Parse CLI arguments and run probe training or evaluation.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(
        description='Train a lightweight probe model for fast training-data iteration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (must match configs/{dataset}.yaml)',
    )
    parser.add_argument(
        '--training-data',
        required=True,
        help='Path to distillation JSONL (from export_for_distillation.py)',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory for model + results (default: outputs/{dataset}/probe/)',
    )
    parser.add_argument(
        '--model',
        default=None,
        help='HuggingFace model ID (overrides config; default: roberta-base)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Training epochs (overrides config)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Per-device batch size (overrides config)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=None,
        help='Tokenizer max sequence length (overrides config)',
    )
    parser.add_argument(
        '--no-weights',
        action='store_true',
        help='Disable per-sample training weights (equal weighting)',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Enable mixed-precision training (requires CUDA)',
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Skip training; evaluate a previously saved model',
    )
    parser.add_argument(
        '--model-dir',
        default=None,
        help='Directory of saved probe model for --eval-only mode',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable DEBUG logging',
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load dataset config
    config_path = Path(f'configs/{args.dataset}.yaml')
    if not config_path.exists():
        logger.error(f'Config not found: {config_path}')
        available = [f.stem for f in Path('configs').glob('*.yaml')]
        logger.error(f'Available configs: {available}')
        return 1

    dataset_config = DatasetConfig.from_yaml(config_path)
    logger.info(f'Loaded config for {dataset_config.name}')

    # Build ProbeConfig: start from YAML probe block, apply CLI overrides
    probe_config_dict = getattr(dataset_config, 'probe_config', None) or {}
    # Try reading from the raw YAML if probe_config is not on the dataclass
    if not probe_config_dict:
        import yaml
        with open(config_path, encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        probe_config_dict = raw.get('probe', {}) or {}

    probe_config = ProbeConfig.from_dict(probe_config_dict)

    # CLI overrides
    if args.model:
        probe_config.model_name = args.model
    if args.epochs is not None:
        probe_config.epochs = args.epochs
    if args.batch_size is not None:
        probe_config.batch_size = args.batch_size
    if args.lr is not None:
        probe_config.learning_rate = args.lr
    if args.max_length is not None:
        probe_config.max_length = args.max_length
    if args.no_weights:
        probe_config.use_training_weights = False
    if args.fp16:
        probe_config.fp16 = True
    if args.output:
        probe_config.output_dir = args.output

    trainer = ProbeTrainer(probe_config, dataset_config)

    # ------------------------------------------------------------------
    # Eval-only mode
    # ------------------------------------------------------------------
    if args.eval_only:
        model_dir = args.model_dir or probe_config.resolve_output_dir(dataset_config.name)
        if not Path(model_dir).exists():
            logger.error(f'Model directory not found: {model_dir}')
            return 1

        logger.info(f'Evaluating saved model at {model_dir}...')
        try:
            metrics = trainer.evaluate_saved(
                model_dir=model_dir,
                eval_data=args.training_data,
            )
        except Exception as exc:
            logger.error(f'Evaluation failed: {exc}')
            return 1

        print_metrics_table(metrics, dataset_config.name)
        return 0

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------
    logger.info(
        f'Training probe: {probe_config.model_name} | '
        f'epochs={probe_config.epochs} | '
        f'batch={probe_config.batch_size} | '
        f'lr={probe_config.learning_rate}'
    )

    try:
        results = trainer.train(
            training_data=args.training_data,
            output_dir=args.output,
        )
    except ImportError as exc:
        logger.error(str(exc))
        logger.error("Install probe dependencies: pip install 'sibyls[probe]'")
        return 1
    except Exception as exc:
        logger.error(f'Training failed: {exc}')
        return 1

    print_metrics_table(results['metrics'], dataset_config.name)

    logger.info(f"Model saved to: {results['model_path']}")
    logger.info(f"Train size: {results['train_size']:,} | Val size: {results['val_size']:,}")

    # Write a compact JSON summary alongside the model
    summary_path = Path(results['model_path']) / 'probe_summary.json'
    summary = {
        'dataset': dataset_config.name,
        'model': probe_config.model_name,
        'epochs': probe_config.epochs,
        'train_size': results['train_size'],
        'val_size': results['val_size'],
        'metrics': {
            k: v for k, v in results['metrics'].items()
            if isinstance(v, (int, float))
        },
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'Summary saved to {summary_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
