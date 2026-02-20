#!/usr/bin/env python3
"""Calibrate jury weights from human-labeled data.

This script learns per-model, per-class reliability weights from a calibration dataset
where both model predictions and human labels are available. The learned weights
significantly improve aggregation accuracy (LLM Jury-on-Demand, ICLR 2026).

Usage:
    # Basic usage with model columns from the dataframe
    python scripts/calibrate_jury.py \\
        --dataset fed_headlines \\
        --calibration-data datasets/human_labeled_LIVE_20251103.csv \\
        --model-columns gpt4_label claude_label gemini_label \\
        --true-label-column human_label \\
        --output outputs/fed_headlines/jury_weights.json

    # With custom model names for clarity
    python scripts/calibrate_jury.py \\
        --dataset fed_headlines \\
        --calibration-data datasets/human_labeled.csv \\
        --model-columns col1 col2 col3 \\
        --model-names "gpt-4o" "claude-sonnet-4" "gemini-2.5-pro" \\
        --true-label-column label \\
        --output outputs/jury_weights.json

Output:
    - JSON file with learned weights
    - Summary statistics printed to console
    - Per-class accuracy table for each model
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.autolabeler.core.quality.jury_weighting import JuryWeightLearner


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, level="DEBUG" if verbose else "INFO")


def main() -> int:
    """Parse arguments and run calibration."""
    parser = argparse.ArgumentParser(
        description="Learn per-model, per-class jury weights from human labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (for reference in output)",
    )

    parser.add_argument(
        "--calibration-data",
        required=True,
        help="Path to CSV with human labels and model predictions",
    )

    parser.add_argument(
        "--model-columns",
        nargs="+",
        required=True,
        help="Column names containing model predictions (space-separated)",
    )

    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Human-readable model names (defaults to column names)",
    )

    parser.add_argument(
        "--true-label-column",
        required=True,
        help="Column name containing ground truth labels",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for learned weights",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum calibration samples required (default: 50)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load calibration data
    logger.info(f"Loading calibration data from {args.calibration_data}")
    try:
        df = pd.read_csv(args.calibration_data)
    except Exception as e:
        logger.error(f"Failed to load calibration data: {e}")
        return 1

    logger.info(f"Loaded {len(df)} calibration samples")

    # Validate columns
    if args.true_label_column not in df.columns:
        logger.error(
            f"True label column '{args.true_label_column}' not found. "
            f"Available: {df.columns.tolist()}"
        )
        return 1

    for col in args.model_columns:
        if col not in df.columns:
            logger.error(
                f"Model column '{col}' not found. "
                f"Available: {df.columns.tolist()}"
            )
            return 1

    # Check minimum samples
    if len(df) < args.min_samples:
        logger.warning(
            f"Only {len(df)} calibration samples (minimum {args.min_samples} recommended). "
            "Weights may be unreliable."
        )

    # Learn weights
    logger.info("Learning jury weights...")
    learner = JuryWeightLearner()
    learner.fit(
        df=df,
        model_columns=args.model_columns,
        true_label_column=args.true_label_column,
        model_names=args.model_names,
    )

    # Print summary
    summary = learner.get_summary()
    logger.info("")
    logger.info("=" * 60)
    logger.info("JURY WEIGHT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Calibration samples: {summary['n_calibration_samples']}")
    logger.info(f"Models: {summary['n_models']}")
    logger.info(f"Labels: {summary['n_labels']}")
    logger.info(f"Weight pairs learned: {summary['n_weight_pairs']}")
    logger.info(f"Default weight: {summary['default_weight']:.3f}")
    logger.info(f"Mean weight: {summary['mean_weight']:.3f}")
    logger.info(f"Weight range: [{summary['min_weight']:.3f}, {summary['max_weight']:.3f}]")

    # Print per-class stats table
    if "per_class_stats" in summary and summary["per_class_stats"]:
        logger.info("")
        logger.info("Per-Class Accuracy Table:")
        logger.info("-" * 60)
        logger.info(f"{'Model':<20} {'Label':<10} {'N':<8} {'Correct':<8} {'Acc':<8}")
        logger.info("-" * 60)
        for stat in summary["per_class_stats"]:
            logger.info(
                f"{stat['model']:<20} {stat['label']:<10} "
                f"{stat['n_predicted']:<8} {stat['n_correct']:<8} "
                f"{stat['accuracy']:<8.3f}"
            )

    # Save weights
    output_path = Path(args.output)
    learner.save(output_path)

    logger.info("")
    logger.info(f"Jury weights saved to {output_path}")
    logger.info("")
    logger.info("To use these weights in labeling, add to your dataset config:")
    logger.info(f"  jury_weights_path: \"{output_path}\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())
