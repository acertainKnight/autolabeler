#!/usr/bin/env python3
"""Export labeled data for model distillation.

This script takes LLM-labeled data (from run_labeling.py) and exports it in a
format optimized for training distilled models, with confidence-based training
weights, soft label distributions, and optional human label mixing.

Usage:
    # Basic export
    python scripts/export_for_distillation.py \\
        --llm-labels outputs/fed_headlines/labeled.csv \\
        --output outputs/fed_headlines/distillation.jsonl

    # With human label mixing
    python scripts/export_for_distillation.py \\
        --llm-labels outputs/fed_headlines/labeled.csv \\
        --human-labels datasets/human_labeled_LIVE_20251103.csv \\
        --human-text-column headline \\
        --human-label-column label_hawk_dove \\
        --human-oversample 3.0 \\
        --output outputs/fed_headlines/distillation_with_human.jsonl

Output format (JSONL):
    {
      "text": "...",
      "hard_label": "0",
      "soft_label": {"0": 0.72, "1": 0.28},
      "training_weight": 0.85,
      "source": "llm",  # or "human"
      "tier": "ACCEPT",
      "verified": true,
      "confidence": 0.88,
      "agreement": "unanimous"
    }

Training weight formula:
    - ACCEPT tier, verified: 1.0
    - ACCEPT tier, unverified: 0.9
    - ACCEPT-M (candidate): 0.7
    - SOFT: 0.5
    - QUARANTINE: 0.0 (excluded by default)
    - Human label: 1.2 (oversampled)
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sibyls.core.export.distillation_export import DistillationExporter


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
    """Parse arguments and run export."""
    parser = argparse.ArgumentParser(
        description="Export labeled data for model distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--llm-labels",
        required=True,
        help="Path to LLM-labeled CSV (from run_labeling.py)",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for distillation data",
    )

    parser.add_argument(
        "--human-labels",
        help="Optional path to human-labeled CSV for mixing",
    )

    parser.add_argument(
        "--human-text-column",
        default="text",
        help="Column name for text in human data (default: text)",
    )

    parser.add_argument(
        "--human-label-column",
        default="label",
        help="Column name for label in human data (default: label)",
    )

    parser.add_argument(
        "--human-oversample",
        type=float,
        default=3.0,
        help="Factor to oversample human labels (default: 3.0)",
    )

    parser.add_argument(
        "--include-quarantine",
        action="store_true",
        help="Include QUARANTINE tier records (excluded by default)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Create exporter
    exporter = DistillationExporter()

    # Run export
    logger.info("Starting distillation export...")
    stats = exporter.export(
        llm_labeled_csv=args.llm_labels,
        output_path=args.output,
        human_labeled_csv=args.human_labels,
        human_text_column=args.human_text_column,
        human_label_column=args.human_label_column,
        human_oversample=args.human_oversample,
        exclude_quarantine=not args.include_quarantine,
    )

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DISTILLATION EXPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records: {stats['total_records']}")
    logger.info("")
    logger.info("Source distribution:")
    for source, count in stats['source_distribution'].items():
        pct = count / stats['total_records'] * 100
        avg_weight = stats['source_avg_weight'][source]
        logger.info(f"  {source}: {count} ({pct:.1f}%) - avg weight: {avg_weight:.2f}")
    logger.info("")
    logger.info("Tier distribution:")
    for tier, count in stats['tier_distribution'].items():
        pct = count / stats['total_records'] * 100
        logger.info(f"  {tier}: {count} ({pct:.1f}%)")
    logger.info("")
    logger.info(f"Soft labels: {stats['soft_label_count']}")
    logger.info(f"Hard labels: {stats['hard_label_count']}")
    logger.info("")
    logger.info(f"Output saved to: {args.output}")
    logger.info(f"Statistics saved to: {Path(args.output).with_suffix('.stats.json')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
