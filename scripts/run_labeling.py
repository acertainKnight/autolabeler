#!/usr/bin/env python3
"""Unified entry point for LLM-based data labeling.

This is the ONE script to run labeling for any dataset.
Configured via YAML files in configs/{dataset}.yaml

Usage:
    # Single CSV file
    python scripts/run_labeling.py \\
        --dataset fed_headlines \\
        --input datasets/headlines.csv \\
        --output outputs/fed_headlines/labeled.csv

    # Directory of JSONL splits (all files processed together)
    python scripts/run_labeling.py \\
        --dataset fed_headlines \\
        --input datasets/headlines/ \\
        --output outputs/fed_headlines/labeled.csv

    # Resume an interrupted run
    python scripts/run_labeling.py \\
        --dataset tpu \\
        --input datasets/tpu_articles.csv \\
        --output outputs/tpu/labeled.csv \\
        --resume

    # With budget limit
    python scripts/run_labeling.py \\
        --dataset fed_headlines \\
        --input datasets/headlines.csv \\
        --output outputs/labeled.csv \\
        --max-budget 10.0 \\
        --batch-size 20

Evidence-Based Architecture:
    - Heterogeneous jury (NeurIPS 2025, A-HMAD)
    - Confidence-weighted voting (Amazon 2024)
    - Candidate annotation for disagreements (ACL 2025)
    - Tier assignment (ACCEPT/ACCEPT-M/SOFT/QUARANTINE)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sibyls.core.dataset_config import DatasetConfig
from src.sibyls.core.labeling.pipeline import LabelingPipeline
from src.sibyls.core.llm_providers.providers import load_provider_module
from src.sibyls.core.prompts.registry import PromptRegistry
from src.sibyls.core.quality.confidence_scorer import ConfidenceScorer
from src.sibyls.core.utils.data_utils import load_input_data


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, format=log_format, level=level)


async def run_labeling(args):
    """Run labeling pipeline."""

    # Load any custom provider modules before building the pipeline so that
    # register_provider() calls within them are in effect when get_provider() runs.
    for module_path in args.provider_module or []:
        load_provider_module(module_path)

    # Load dataset configuration
    config_path = Path(f"configs/{args.dataset}.yaml")
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        logger.error("Available configs:")
        for f in Path("configs").glob("*.yaml"):
            logger.error(f"  - {f.stem}")
        return 1
    
    config = DatasetConfig.from_yaml(config_path)
    logger.info(f"Loaded config for {config.name}")
    logger.info(f"  Labels: {config.labels}")
    logger.info(f"  Jury models: {len(config.jury_models)}")
    logger.info(f"  Relevancy gate: {config.use_relevancy_gate}")
    logger.info(f"  Candidate annotation: {config.use_candidate_annotation}")
    
    # Load prompts
    prompts = PromptRegistry(args.dataset)
    logger.info(f"Loaded prompts from prompts/{args.dataset}/")
    
    # Initialize confidence scorer
    confidence_scorer = ConfidenceScorer()
    
    # Initialize pipeline
    pipeline = LabelingPipeline(
        config,
        prompts,
        confidence_scorer,
        max_budget=args.max_budget or 0.0,
    )
    
    # Load input data (single file or directory of files)
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return 1

    try:
        df = load_input_data(
            input_path,
            input_format=config.input_format,
            text_column=config.text_column,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1
    
    # Apply limits
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to first {args.limit} rows")
    
    # Run labeling
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.max_budget:
        logger.info(f"Budget ceiling set to ${args.max_budget:.2f}")
    
    logger.info("Starting labeling pipeline...")
    logger.info(f"Output will be saved to: {output_path}")
    
    results_df = await pipeline.label_dataframe(
        df=df,
        output_path=output_path,
        resume=args.resume,
    )
    
    cost_info = f"${pipeline.cost_tracker.total_cost:.4f}"
    if args.max_budget:
        cost_info += f" of ${args.max_budget:.2f} budget"
    logger.info(
        f"Labeling complete! {len(results_df)} rows saved to {output_path} "
        f"(total cost: {cost_info}, {pipeline.cost_tracker.call_count} API calls)"
    )
    
    # Also export soft labels to a separate JSONL file for distillation
    jsonl_path = output_path.with_suffix('.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in results_df.iterrows():
            record = {
                "text": row[config.text_column],
                "hard_label": row["label"],
                "tier": row["tier"],
                "training_weight": row["training_weight"],
                "agreement": row["agreement"],
            }
            # Parse soft_label JSON
            if pd.notna(row.get("soft_label")):
                record["soft_label"] = json.loads(row["soft_label"])
            else:
                # Fallback: create one-hot distribution
                record["soft_label"] = {row["label"]: 1.0}
            
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Soft labels exported to {jsonl_path}")
    
    # Summary statistics
    tier_counts = results_df['tier'].value_counts()
    logger.info("Tier Distribution:")
    for tier, count in tier_counts.items():
        pct = count / len(results_df) * 100
        logger.info(f"  {tier}: {count} ({pct:.1f}%)")
    
    label_counts = results_df['label'].value_counts()
    logger.info("Label Distribution:")
    for label, count in label_counts.items():
        pct = count / len(results_df) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    agreement_counts = results_df['agreement'].value_counts()
    logger.info("Agreement Distribution:")
    for agreement, count in agreement_counts.items():
        pct = count / len(results_df) * 100
        logger.info(f"  {agreement}: {count} ({pct:.1f}%)")
    
    return 0


def main():
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(
        description="Unified LLM-based data labeling service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match a config file: configs/{dataset}.yaml)"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input file or directory. When a directory is given, all files "
            "matching the config's input_format (csv/jsonl) are loaded and "
            "concatenated."
        ),
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already-labeled rows)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for processing (overrides config)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to first N rows (for testing)"
    )
    
    parser.add_argument(
        "--max-budget",
        type=float,
        help="Maximum budget in USD (stops when reached)"
    )
    
    parser.add_argument(
        "--provider-module",
        action="append",
        metavar="MODULE",
        help=(
            "Dotted Python module path to import before running the pipeline "
            "(e.g., my_company.llm_proxy). The module must call "
            "register_provider() at import time. Can be repeated to load "
            "multiple modules."
        ),
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Check for required environment variables
    required_env = []
    if not os.getenv("OPENAI_API_KEY"):
        required_env.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        required_env.append("ANTHROPIC_API_KEY")
    if not os.getenv("GOOGLE_API_KEY"):
        required_env.append("GOOGLE_API_KEY")
    
    if required_env:
        logger.warning(
            f"Missing environment variables: {', '.join(required_env)}. "
            "Some providers may fail."
        )
    
    try:
        exit_code = asyncio.run(run_labeling(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
