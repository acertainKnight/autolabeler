#!/usr/bin/env python3
"""Optimize labeling prompts using DSPy MIPROv2 and labeled data.

Takes your existing hand-crafted prompts and a dataset with human labels,
then uses MIPROv2 to find better prompt phrasings and few-shot examples.

Usage:
    # Basic optimization using OpenRouter
    python scripts/optimize_prompts.py \
        --dataset fed_headlines \
        --labeled-data datasets/human_labeled_LIVE_20251103.csv \
        --text-column headline \
        --label-column label_hawk_dove

    # With custom model and budget
    python scripts/optimize_prompts.py \
        --dataset fed_headlines \
        --labeled-data datasets/human_labeled_LIVE_20251103.csv \
        --text-column headline \
        --label-column label_hawk_dove \
        --model openrouter/google/gemini-2.5-flash \
        --num-trials 30 \
        --num-candidates 15

    # Write improvements to prompt files for review
    python scripts/optimize_prompts.py \
        --dataset fed_headlines \
        --labeled-data datasets/human_labeled_LIVE_20251103.csv \
        --text-column headline \
        --label-column label_hawk_dove \
        --write-prompts

Output:
    - Console report comparing baseline vs optimized accuracy
    - JSON report saved to outputs/{dataset}/dspy_optimization.json
    - Optionally: prompts/{dataset}/rules_dspy.md and examples_dspy.md
      (review these and replace the originals if they're better)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sibyls.core.dataset_config import DatasetConfig
from src.sibyls.core.optimization.dspy_optimizer import (
    DSPyConfig,
    DSPyOptimizer,
)
from src.sibyls.core.prompts.registry import PromptRegistry


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, level="DEBUG" if verbose else "INFO")


def load_and_split(
    path: str,
    text_column: str,
    label_column: str,
    val_split: float = 0.2,
    max_rows: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labeled data and split into train/val.

    Parameters:
        path: Path to CSV or JSONL file.
        text_column: Column containing text.
        label_column: Column containing labels.
        val_split: Fraction of data for validation.
        max_rows: Limit total rows (for cost control).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df).
    """
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. "
            f"Available: {df.columns.tolist()}"
        )
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    # Drop rows with missing labels
    df = df.dropna(subset=[text_column, label_column])
    logger.info(f"Loaded {len(df)} labeled rows from {path}")

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
        logger.info(f"Sampled down to {max_rows} rows")

    # Stratified-ish split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_split))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")

    # Label distribution
    for name, subset in [("Train", train_df), ("Val", val_df)]:
        dist = subset[label_column].value_counts().to_dict()
        logger.info(f"  {name} labels: {dist}")

    return train_df, val_df


def resolve_api_key(model_name: str) -> tuple[str | None, str | None]:
    """Resolve the API key and base URL for the given model.

    Parameters:
        model_name: Model name in litellm format.

    Returns:
        Tuple of (api_key, api_base).
    """
    if model_name.startswith("openrouter/") or "openrouter" in model_name:
        key = os.getenv("OPENROUTER_API_KEY")
        return key, "https://openrouter.ai/api/v1"

    if "anthropic" in model_name or "claude" in model_name:
        return os.getenv("ANTHROPIC_API_KEY"), None

    if "gemini" in model_name or "google" in model_name:
        return os.getenv("GOOGLE_API_KEY"), None

    # Default: OpenAI
    return os.getenv("OPENAI_API_KEY"), None


def main() -> int:
    """Parse arguments and run optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize labeling prompts using DSPy MIPROv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match configs/{dataset}.yaml and prompts/{dataset}/)",
    )
    parser.add_argument(
        "--labeled-data",
        required=True,
        help="Path to CSV/JSONL with human labels",
    )
    parser.add_argument(
        "--text-column",
        required=True,
        help="Column name containing text to classify",
    )
    parser.add_argument(
        "--label-column",
        required=True,
        help="Column name containing ground-truth labels",
    )
    parser.add_argument(
        "--model",
        default="openrouter/google/gemini-2.5-flash",
        help="LLM model for optimization (litellm format, default: openrouter/google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split fraction (default: 0.2)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Max rows to use (controls cost, default: 200)",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=7, help="MIPROv2 candidates (default: 7)"
    )
    parser.add_argument(
        "--num-trials", type=int, default=15, help="MIPROv2 trials (default: 15)"
    )
    parser.add_argument(
        "--max-demos", type=int, default=6, help="Max few-shot demos (default: 6)"
    )
    parser.add_argument(
        "--write-prompts",
        action="store_true",
        help="Write optimized prompts to prompts/{dataset}/ as *_dspy.md files for review",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path (default: outputs/{dataset}/dspy_optimization.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Load dataset config and prompts
    config_path = Path(f"configs/{args.dataset}.yaml")
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1

    dataset_config = DatasetConfig.from_yaml(config_path)
    prompt_registry = PromptRegistry(args.dataset)

    logger.info(f"Dataset: {dataset_config.name}")
    logger.info(f"Labels: {dataset_config.labels}")

    # Load and split labeled data
    train_df, val_df = load_and_split(
        args.labeled_data,
        args.text_column,
        args.label_column,
        val_split=args.val_split,
        max_rows=args.max_rows,
    )

    # Resolve API credentials
    api_key, api_base = resolve_api_key(args.model)
    if not api_key:
        logger.error(
            f"No API key found for model '{args.model}'. "
            "Set the appropriate environment variable."
        )
        return 1

    # Configure DSPy
    dspy_config = DSPyConfig(
        model_name=args.model,
        api_key=api_key,
        api_base=api_base,
        num_candidates=args.num_candidates,
        num_trials=args.num_trials,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
    )

    # Run optimization
    optimizer = DSPyOptimizer(dspy_config)
    result = optimizer.optimize(
        train_df=train_df,
        val_df=val_df,
        text_column=args.text_column,
        label_column=args.label_column,
        dataset_config=dataset_config,
        prompt_registry=prompt_registry,
    )

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline accuracy:  {result.baseline_accuracy:.1%}")
    logger.info(f"Optimized accuracy: {result.optimized_accuracy:.1%}")
    logger.info(f"Improvement:        {result.improvement:+.1%}")
    logger.info(f"Estimated cost:     ${result.optimization_cost:.2f}")
    logger.info(f"Converged:          {result.converged}")
    logger.info(f"Examples selected:  {len(result.selected_examples)}")

    if result.selected_examples:
        logger.info("")
        logger.info("Selected few-shot examples:")
        for i, ex in enumerate(result.selected_examples, 1):
            logger.info(f"  {i}. [{ex.get('label', '?')}] {ex.get('text', '')[:80]}...")

    # Save JSON report
    output_path = Path(
        args.output or f"outputs/{args.dataset}/dspy_optimization.json"
    )
    optimizer.save_result(result, output_path)

    # Optionally write prompt files
    if args.write_prompts:
        written = optimizer.update_prompt_files(result, prompt_registry)
        logger.info("")
        logger.info("Wrote optimized prompt files for review:")
        for name, path in written.items():
            logger.info(f"  {path}")
        logger.info("")
        logger.info(
            "Review these files and replace the originals if they improve results:"
        )
        logger.info(f"  diff prompts/{args.dataset}/rules.md prompts/{args.dataset}/rules_dspy.md")
        logger.info(
            f"  diff prompts/{args.dataset}/examples.md prompts/{args.dataset}/examples_dspy.md"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
