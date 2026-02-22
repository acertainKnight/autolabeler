#!/usr/bin/env python3
"""Generate and evaluate labeling programs using ALCHEmist approach.

This script implements the ALCHEmist workflow:
1. Load high-confidence jury-labeled data as "seed" examples
2. Use few-shot LLM to generate candidate Python labeling programs
3. Evaluate programs on seed data (precision, recall, coverage)
4. Keep only high-quality programs
5. Optionally apply programs to new data

Usage:
    # Generate and evaluate programs
    python scripts/generate_programs.py \\
        --config configs/fed_headlines.yaml \\
        --seed-data outputs/fed_headlines/labeled.csv \\
        --output outputs/fed_headlines/programs.json \\
        --n-programs 15 \\
        --precision-threshold 0.85

    # Apply saved programs to new data
    python scripts/generate_programs.py \\
        --config configs/fed_headlines.yaml \\
        --load-programs outputs/fed_headlines/programs.json \\
        --apply-to datasets/unlabeled.csv \\
        --output outputs/fed_headlines/program_labeled.csv

The seed data should be high-confidence jury labels (e.g., ACCEPT tier with
agreement >= 0.8) to ensure programs learn from reliable examples.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sibyls.core.dataset_config import DatasetConfig
from src.sibyls.core.llm_providers.providers import get_provider
from src.sibyls.core.prompts.prompt_registry import PromptRegistry
from src.sibyls.core.labeling.program_generation import (
    ProgramGenerator,
    ProgramLabeler,
)
import asyncio


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, level="DEBUG" if verbose else "INFO")


async def generate_and_evaluate(
    config: DatasetConfig,
    seed_data_path: str,
    output_path: str,
    n_programs: int,
    precision_threshold: float,
    coverage_threshold: float,
    confidence_threshold: float,
) -> None:
    """Generate and evaluate programs."""
    # Load seed data
    logger.info(f"Loading seed data from {seed_data_path}")
    seed_df = pd.read_csv(seed_data_path)
    
    # Filter to high-confidence examples
    if "confidence" in seed_df.columns and "tier" in seed_df.columns:
        orig_len = len(seed_df)
        seed_df = seed_df[
            (seed_df["confidence"] >= confidence_threshold) &
            (seed_df["tier"].isin(["ACCEPT", "ACCEPT_verified"]))
        ]
        logger.info(
            f"Filtered seed data: {len(seed_df)} / {orig_len} examples "
            f"(confidence >= {confidence_threshold}, tier = ACCEPT)"
        )
    
    if len(seed_df) < 10:
        logger.error("Need at least 10 high-confidence seed examples")
        return
    
    # Ensure required columns
    if "text" not in seed_df.columns:
        # Try common alternatives
        for col in ["headline", "content", "body"]:
            if col in seed_df.columns:
                seed_df["text"] = seed_df[col]
                break
        else:
            logger.error("Seed data must have 'text' column")
            return
    
    # Initialize components
    prompts = PromptRegistry(config.prompt_dir)
    
    # Use first jury model for program generation
    gen_model = config.jury_models[0]
    gen_provider = get_provider(gen_model.provider)
    
    logger.info(f"Using {gen_model.provider}::{gen_model.name} for program generation")
    
    # Generate programs
    generator = ProgramGenerator(gen_provider, gen_model, prompts, config)
    programs = await generator.generate(
        seed_df=seed_df,
        n_programs=n_programs,
        max_examples_per_class=5,
    )
    
    if not programs:
        logger.error("No programs generated")
        return
    
    # Evaluate programs
    labeler = ProgramLabeler(
        precision_threshold=precision_threshold,
        coverage_threshold=coverage_threshold,
    )
    
    stats = labeler.evaluate_programs(programs, seed_df)
    
    # Save results
    labeler.save_programs(output_path)
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROGRAM GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Generated: {len(programs)} programs")
    logger.info(f"Kept: {len(labeler.programs)} programs")
    logger.info("")
    logger.info("Kept programs:")
    for prog in labeler.programs:
        s = prog["stats"]
        logger.info(
            f"  [{s['program_id']}] {s['description'][:60]} | "
            f"P={s['precision']:.2f} R={s['recall']:.2f} C={s['coverage']:.2f}"
        )
    logger.info("")
    logger.info(f"Results saved to: {output_path}")


def apply_programs(
    load_path: str,
    apply_data_path: str,
    output_path: str,
) -> None:
    """Apply saved programs to new data."""
    # Load programs
    logger.info(f"Loading programs from {load_path}")
    labeler = ProgramLabeler.load_programs(load_path)
    
    # Load data
    logger.info(f"Loading data from {apply_data_path}")
    df = pd.read_csv(apply_data_path)
    
    # Ensure text column
    if "text" not in df.columns:
        for col in ["headline", "content", "body"]:
            if col in df.columns:
                df["text"] = df[col]
                break
        else:
            logger.error("Data must have 'text' column")
            return
    
    # Apply programs
    result_df = labeler.apply_programs(df, return_confidence=True)
    
    # Save
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved program-labeled data to {output_path}")
    
    # Print summary
    coverage = (result_df["program_label"].notna()).sum() / len(result_df)
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROGRAM APPLICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total examples: {len(result_df)}")
    logger.info(f"Labeled by programs: {(result_df['program_label'].notna()).sum()}")
    logger.info(f"Coverage: {coverage:.2%}")
    logger.info(f"Avg confidence (where labeled): {result_df['program_confidence'].mean():.2f}")


def main() -> int:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(
        description="Generate and evaluate labeling programs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to dataset config YAML",
    )

    # Generation mode
    gen_group = parser.add_argument_group("program generation")
    gen_group.add_argument(
        "--seed-data",
        help="Path to high-confidence labeled seed data CSV",
    )
    gen_group.add_argument(
        "--n-programs",
        type=int,
        default=15,
        help="Number of programs to generate (default: 15)",
    )
    gen_group.add_argument(
        "--precision-threshold",
        type=float,
        default=0.8,
        help="Minimum precision to keep program (default: 0.8)",
    )
    gen_group.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.1,
        help="Minimum coverage to keep program (default: 0.1)",
    )
    gen_group.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for seed examples (default: 0.7)",
    )

    # Application mode
    app_group = parser.add_argument_group("program application")
    app_group.add_argument(
        "--load-programs",
        help="Path to saved programs JSON (for application mode)",
    )
    app_group.add_argument(
        "--apply-to",
        help="Path to data CSV to apply programs to",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output path (programs.json or labeled.csv)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load config
    config = DatasetConfig.from_yaml(args.config)

    # Determine mode
    if args.load_programs and args.apply_to:
        # Application mode
        apply_programs(
            load_path=args.load_programs,
            apply_data_path=args.apply_to,
            output_path=args.output,
        )
    elif args.seed_data:
        # Generation mode
        asyncio.run(generate_and_evaluate(
            config=config,
            seed_data_path=args.seed_data,
            output_path=args.output,
            n_programs=args.n_programs,
            precision_threshold=args.precision_threshold,
            coverage_threshold=args.coverage_threshold,
            confidence_threshold=args.confidence_threshold,
        ))
    else:
        logger.error(
            "Must provide either --seed-data (generation mode) or "
            "--load-programs + --apply-to (application mode)"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
