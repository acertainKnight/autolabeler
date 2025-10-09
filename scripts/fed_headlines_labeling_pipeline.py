"""
Federal Reserve Headlines Multi-Label Classification Pipeline

This script implements a two-phase labeling approach:
1. Phase 1: Label 1000 examples with rule evolution enabled to learn optimal principles
2. Phase 2: Label remaining 19,000 examples using learned rules

The pipeline handles 11 classification tasks:
- relevancy (relevant/not_relevant)
- hawk_dove (-2 to 2 scale)
- speaker (7 categories)
- 7 binary topic indicators (inflation, labor_market, growth, etc.)
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

# Add parent directory to path to import autolabeler
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autolabeler.config import Settings
from autolabeler.agents import MultiAgentService
from autolabeler.constitutional import ConstitutionalService
from autolabeler.active_learning import RuleEvolutionService


def load_task_configs(config_path: Path) -> dict:
    """Load task configurations from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def sample_data_across_time(df: pd.DataFrame, n_samples: int, date_column: str = "capturetime") -> pd.DataFrame:
    """Sample data uniformly across time to get representative sample.

    Args:
        df: Full DataFrame
        n_samples: Number of samples to extract
        date_column: Column containing timestamps

    Returns:
        DataFrame with n_samples rows sampled across time
    """
    # Convert to datetime if needed
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort by time
    df = df.sort_values(date_column)

    # Take evenly spaced samples
    indices = [int(i * len(df) / n_samples) for i in range(n_samples)]
    sampled_df = df.iloc[indices].copy()

    logger.info(f"Sampled {len(sampled_df)} rows from {df[date_column].min()} to {df[date_column].max()}")
    return sampled_df


def phase1_learn_rules(
    input_file: Path,
    output_file: Path,
    task_configs_file: Path,
    n_samples: int = 1000,
    batch_size: int = 50,
    confidence_threshold: float = 0.7,
    settings: Settings = None
) -> dict:
    """Phase 1: Label initial sample with rule evolution to learn optimal principles.

    Returns:
        Dictionary of learned rules/principles
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: Rule Learning (1000 examples)")
    logger.info("=" * 80)

    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} total rows from {input_file}")

    # Sample across time
    sample_df = sample_data_across_time(df, n_samples)

    # Load task configurations
    task_configs = load_task_configs(task_configs_file)
    task_list = list(task_configs.keys())
    logger.info(f"Tasks to label: {', '.join(task_list)}")

    # Initialize settings
    if settings is None:
        settings = Settings()
    logger.info(f"Using LLM: {settings.llm_model} (provider: {settings.llm_provider})")

    # Initialize services
    multi_agent = MultiAgentService(settings, task_configs)

    initial_principles = {
        task: config.get("principles", [])
        for task, config in task_configs.items()
    }
    constitutional = ConstitutionalService(
        principles=initial_principles,
        enforcement_level="strict"
    )

    rule_evolution = RuleEvolutionService(
        initial_rules=initial_principles,
        improvement_strategy="feedback_driven",
        settings=settings
    )

    # Process in batches with rule evolution
    all_results = []
    total_batches = (len(sample_df) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sample_df))
        batch_df = sample_df.iloc[start_idx:end_idx].copy()

        logger.info(f"\nBatch {batch_idx + 1}/{total_batches} (rows {start_idx}-{end_idx})")

        # Label batch
        labeled_batch = multi_agent.label_with_agents(batch_df, "headline", task_list)
        all_results.append(labeled_batch)

        # Rule evolution: check for uncertain predictions (skip last batch)
        if batch_idx < total_batches - 1:
            uncertain_rows = []
            for task in task_list:
                conf_col = f"confidence_{task}"
                uncertain = labeled_batch[labeled_batch[conf_col] < confidence_threshold]
                if len(uncertain) > 0:
                    uncertain_rows.append(uncertain)

            if uncertain_rows:
                feedback_df = pd.concat(uncertain_rows, ignore_index=True)
                logger.info(f"  Found {len(feedback_df)} uncertain predictions, improving rules...")

                # Improve rules
                current_rules = constitutional.get_principles()
                updated_rules = rule_evolution.improve_rules(current_rules, feedback_df)

                # Update services
                constitutional.update_principles(updated_rules)
                multi_agent.update_task_configs(updated_rules)

                logger.info("  Rules updated for next batch")

    # Combine results
    final_df = pd.concat(all_results, ignore_index=True)

    # Save Phase 1 results
    final_df.to_csv(output_file, index=False)
    logger.info(f"\nPhase 1 complete: {len(final_df)} rows labeled")
    logger.info(f"Results saved to: {output_file}")

    # Get and save learned rules
    learned_rules = constitutional.get_principles()
    stats = rule_evolution.get_improvement_stats()

    logger.info("\nRule Evolution Statistics:")
    logger.info(f"  Total patterns identified: {stats['total_patterns_identified']}")
    logger.info(f"  Rules generated: {stats['rules_generated']}")
    logger.info(f"  Pattern types: {stats['pattern_types']}")

    # Log confidence statistics
    for task in task_list:
        conf_col = f"confidence_{task}"
        mean_conf = final_df[conf_col].mean()
        low_conf = (final_df[conf_col] < confidence_threshold).sum()
        logger.info(f"  {task}: mean confidence={mean_conf:.3f}, uncertain={low_conf}")

    return learned_rules


def phase2_full_labeling(
    input_file: Path,
    output_file: Path,
    task_configs_file: Path,
    learned_rules: dict,
    skip_first_n: int = 1000,
    max_rows: int = 20000,
    batch_size: int = 100,
    settings: Settings = None
):
    """Phase 2: Label remaining data using learned rules (no rule evolution).

    Args:
        input_file: Path to full dataset
        output_file: Path to save labeled output
        task_configs_file: Path to task configurations
        learned_rules: Dictionary of learned principles from Phase 1
        skip_first_n: Number of rows to skip (already labeled in Phase 1)
        max_rows: Maximum total rows to label
        batch_size: Batch size for processing
        settings: LLM settings
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"PHASE 2: Full Labeling ({max_rows - skip_first_n} examples)")
    logger.info("=" * 80)

    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} total rows from {input_file}")

    # Sample across time for full dataset
    full_sample = sample_data_across_time(df, max_rows)

    # Skip rows already labeled in Phase 1
    phase2_df = full_sample.iloc[skip_first_n:].copy()
    logger.info(f"Processing rows {skip_first_n} to {len(full_sample)} ({len(phase2_df)} rows)")

    # Load task configurations with learned rules
    task_configs = load_task_configs(task_configs_file)

    # Update task configs with learned rules
    for task, rules in learned_rules.items():
        if task in task_configs:
            task_configs[task]["principles"] = rules

    task_list = list(task_configs.keys())

    # Initialize settings
    if settings is None:
        settings = Settings()

    # Initialize services (no rule evolution in Phase 2)
    multi_agent = MultiAgentService(settings, task_configs)
    logger.info(f"Using learned rules from Phase 1 for {len(task_list)} tasks")

    # Process in batches
    all_results = []
    total_batches = (len(phase2_df) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(phase2_df))
        batch_df = phase2_df.iloc[start_idx:end_idx].copy()

        logger.info(f"Batch {batch_idx + 1}/{total_batches} (rows {start_idx + skip_first_n}-{end_idx + skip_first_n})")

        # Label batch
        labeled_batch = multi_agent.label_with_agents(batch_df, "headline", task_list)
        all_results.append(labeled_batch)

        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Processed {(batch_idx + 1) * batch_size} / {len(phase2_df)} rows")

    # Combine results
    final_df = pd.concat(all_results, ignore_index=True)

    # Save Phase 2 results
    final_df.to_csv(output_file, index=False)
    logger.info(f"\nPhase 2 complete: {len(final_df)} rows labeled")
    logger.info(f"Results saved to: {output_file}")

    # Log confidence statistics
    for task in task_list:
        conf_col = f"confidence_{task}"
        mean_conf = final_df[conf_col].mean()
        logger.info(f"  {task}: mean confidence={mean_conf:.3f}")


def run_complete_pipeline(
    input_file: str = "datasets/fed_data_full.csv",
    output_dir: str = "outputs/fed_headlines",
    task_configs_file: str = "configs/fed_headlines_tasks.json",
    phase1_samples: int = 1000,
    phase2_total: int = 20000,
    phase1_batch_size: int = 50,
    phase2_batch_size: int = 100,
    confidence_threshold: float = 0.7
):
    """Run complete two-phase labeling pipeline.

    Args:
        input_file: Path to input CSV
        output_dir: Directory for output files
        task_configs_file: Path to task configuration JSON
        phase1_samples: Number of samples for Phase 1 rule learning
        phase2_total: Total samples to label (includes Phase 1)
        phase1_batch_size: Batch size for Phase 1
        phase2_batch_size: Batch size for Phase 2
        confidence_threshold: Threshold for uncertain predictions
    """
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / input_file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    configs_path = project_root / task_configs_file

    # Phase 1 outputs
    phase1_output = output_path / "phase1_labeled_1000.csv"
    phase1_rules = output_path / "phase1_learned_rules.json"

    # Phase 2 outputs
    phase2_output = output_path / "phase2_labeled_19000.csv"
    combined_output = output_path / "fed_headlines_labeled_20000.csv"

    logger.info("Federal Reserve Headlines Labeling Pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Task configs: {configs_path}")

    # Initialize settings
    settings = Settings()

    # Phase 1: Learn rules
    learned_rules = phase1_learn_rules(
        input_file=input_path,
        output_file=phase1_output,
        task_configs_file=configs_path,
        n_samples=phase1_samples,
        batch_size=phase1_batch_size,
        confidence_threshold=confidence_threshold,
        settings=settings
    )

    # Save learned rules
    with open(phase1_rules, 'w') as f:
        json.dump(learned_rules, f, indent=2)
    logger.info(f"\nLearned rules saved to: {phase1_rules}")

    # Phase 2: Full labeling
    phase2_full_labeling(
        input_file=input_path,
        output_file=phase2_output,
        task_configs_file=configs_path,
        learned_rules=learned_rules,
        skip_first_n=phase1_samples,
        max_rows=phase2_total,
        batch_size=phase2_batch_size,
        settings=settings
    )

    # Combine Phase 1 and Phase 2 results
    logger.info("\n" + "=" * 80)
    logger.info("Combining Phase 1 and Phase 2 results")
    logger.info("=" * 80)

    phase1_df = pd.read_csv(phase1_output)
    phase2_df = pd.read_csv(phase2_output)

    combined_df = pd.concat([phase1_df, phase2_df], ignore_index=True)
    combined_df.to_csv(combined_output, index=False)

    logger.info(f"\nFinal combined dataset: {len(combined_df)} rows")
    logger.info(f"Saved to: {combined_output}")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  Phase 1 (1000 rows): {phase1_output}")
    logger.info(f"  Phase 1 rules: {phase1_rules}")
    logger.info(f"  Phase 2 (19000 rows): {phase2_output}")
    logger.info(f"  Combined (20000 rows): {combined_output}")


if __name__ == "__main__":
    # Run the complete pipeline
    run_complete_pipeline()
