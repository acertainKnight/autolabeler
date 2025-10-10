"""
Policy Uncertainty News Articles - Phase 1: Rule Learning

This script implements Phase 1 of the two-phase labeling approach for policy uncertainty indicators
in news articles. It labels 5000 examples with rule evolution enabled to learn optimal principles
for 14 binary classification tasks covering all major policy uncertainty indices.

The 14 indicators can be combined to create various policy uncertainty indices:
- EPU Index: is_economic AND is_policy AND is_uncertain
- Trade Policy Uncertainty: is_trade AND is_policy AND is_uncertain
- Geopolitical Risk: is_geopolitical AND is_risk AND is_event
- Climate Policy Uncertainty: is_climate AND is_policy AND is_uncertain
- Monetary Policy Uncertainty: is_monetary AND is_policy AND is_uncertain
- Plus 8 categorical EPU indices using combinations

Budget: $100
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


def sample_data_across_time(df: pd.DataFrame, n_samples: int, date_column: str = None) -> pd.DataFrame:
    """Sample data randomly within each time period to get representative sample.

    Args:
        df: Full DataFrame
        n_samples: Number of samples to extract
        date_column: Column containing timestamps (auto-detected if None)

    Returns:
        DataFrame with n_samples rows sampled temporally
    """
    # Auto-detect date column if not specified
    if date_column is None:
        date_candidates = ['date', 'published', 'timestamp', 'capturetime', 'publication_date']
        for col in date_candidates:
            if col in df.columns:
                date_column = col
                break

    if date_column is None or date_column not in df.columns:
        logger.warning(f"No date column found, using random sampling")
        return df.sample(n=min(n_samples, len(df)), random_state=42)

    # Convert to datetime if needed
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], format='mixed', errors='coerce')

    # Create year-month column for grouping
    df['year_month'] = df[date_column].dt.to_period('M')

    # Get unique months and calculate samples per month
    unique_months = df['year_month'].unique()
    samples_per_month = n_samples // len(unique_months)
    remaining_samples = n_samples % len(unique_months)

    sampled_dfs = []

    for i, month in enumerate(unique_months):
        month_data = df[df['year_month'] == month]

        # Add one extra sample to first few months if there are remaining samples
        month_samples = samples_per_month + (1 if i < remaining_samples else 0)
        month_samples = min(month_samples, len(month_data))  # Don't exceed available data

        if month_samples > 0:
            sampled_month = month_data.sample(n=month_samples, random_state=42)
            sampled_dfs.append(sampled_month)

    # Combine all monthly samples
    sampled_df = pd.concat(sampled_dfs, ignore_index=True).drop('year_month', axis=1)

    # Sort by time for consistent ordering
    sampled_df = sampled_df.sort_values(date_column)

    logger.info(f"Sampled {len(sampled_df)} rows randomly from {len(unique_months)} months")
    logger.info(f"Date range: {sampled_df[date_column].min()} to {sampled_df[date_column].max()}")
    return sampled_df


def run_phase1(
    input_file: str,
    output_file: str,
    task_configs_file: str,
    text_column: str = None,
    n_samples: int = 5000,
    batch_size: int = 50,
    confidence_threshold: float = 0.7,
    budget: float = 5.0
):
    """Phase 1: Label initial sample with rule evolution to learn optimal principles.

    Args:
        input_file: Path to input CSV file
        output_file: Path to save labeled output
        task_configs_file: Path to task configurations JSON
        text_column: Column name containing text to label (auto-detected if None)
        n_samples: Number of samples for Phase 1
        batch_size: Batch size for processing
        confidence_threshold: Threshold for rule evolution triggering
        budget: LLM budget for Phase 1
    """
    logger.info("=" * 80)
    logger.info(f"PHASE 1: Rule Learning ({n_samples} examples)")
    logger.info("=" * 80)

    # Load data
    input_path = Path(input_file)
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} total rows from {input_path}")

    # Auto-detect text column if not specified
    if text_column is None:
        text_candidates = ['text', 'content', 'article', 'headline', 'title', 'body']
        for col in text_candidates:
            if col in df.columns:
                text_column = col
                break
        if text_column is None:
            raise ValueError(f"Could not auto-detect text column. Available columns: {list(df.columns)}")

    logger.info(f"Using text column: {text_column}")

    # Sample across time
    sample_df = sample_data_across_time(df, n_samples)

    # Load task configurations
    config_path = Path(task_configs_file)
    task_configs = load_task_configs(config_path)
    task_list = list(task_configs.keys())
    logger.info(f"Tasks to label ({len(task_list)}): {', '.join(task_list)}")

    # Initialize settings
    settings = Settings()
    settings.llm_budget = budget
    logger.info(f"Using LLM: {settings.llm_model} (provider: {settings.llm_provider})")
    logger.info(f"Budget limit: ${settings.llm_budget:.2f}")

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
        core_rules=initial_principles,  # Protect original config rules from modification
        improvement_strategy="feedback_driven",
        settings=settings
    )

    # Check for existing checkpoint to resume from
    output_path = Path(output_file)
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{output_path.stem}_checkpoint.csv"
    checkpoint_rules_file = checkpoint_dir / f"{output_path.stem}_rules_checkpoint.json"

    start_batch = 0
    all_results = []

    # Resume from checkpoint if exists
    if checkpoint_file.exists():
        logger.info(f"Found checkpoint at {checkpoint_file}, resuming...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        all_results.append(checkpoint_df)
        start_batch = len(checkpoint_df) // batch_size
        logger.info(f"Resuming from batch {start_batch + 1} (already processed {len(checkpoint_df)} rows)")

        # Load checkpoint rules if available
        if checkpoint_rules_file.exists():
            with open(checkpoint_rules_file) as f:
                checkpoint_rules = json.load(f)
            constitutional.update_principles(checkpoint_rules)
            multi_agent.update_task_configs(checkpoint_rules)
            logger.info("Loaded checkpoint rules")

    # Process in batches with rule evolution
    total_batches = (len(sample_df) + batch_size - 1) // batch_size

    for batch_idx in range(start_batch, total_batches):
        # Check budget before starting batch
        if multi_agent.cost_tracker and multi_agent.cost_tracker.is_budget_exceeded():
            logger.warning("Budget exceeded, stopping processing")
            logger.info(f"Processed {batch_idx * batch_size} / {len(sample_df)} rows before budget limit")
            break

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sample_df))
        batch_df = sample_df.iloc[start_idx:end_idx].copy()

        logger.info(f"\nBatch {batch_idx + 1}/{total_batches} (rows {start_idx}-{end_idx})")

        try:
            # Label batch
            labeled_batch = multi_agent.label_with_agents(batch_df, text_column, task_list)
            all_results.append(labeled_batch)

            # Save checkpoint after each batch
            checkpoint_progress = pd.concat(all_results, ignore_index=True)
            checkpoint_progress.to_csv(checkpoint_file, index=False)
            logger.info(f"  Checkpoint saved: {len(checkpoint_progress)} rows")

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

                    # Save rules checkpoint
                    with open(checkpoint_rules_file, 'w') as f:
                        json.dump(updated_rules, f, indent=2)

                    logger.info("  Rules updated for next batch")

        except Exception as e:
            from autolabeler.core.utils.budget_tracker import BudgetExceededError

            # If budget exceeded, save progress and stop gracefully
            if isinstance(e, BudgetExceededError):
                logger.warning(f"Budget exceeded: {e}")
                logger.info(f"Progress saved to checkpoint: {checkpoint_file}")
                logger.info(f"Processed {len(all_results)} batches before budget limit")
                break

            # For other errors, log and re-raise
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            logger.info(f"Progress saved to checkpoint: {checkpoint_file}")
            raise

    # Combine results
    final_df = pd.concat(all_results, ignore_index=True)

    # Save Phase 1 results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"\nPhase 1 complete: {len(final_df)} rows labeled")
    logger.info(f"Results saved to: {output_path}")

    # Clean up checkpoint files on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Checkpoint file removed (processing complete)")
    if checkpoint_rules_file.exists():
        checkpoint_rules_file.unlink()
        logger.info("Checkpoint rules file removed (processing complete)")

    # Log cost summary
    multi_agent.log_cost_summary()

    # Get and save learned rules
    learned_rules = constitutional.get_principles()
    stats = rule_evolution.get_improvement_stats()

    logger.info("\nRule Evolution Statistics:")
    logger.info(f"  Total patterns identified: {stats['total_patterns_identified']}")
    logger.info(f"  Rules generated: {stats['rules_generated']}")
    logger.info(f"  Pattern types: {stats['pattern_types']}")

    # Save learned rules to JSON
    rules_file = output_path.parent / f"{output_path.stem}_learned_rules.json"
    with open(rules_file, 'w') as f:
        json.dump(learned_rules, f, indent=2)
    logger.info(f"Learned rules saved to: {rules_file}")

    # Log confidence statistics
    for task in task_list:
        conf_col = f"confidence_{task}"
        mean_conf = final_df[conf_col].mean()
        low_conf = (final_df[conf_col] < confidence_threshold).sum()
        logger.info(f"  {task}: mean confidence={mean_conf:.3f}, uncertain={low_conf}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Policy Uncertainty News - Phase 1: Rule Learning")
    parser.add_argument("--input", required=True, help="Input CSV file with news articles")
    parser.add_argument("--output", required=True, help="Output CSV file for labeled data")
    parser.add_argument("--config", default="configs/policy_uncertainty_news_tasks.json",
                       help="Task configuration JSON file")
    parser.add_argument("--text-column", default=None, help="Column containing text (auto-detected if not specified)")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples for Phase 1")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--budget", type=float, default=100.0, help="LLM budget for Phase 1")

    args = parser.parse_args()

    run_phase1(
        input_file=args.input,
        output_file=args.output,
        task_configs_file=args.config,
        text_column=args.text_column,
        n_samples=args.samples,
        batch_size=args.batch_size,
        budget=args.budget
    )
