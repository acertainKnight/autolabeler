"""
Policy Uncertainty News Articles - Phase 2: Full Labeling

This script implements Phase 2 of the two-phase labeling approach. It applies learned rules
from Phase 1 to label the remaining examples without rule evolution for cost efficiency.

Budget: $50
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


def load_task_configs(config_path: Path) -> dict:
    """Load task configurations from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def sample_data_across_time(df: pd.DataFrame, n_samples: int, date_column: str = None) -> pd.DataFrame:
    """Sample data randomly within each time period to get representative sample."""
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
        month_samples = samples_per_month + (1 if i < remaining_samples else 0)
        month_samples = min(month_samples, len(month_data))

        if month_samples > 0:
            sampled_month = month_data.sample(n=month_samples, random_state=42)
            sampled_dfs.append(sampled_month)

    # Combine all monthly samples
    sampled_df = pd.concat(sampled_dfs, ignore_index=True).drop('year_month', axis=1)
    sampled_df = sampled_df.sort_values(date_column)

    logger.info(f"Sampled {len(sampled_df)} rows randomly from {len(unique_months)} months")
    logger.info(f"Date range: {sampled_df[date_column].min()} to {sampled_df[date_column].max()}")
    return sampled_df


def run_phase2(
    input_file: str,
    output_file: str,
    task_configs_file: str,
    learned_rules_file: str,
    text_column: str = None,
    skip_first_n: int = 5000,
    max_rows: int = 25000,
    batch_size: int = 100,
    budget: float = 50.0
):
    """Phase 2: Label remaining data using learned rules (no rule evolution).

    Args:
        input_file: Path to input CSV file
        output_file: Path to save labeled output
        task_configs_file: Path to task configurations JSON
        learned_rules_file: Path to learned rules from Phase 1
        text_column: Column name containing text to label (auto-detected if None)
        skip_first_n: Number of rows to skip (already labeled in Phase 1)
        max_rows: Maximum total rows to label
        batch_size: Batch size for processing
        budget: LLM budget for Phase 2
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"PHASE 2: Full Labeling ({max_rows - skip_first_n} examples)")
    logger.info("=" * 80)

    # Load data
    input_path = Path(input_file)
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} total rows from {input_path}")

    # Sample across time for full dataset
    full_sample = sample_data_across_time(df, max_rows)

    # Skip rows already labeled in Phase 1
    phase2_df = full_sample.iloc[skip_first_n:].copy()
    logger.info(f"Processing rows {skip_first_n} to {len(full_sample)} ({len(phase2_df)} rows)")

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

    # Load task configurations
    config_path = Path(task_configs_file)
    task_configs = load_task_configs(config_path)

    # Load learned rules from Phase 1
    rules_path = Path(learned_rules_file)
    with open(rules_path) as f:
        learned_rules = json.load(f)
    logger.info(f"Loaded learned rules from: {rules_path}")

    # Update task configs with learned rules
    for task, rules in learned_rules.items():
        if task in task_configs:
            task_configs[task]["principles"] = rules

    task_list = list(task_configs.keys())
    logger.info(f"Tasks to label ({len(task_list)}): {', '.join(task_list)}")

    # Initialize settings
    settings = Settings()
    settings.llm_budget = budget
    logger.info(f"Using LLM: {settings.llm_model} (provider: {settings.llm_provider})")
    logger.info(f"Budget limit: ${settings.llm_budget:.2f}")

    # Initialize services (no rule evolution in Phase 2)
    multi_agent = MultiAgentService(settings, task_configs)
    logger.info(f"Using learned rules from Phase 1 for {len(task_list)} tasks")

    # Check for existing checkpoint to resume from
    output_path = Path(output_file)
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{output_path.stem}_checkpoint.csv"

    start_batch = 0
    all_results = []

    # Resume from checkpoint if exists
    if checkpoint_file.exists():
        logger.info(f"Found checkpoint at {checkpoint_file}, resuming...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        all_results.append(checkpoint_df)
        start_batch = len(checkpoint_df) // batch_size
        logger.info(f"Resuming from batch {start_batch + 1} (already processed {len(checkpoint_df)} rows)")

    # Process in batches
    total_batches = (len(phase2_df) + batch_size - 1) // batch_size

    for batch_idx in range(start_batch, total_batches):
        # Check budget before starting batch
        if multi_agent.cost_tracker and multi_agent.cost_tracker.is_budget_exceeded():
            logger.warning("Budget exceeded, stopping processing")
            logger.info(f"Processed {batch_idx * batch_size} / {len(phase2_df)} rows before budget limit")
            break

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(phase2_df))
        batch_df = phase2_df.iloc[start_idx:end_idx].copy()

        logger.info(f"Batch {batch_idx + 1}/{total_batches} (rows {start_idx + skip_first_n}-{end_idx + skip_first_n})")

        try:
            # Label batch
            labeled_batch = multi_agent.label_with_agents(batch_df, text_column, task_list)
            all_results.append(labeled_batch)

            # Save checkpoint after each batch
            checkpoint_progress = pd.concat(all_results, ignore_index=True)
            checkpoint_progress.to_csv(checkpoint_file, index=False)

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {(batch_idx + 1) * batch_size} / {len(phase2_df)} rows")
                logger.info(f"  Checkpoint saved: {len(checkpoint_progress)} rows")

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

    # Save Phase 2 results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"\nPhase 2 complete: {len(final_df)} rows labeled")
    logger.info(f"Results saved to: {output_path}")

    # Clean up checkpoint file on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Checkpoint file removed (processing complete)")

    # Log cost summary
    multi_agent.log_cost_summary()

    # Log confidence statistics
    for task in task_list:
        conf_col = f"confidence_{task}"
        mean_conf = final_df[conf_col].mean()
        logger.info(f"  {task}: mean confidence={mean_conf:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Policy Uncertainty News - Phase 2: Full Labeling")
    parser.add_argument("--input", required=True, help="Input CSV file with news articles")
    parser.add_argument("--output", required=True, help="Output CSV file for labeled data")
    parser.add_argument("--config", default="configs/policy_uncertainty_news_tasks.json",
                       help="Task configuration JSON file")
    parser.add_argument("--rules", required=True, help="Learned rules JSON file from Phase 1")
    parser.add_argument("--text-column", default=None, help="Column containing text (auto-detected if not specified)")
    parser.add_argument("--skip", type=int, default=5000, help="Number of rows to skip (labeled in Phase 1)")
    parser.add_argument("--max-rows", type=int, default=25000, help="Maximum total rows to label")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--budget", type=float, default=50.0, help="LLM budget for Phase 2")

    args = parser.parse_args()

    run_phase2(
        input_file=args.input,
        output_file=args.output,
        task_configs_file=args.config,
        learned_rules_file=args.rules,
        text_column=args.text_column,
        skip_first_n=args.skip,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        budget=args.budget
    )
