"""
Policy Uncertainty Corporate Earnings Transcripts - Phase 1: Rule Learning

This script implements Phase 1 for earnings transcripts. Adapted for longer-form text
with more formal corporate language compared to news articles.

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
    """Sample data temporally to get representative sample of transcripts."""
    # Auto-detect date column
    if date_column is None:
        date_candidates = ['date', 'quarter_date', 'filing_date', 'earnings_date', 'timestamp']
        for col in date_candidates:
            if col in df.columns:
                date_column = col
                break

    if date_column is None or date_column not in df.columns:
        logger.warning(f"No date column found, using random sampling")
        return df.sample(n=min(n_samples, len(df)), random_state=42)

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], format='mixed', errors='coerce')
    df['year_quarter'] = df[date_column].dt.to_period('Q')

    unique_quarters = df['year_quarter'].unique()
    samples_per_quarter = n_samples // len(unique_quarters)
    remaining_samples = n_samples % len(unique_quarters)

    sampled_dfs = []
    for i, quarter in enumerate(unique_quarters):
        quarter_data = df[df['year_quarter'] == quarter]
        quarter_samples = samples_per_quarter + (1 if i < remaining_samples else 0)
        quarter_samples = min(quarter_samples, len(quarter_data))

        if quarter_samples > 0:
            sampled_quarter = quarter_data.sample(n=quarter_samples, random_state=42)
            sampled_dfs.append(sampled_quarter)

    sampled_df = pd.concat(sampled_dfs, ignore_index=True).drop('year_quarter', axis=1)
    sampled_df = sampled_df.sort_values(date_column)

    logger.info(f"Sampled {len(sampled_df)} transcripts from {len(unique_quarters)} quarters")
    logger.info(f"Date range: {sampled_df[date_column].min()} to {sampled_df[date_column].max()}")
    return sampled_df


def run_phase1(
    input_file: str,
    output_file: str,
    task_configs_file: str,
    text_column: str = None,
    n_samples: int = 5000,
    batch_size: int = 30,
    confidence_threshold: float = 0.7,
    budget: float = 100.0,
    max_text_length: int = 8000
):
    """Phase 1: Label transcripts with rule evolution.

    Note: Smaller batch size (30) due to longer text length of transcripts.

    Args:
        max_text_length: Maximum characters per transcript (truncate longer ones)
    """
    logger.info("=" * 80)
    logger.info(f"PHASE 1: Rule Learning - Earnings Transcripts ({n_samples} examples)")
    logger.info("=" * 80)

    # Load data
    input_path = Path(input_file)
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} total rows from {input_path}")

    # Auto-detect text column
    if text_column is None:
        text_candidates = ['text', 'transcript', 'content', 'full_text', 'body', 'qa_section']
        for col in text_candidates:
            if col in df.columns:
                text_column = col
                break
        if text_column is None:
            raise ValueError(f"Could not auto-detect text column. Available: {list(df.columns)}")

    logger.info(f"Using text column: {text_column}")

    # Sample across time
    sample_df = sample_data_across_time(df, n_samples)

    # Truncate long transcripts
    sample_df[text_column] = sample_df[text_column].apply(
        lambda x: x[:max_text_length] if isinstance(x, str) and len(x) > max_text_length else x
    )
    logger.info(f"Truncated transcripts to max {max_text_length} characters")

    # Load task configurations
    config_path = Path(task_configs_file)
    task_configs = load_task_configs(config_path)
    task_list = list(task_configs.keys())
    logger.info(f"Tasks to label ({len(task_list)}): {', '.join(task_list)}")

    # Initialize settings
    settings = Settings()
    settings.llm_budget = budget
    logger.info(f"Using LLM: {settings.llm_model}")
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

            # Rule evolution
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

                    current_rules = constitutional.get_principles()
                    updated_rules = rule_evolution.improve_rules(current_rules, feedback_df)

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

    # Save results
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

    multi_agent.log_cost_summary()

    # Save learned rules
    learned_rules = constitutional.get_principles()
    rules_file = output_path.parent / f"{output_path.stem}_learned_rules.json"
    with open(rules_file, 'w') as f:
        json.dump(learned_rules, f, indent=2)
    logger.info(f"Learned rules saved to: {rules_file}")

    # Log statistics
    stats = rule_evolution.get_improvement_stats()
    logger.info("\nRule Evolution Statistics:")
    logger.info(f"  Total patterns identified: {stats['total_patterns_identified']}")
    logger.info(f"  Rules generated: {stats['rules_generated']}")

    for task in task_list:
        conf_col = f"confidence_{task}"
        mean_conf = final_df[conf_col].mean()
        low_conf = (final_df[conf_col] < confidence_threshold).sum()
        logger.info(f"  {task}: mean confidence={mean_conf:.3f}, uncertain={low_conf}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Policy Uncertainty Transcripts - Phase 1")
    parser.add_argument("--input", required=True, help="Input CSV file with transcripts")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--config", default="configs/policy_uncertainty_transcripts_tasks.json")
    parser.add_argument("--text-column", default=None, help="Text column name")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--max-length", type=int, default=8000)

    args = parser.parse_args()

    run_phase1(
        input_file=args.input,
        output_file=args.output,
        task_configs_file=args.config,
        text_column=args.text_column,
        n_samples=args.samples,
        batch_size=args.batch_size,
        budget=args.budget,
        max_text_length=args.max_length
    )
