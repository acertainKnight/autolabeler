"""Test script to run Fed headlines pipeline on 100 examples."""

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autolabeler.agents import MultiAgentService
from autolabeler.config import Settings

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def main():
    """Run test on 100 Fed headlines."""

    # Check for OpenRouter API key
    settings = Settings()
    if not settings.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not set in environment")
        logger.info("Set it with: export OPENROUTER_API_KEY=your_key")
        return 1

    logger.info(f"Using model: {settings.llm_model}")
    logger.info(f"Provider: {settings.llm_provider}")

    # Load task configs
    config_path = Path("configs/fed_headlines_tasks.json")
    with open(config_path) as f:
        task_configs = json.load(f)

    tasks = list(task_configs.keys())
    logger.info(f"Tasks to label: {', '.join(tasks)}")

    # Load data (first 100 rows)
    data_path = Path("datasets/fed_data_full.csv")
    logger.info(f"Loading first 100 rows from {data_path}")
    df = pd.read_csv(data_path, nrows=100)
    logger.info(f"Loaded {len(df)} headlines")

    # Initialize service
    logger.info("Initializing MultiAgentService with OpenRouter...")
    service = MultiAgentService(settings, task_configs)

    # Test on first 3 examples
    logger.info("\n" + "="*80)
    logger.info("TESTING ON 3 EXAMPLES")
    logger.info("="*80)

    for idx in range(3):
        headline = df.iloc[idx]["headline"]
        logger.info(f"\nExample {idx + 1}:")
        logger.info(f"Headline: {headline}")

        result = service.label_single(headline, tasks)

        logger.info(f"\nResults:")
        logger.info(f"  Relevancy: {result.labels['relevancy']} (confidence: {result.confidences['relevancy']:.2f})")
        logger.info(f"  Hawk/Dove: {result.labels['hawk_dove']} (confidence: {result.confidences['hawk_dove']:.2f})")
        logger.info(f"  Speaker: {result.labels['speaker']} (confidence: {result.confidences['speaker']:.2f})")
        logger.info(f"  Reasoning (relevancy): {result.reasoning['relevancy'][:100]}...")

    # Label all 100
    logger.info("\n" + "="*80)
    logger.info("LABELING ALL 100 EXAMPLES")
    logger.info("="*80)

    labeled_df = service.label_with_agents(df, text_column="headline", tasks=tasks)

    # Save results
    output_path = Path("outputs/test_100_labeled.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved results to: {output_path}")

    # Show summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    logger.info(f"\nRelevancy distribution:")
    logger.info(labeled_df['label_relevancy'].value_counts().to_string())

    logger.info(f"\nHawk/Dove distribution:")
    logger.info(labeled_df['label_hawk_dove'].value_counts().to_string())

    logger.info(f"\nTop 5 speakers:")
    logger.info(labeled_df['label_speaker'].value_counts().head(5).to_string())

    logger.info(f"\nMean confidences:")
    for task in ['relevancy', 'hawk_dove', 'speaker']:
        mean_conf = labeled_df[f'confidence_{task}'].mean()
        logger.info(f"  {task}: {mean_conf:.3f}")

    logger.info(f"\n✓ Test complete! Check {output_path} for full results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
