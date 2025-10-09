"""Test async batch processing on 10 examples."""
import time
from pathlib import Path
import json
import pandas as pd
from autolabeler.config import Settings
from autolabeler.agents import MultiAgentService

def main():
    settings = Settings()

    # Load task configs
    config_path = Path("configs/fed_headlines_tasks.json")
    with open(config_path) as f:
        task_configs = json.load(f)
    tasks = list(task_configs.keys())

    # Load 10 test rows
    df = pd.read_csv("datasets/fed_data_full.csv", nrows=10)

    # Initialize service
    service = MultiAgentService(settings, task_configs)

    print(f"\n{'='*80}")
    print("ASYNC CONCURRENT TEST (10 rows)")
    print(f"{'='*80}\n")

    start = time.time()
    labeled_df = service.label_with_agents(df, text_column="headline", tasks=tasks, use_async=True)
    elapsed = time.time() - start

    print(f"\n{'='*80}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Average: {elapsed/10:.2f} seconds per row")
    print(f"{'='*80}\n")

    # Show sample results
    for i in range(3):
        row = labeled_df.iloc[i]
        print(f"Row {i+1}: {row['headline'][:80]}...")
        print(f"  Relevancy: {row['label_relevancy']} ({row['confidence_relevancy']:.2f})")
        print(f"  Hawk/Dove: {row['label_hawk_dove']} ({row['confidence_hawk_dove']:.2f})")
        print(f"  Speaker: {row['label_speaker']} ({row['confidence_speaker']:.2f})\n")

if __name__ == "__main__":
    main()
