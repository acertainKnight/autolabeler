"""Run Phase 1 only: Learn rules from 1000 sampled examples."""
from pathlib import Path
from scripts.fed_headlines_labeling_pipeline import phase1_learn_rules
from autolabeler.config import Settings
import json

if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent
    input_file = project_root / "datasets/fed_data_full.csv"
    output_dir = project_root / "outputs/fed_headlines"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_output = output_dir / "phase1_labeled_1000.csv"
    phase1_rules = output_dir / "phase1_learned_rules.json"
    task_configs_file = project_root / "configs/fed_headlines_tasks.json"

    # Settings
    settings = Settings()

    # Run Phase 1
    learned_rules = phase1_learn_rules(
        input_file=input_file,
        output_file=phase1_output,
        task_configs_file=task_configs_file,
        n_samples=1000,
        batch_size=50,
        confidence_threshold=0.7,
        settings=settings
    )

    # Save learned rules
    with open(phase1_rules, 'w') as f:
        json.dump(learned_rules, f, indent=2)

    print(f"\nPhase 1 complete!")
    print(f"Labeled data: {phase1_output}")
    print(f"Learned rules: {phase1_rules}")
