"""Run Phase 2 only: Label remaining data using learned rules from Phase 1."""
from pathlib import Path
from scripts.fed_headlines_labeling_pipeline import phase2_full_labeling
from autolabeler.config import Settings
import json

if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent
    input_file = project_root / "datasets/fed_data_full.csv"
    output_dir = project_root / "outputs/fed_headlines"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_rules = output_dir / "phase1_learned_rules.json"
    phase2_output = output_dir / "phase2_labeled_19000.csv"
    task_configs_file = project_root / "configs/fed_headlines_tasks.json"

    # Check if Phase 1 rules exist
    if not phase1_rules.exists():
        print(f"ERROR: Phase 1 learned rules not found at {phase1_rules}")
        print("Please run Phase 1 first: python run_phase1.py")
        exit(1)

    # Load learned rules from Phase 1
    with open(phase1_rules, 'r') as f:
        learned_rules = json.load(f)

    print(f"Loaded learned rules from Phase 1: {phase1_rules}")
    print(f"Rules for {len(learned_rules)} tasks")

    # Settings
    settings = Settings()

    # Run Phase 2
    phase2_full_labeling(
        input_file=input_file,
        output_file=phase2_output,
        task_configs_file=task_configs_file,
        learned_rules=learned_rules,
        skip_first_n=1000,
        max_rows=20000,
        batch_size=100,
        settings=settings
    )

    print(f"\nPhase 2 complete!")
    print(f"Labeled data: {phase2_output}")
    print(f"\nTo combine results, run:")
    print(f"  python -c 'import pandas as pd; pd.concat([pd.read_csv(\"outputs/fed_headlines/phase1_labeled_1000.csv\"), pd.read_csv(\"{phase2_output}\")]).to_csv(\"outputs/fed_headlines/fed_headlines_labeled_20000.csv\", index=False)'")
