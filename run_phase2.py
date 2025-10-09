"""Run Phase 2 only: Label remaining data using learned rules from Phase 1."""
from pathlib import Path
from scripts.fed_headlines_labeling_pipeline import phase2_full_labeling
from autolabeler.config import Settings
import json
import pandas as pd
import glob

if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent
    input_file = project_root / "datasets/fed_data_full.csv"
    output_dir = project_root / "outputs/fed_headlines"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_rules = output_dir / "phase1_learned_rules.json"
    task_configs_file = project_root / "configs/fed_headlines_tasks.json"

    # Check if Phase 1 rules exist
    if not phase1_rules.exists():
        print(f"ERROR: Phase 1 learned rules not found at {phase1_rules}")
        print("Please run Phase 1 first: python run_phase1.py")
        exit(1)

    # Auto-detect Phase 1 output file
    phase1_files = list(output_dir.glob("phase1_labeled_*.csv"))
    if not phase1_files:
        print(f"ERROR: No Phase 1 output files found in {output_dir}")
        print("Expected file matching pattern: phase1_labeled_*.csv")
        exit(1)

    phase1_output_file = phase1_files[0]
    print(f"Found Phase 1 output: {phase1_output_file}")

    # Read Phase 1 output to determine how many rows were labeled
    phase1_df = pd.read_csv(phase1_output_file)
    phase1_count = len(phase1_df)
    print(f"Phase 1 labeled {phase1_count} rows")

    # Load learned rules from Phase 1
    with open(phase1_rules, 'r') as f:
        learned_rules = json.load(f)

    print(f"Loaded learned rules from Phase 1: {phase1_rules}")
    print(f"Rules for {len(learned_rules)} tasks")

    # Settings
    settings = Settings()

    # Determine total sample size and Phase 2 row count
    # Default to 4x Phase 1 size if not specified
    max_rows = phase1_count * 4  # e.g., 5000 -> 20000
    phase2_count = max_rows - phase1_count

    print(f"\nPhase 2 Configuration:")
    print(f"  Total sample size: {max_rows}")
    print(f"  Skip first: {phase1_count} (Phase 1)")
    print(f"  Label next: {phase2_count} rows")

    # Generate output filename based on count
    phase2_output = output_dir / f"phase2_labeled_{phase2_count}.csv"

    # Run Phase 2
    phase2_full_labeling(
        input_file=input_file,
        output_file=phase2_output,
        task_configs_file=task_configs_file,
        learned_rules=learned_rules,
        skip_first_n=phase1_count,  # Automatically matches Phase 1
        max_rows=max_rows,           # 4x Phase 1 by default
        batch_size=100,
        settings=settings
    )

    print(f"\nPhase 2 complete!")
    print(f"Labeled data: {phase2_output}")
    print(f"Rows labeled: {phase1_count}-{max_rows} ({phase2_count} rows)")

    # Generate combined output filename
    combined_output = output_dir / f"fed_headlines_labeled_{max_rows}.csv"
    print(f"\nTo combine results:")
    print(f"  python -c 'import pandas as pd; pd.concat([pd.read_csv(\"{phase1_output_file}\"), pd.read_csv(\"{phase2_output}\")]).to_csv(\"{combined_output}\", index=False)'")
