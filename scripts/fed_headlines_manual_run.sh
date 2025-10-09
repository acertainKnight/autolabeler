#!/bin/bash
# Manual execution script for Federal Reserve headlines labeling
#
# This script provides manual control over the two-phase labeling process.
# You can run phases individually or the complete pipeline.

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Fed Headlines Labeling Pipeline"
echo "=========================================="
echo ""

# Check if configs exist
if [ ! -f "configs/fed_headlines_tasks.json" ]; then
    echo "ERROR: Task config file not found: configs/fed_headlines_tasks.json"
    exit 1
fi

if [ ! -f "datasets/fed_data_full.csv" ]; then
    echo "ERROR: Input data file not found: datasets/fed_data_full.csv"
    exit 1
fi

# Create output directory
mkdir -p outputs/fed_headlines

echo "Configuration:"
echo "  Input: datasets/fed_data_full.csv"
echo "  Tasks: configs/fed_headlines_tasks.json"
echo "  Output: outputs/fed_headlines/"
echo ""

# Check command line argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 [phase1|phase2|complete]"
    echo ""
    echo "Commands:"
    echo "  phase1    - Run Phase 1 only (1000 examples with rule learning)"
    echo "  phase2    - Run Phase 2 only (19000 examples with learned rules)"
    echo "  complete  - Run complete pipeline (both phases)"
    echo ""
    exit 1
fi

case "$1" in
    phase1)
        echo "Running Phase 1: Rule Learning (1000 examples)"
        echo "=============================================="
        python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from scripts.fed_headlines_labeling_pipeline import phase1_learn_rules
from autolabeler.config import Settings

learned_rules = phase1_learn_rules(
    input_file=Path('datasets/fed_data_full.csv'),
    output_file=Path('outputs/fed_headlines/phase1_labeled_1000.csv'),
    task_configs_file=Path('configs/fed_headlines_tasks.json'),
    n_samples=1000,
    batch_size=50,
    confidence_threshold=0.7,
    settings=Settings()
)

import json
with open('outputs/fed_headlines/phase1_learned_rules.json', 'w') as f:
    json.dump(learned_rules, f, indent=2)
print('\nPhase 1 complete!')
print('Learned rules saved to: outputs/fed_headlines/phase1_learned_rules.json')
"
        ;;

    phase2)
        echo "Running Phase 2: Full Labeling (19000 examples)"
        echo "================================================"

        if [ ! -f "outputs/fed_headlines/phase1_learned_rules.json" ]; then
            echo "ERROR: Phase 1 rules not found. Please run phase1 first."
            exit 1
        fi

        python3 -c "
from pathlib import Path
import sys
import json
sys.path.insert(0, 'src')
from scripts.fed_headlines_labeling_pipeline import phase2_full_labeling
from autolabeler.config import Settings

# Load learned rules from Phase 1
with open('outputs/fed_headlines/phase1_learned_rules.json') as f:
    learned_rules = json.load(f)

phase2_full_labeling(
    input_file=Path('datasets/fed_data_full.csv'),
    output_file=Path('outputs/fed_headlines/phase2_labeled_19000.csv'),
    task_configs_file=Path('configs/fed_headlines_tasks.json'),
    learned_rules=learned_rules,
    skip_first_n=1000,
    max_rows=20000,
    batch_size=100,
    settings=Settings()
)
print('\nPhase 2 complete!')
"
        ;;

    complete)
        echo "Running Complete Pipeline"
        echo "========================="
        python3 scripts/fed_headlines_labeling_pipeline.py
        ;;

    *)
        echo "ERROR: Unknown command: $1"
        echo "Use: phase1, phase2, or complete"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
