#!/usr/bin/env python3
"""
Example script demonstrating AutoLabeler CLI usage.

This script shows how to:
1. Create configuration files
2. Label data with multiple models
3. Generate synthetic examples
4. Run ensemble labeling
5. Balance datasets

Make sure to install the package first:
pip install -e .

And set up your API keys in the configuration file.
"""

import pandas as pd
from pathlib import Path
import subprocess
import sys


def create_sample_data():
    """Create sample data for demonstration."""
    data = [
        {"text": "I love this product! It's amazing!", "true_label": "positive"},
        {"text": "This is terrible. Worst purchase ever.", "true_label": "negative"},
        {"text": "It's okay, nothing special.", "true_label": "neutral"},
        {"text": "Absolutely fantastic! Highly recommend!", "true_label": "positive"},
        {"text": "Waste of money. Very disappointed.", "true_label": "negative"},
        {"text": "Pretty good quality for the price.", "true_label": "positive"},
        {"text": "Meh, it's average.", "true_label": "neutral"},
        {"text": "Outstanding service and product!", "true_label": "positive"},
        {"text": "Not worth it at all.", "true_label": "negative"},
        {"text": "Decent but could be better.", "true_label": "neutral"},
    ]

    df = pd.DataFrame(data)
    df.to_csv("sample_data.csv", index=False)
    print("Created sample_data.csv with 10 labeled examples")
    return df


def run_cli_command(cmd):
    """Run a CLI command and handle errors."""
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Run the CLI demonstration."""
    print("🎯 AutoLabeler CLI Demonstration")
    print("=" * 60)

    # Step 1: Create sample configuration
    print("\n📝 Step 1: Creating sample configuration...")
    if not Path("config_example.json").exists():
        run_cli_command(["autolabeler", "create-config"])
        print("✅ Please edit config_example.json with your API keys before continuing!")
        print("   Set your OpenRouter API key in the 'openrouter_api_key' field.")
        return

    # Step 2: Create sample data
    print("\n📊 Step 2: Creating sample data...")
    create_sample_data()

    # Step 3: Show dataset statistics
    print("\n📈 Step 3: Checking dataset statistics...")
    run_cli_command([
        "autolabeler", "stats", "demo_dataset",
        "--config-file", "config_example.json"
    ])

    # Step 4: Label data with multiple models
    print("\n🏷️  Step 4: Labeling data with multiple models...")
    success = run_cli_command([
        "autolabeler", "label",
        "config_example.json",
        "sample_data.csv",
        "labeled_results.csv",
        "--text-column", "text",
        "--dataset-name", "demo_dataset",
        "--label-column", "sentiment",
        "--use-rag"
    ])

    if success:
        print("✅ Individual model labeling complete! Check labeled_results.csv")

    # Step 5: Run ensemble labeling
    print("\n🎭 Step 5: Running ensemble labeling...")
    success = run_cli_command([
        "autolabeler", "ensemble",
        "config_example.json",
        "sample_data.csv",
        "ensemble_results.csv",
        "--text-column", "text",
        "--dataset-name", "demo_dataset",
        "--ensemble-method", "majority_vote",
        "--save-individual"
    ])

    if success:
        print("✅ Ensemble labeling complete! Check ensemble_results.csv")

    # Step 6: Generate synthetic examples
    print("\n🎨 Step 6: Generating synthetic examples...")
    run_cli_command([
        "autolabeler", "generate",
        "config_example.json",
        "--dataset-name", "demo_dataset",
        "--target-label", "positive",
        "--num-examples", "3",
        "--strategy", "mixed",
        "--output-file", "synthetic_positive.csv",
        "--add-to-kb",
        "--confidence-threshold", "0.7"
    ])

    # Step 7: Balance the dataset
    print("\n⚖️  Step 7: Balancing dataset with synthetic examples...")
    run_cli_command([
        "autolabeler", "balance",
        "config_example.json",
        "--dataset-name", "demo_dataset",
        "--target-balance", "equal",
        "--max-per-label", "5",
        "--confidence-threshold", "0.7",
        "--output-file", "balanced_synthetic.csv"
    ])

    # Step 8: Final statistics
    print("\n📊 Step 8: Final dataset statistics...")
    run_cli_command([
        "autolabeler", "stats", "demo_dataset",
        "--config-file", "config_example.json"
    ])

    print("\n🎉 CLI demonstration complete!")
    print("\nGenerated files:")
    print("  📄 sample_data.csv - Original sample data")
    print("  📄 labeled_results.csv - Individual model predictions")
    print("  📄 ensemble_results.csv - Ensemble predictions")
    print("  📄 synthetic_positive.csv - Generated synthetic examples")
    print("  📄 balanced_synthetic.csv - Balanced synthetic examples")
    print("\nKnowledge base created in: knowledge_bases/demo_dataset/")


if __name__ == "__main__":
    # Check if autolabeler CLI is available
    try:
        subprocess.run(["autolabeler", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ AutoLabeler CLI not found!")
        print("Please install the package first:")
        print("  pip install -e .")
        sys.exit(1)

    main()
