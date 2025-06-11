#!/usr/bin/env python3
"""
Simple CLI Usage Example

Shows how clean and simple the CLI is now - just pass a config file!
All API keys are securely stored in .env file.
All task parameters are in the config file.
"""

import subprocess
import sys
from pathlib import Path


def run_enhanced_labeling():
    """Demonstrate the simplified CLI usage."""

    print("🚀 Enhanced AutoLabeler - Simple CLI Demo")
    print()

    # Example 1: Basic labeling with train/test split and evaluation
    print("📋 Example 1: Complete task with automatic train/test split")
    print("Command:")
    print("  python -m autolabeler.cli label configs/speaker_independence_enhanced.json")
    print()
    print("This single command will:")
    print("  ✅ Load all configuration from the JSON file")
    print("  ✅ Load API keys securely from .env file")
    print("  ✅ Create automatic train/test split (70/30)")
    print("  ✅ Use metadata features (speaker, source)")
    print("  ✅ Apply dynamic rule set for scoring")
    print("  ✅ Prevent data leakage with noise injection")
    print("  ✅ Evaluate performance on test set")
    print("  ✅ Save detailed evaluation results")
    print()

    # Example 2: Multi-model comparison
    print("📊 Example 2: Multi-model ensemble comparison")
    print("Command:")
    print("  python -m autolabeler.cli label configs/enhanced_task_example.json")
    print()
    print("This will run 3 different models and compare results:")
    print("  🤖 GPT-4O Mini (conservative)")
    print("  🤖 GPT-4O Mini (balanced)")
    print("  🤖 Claude Haiku (fast)")
    print()

    # Example 3: Custom configuration
    print("⚙️ Example 3: Custom configuration structure")
    print()
    print("Your config file contains everything:")
    print("""
{
  "data": {
    "input_file": "your_data.csv",
    "output_file": "results.csv",
    "text_column": "text",
    "metadata_columns": ["category", "source"],
    "test_size": 0.2
  },
  "model": {
    "dataset_name": "my_task",
    "ruleset_file": "rules.json",
    "evaluate": true,
    "seed": 42
  },
  "llm_models": [
    {
      "model_id": "main_model",
      "model_name": "openai/gpt-4o-mini",
      "temperature": 0.1
    }
  ]
}
    """)
    print()

    # Environment setup
    print("🔐 Environment Setup (.env file):")
    print("""
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here  # for embeddings
CORPORATE_API_KEY=your_corporate_key  # optional
    """)
    print()

    print("🎯 Key Benefits:")
    print("  🔒 Secure: API keys in .env, never in configs")
    print("  📋 Simple: Single command with config file")
    print("  🔬 Research-ready: Automatic evaluation & reproducibility")
    print("  🛡️ Safe: Built-in data leakage prevention")
    print("  📊 Comprehensive: Metadata features & rule sets")
    print("  🔄 Reproducible: Fixed seeds for consistent results")


def try_run_example():
    """Try to run an actual example if the config exists."""
    config_file = Path("configs/speaker_independence_enhanced.json")

    if config_file.exists():
        print("🏃 Trying to run actual example...")
        print(f"Running: python -m autolabeler.cli label {config_file}")

        try:
            result = subprocess.run([
                sys.executable, "-m", "autolabeler.cli",
                "label", str(config_file)
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("✅ Command executed successfully!")
                print("📤 Output:")
                print(result.stdout[-500:])  # Last 500 chars
            else:
                print("❌ Command failed (this is expected if .env not set up)")
                print("🔍 Error:")
                print(result.stderr[-500:])  # Last 500 chars

        except subprocess.TimeoutExpired:
            print("⏰ Command timed out (normal for demo)")
        except Exception as e:
            print(f"💥 Error running command: {e}")
    else:
        print(f"⚠️ Config file not found: {config_file}")
        print("Create it first using the example above!")


if __name__ == "__main__":
    run_enhanced_labeling()
    print("\n" + "="*60 + "\n")
    try_run_example()
