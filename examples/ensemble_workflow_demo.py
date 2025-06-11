#!/usr/bin/env python3
"""
Ensemble Workflow Demo

Shows both approaches to multi-model prediction:
1. Parallel predictions with individual model outputs
2. Ensemble consolidation with sophisticated voting methods
"""

import json
from pathlib import Path
import pandas as pd


def create_multi_model_config():
    """Create config for parallel multi-model predictions."""
    config = {
        "data": {
            "input_file": "datasets/stacked_headline_scores_simple.csv",
            "output_file": "results/multi_model_predictions.csv",
            "text_column": "headline",
            "label_column": "score",
            "metadata_columns": ["flagged_people_keys"],
            "test_size": 0.2
        },
        "model": {
            "dataset_name": "multi_model_demo",
            "use_rag": True,
            "confidence_threshold": 0.6,
            "seed": 42,
            "evaluate": True,
            "output_dir": "results/multi_model_evaluation"
        },
        "llm_models": [
            {
                "model_id": "conservative",
                "model_name": "openai/gpt-4o-mini",
                "temperature": 0.1,
                "description": "Conservative model for consistent results"
            },
            {
                "model_id": "balanced",
                "model_name": "openai/gpt-4o-mini",
                "temperature": 0.3,
                "description": "Balanced model for good performance"
            },
            {
                "model_id": "creative",
                "model_name": "anthropic/claude-3-haiku",
                "temperature": 0.4,
                "description": "More creative model for diverse perspectives"
            }
        ]
    }

    # Save config
    Path("configs").mkdir(exist_ok=True)
    with open("configs/multi_model_demo.json", "w") as f:
        json.dump(config, f, indent=2)

    return config


def demo_workflow():
    """Demonstrate the complete multi-model workflow."""
    print("🤖 Multi-Model AutoLabeler Demo")
    print()

    # Create config
    config = create_multi_model_config()
    print("📝 Created multi-model configuration with 3 LLMs:")
    for model in config["llm_models"]:
        print(f"  • {model['model_id']}: {model['model_name']} (temp={model['temperature']})")
    print()

    # Method 1: Parallel predictions
    print("🔀 Method 1: Parallel Predictions")
    print("Command:")
    print("  python -m autolabeler.cli label configs/multi_model_demo.json")
    print()
    print("This will create ONE output file with columns for each model:")
    print("  📄 results/multi_model_predictions.csv")
    print("  📊 Columns: score_conservative, score_balanced, score_creative")
    print("  📈 Individual confidence scores for each model")
    print("  🔍 Separate evaluation results for each model")
    print()

    # Method 2: Ensemble consolidation
    print("🎯 Method 2: Ensemble Consolidation")
    print("Commands:")
    print("  # Majority vote ensemble")
    print("  python -m autolabeler.cli ensemble \\")
    print("    configs/multi_model_demo.json \\")
    print("    datasets/stacked_headline_scores_simple.csv \\")
    print("    results/ensemble_majority.csv \\")
    print("    --text-column headline \\")
    print("    --dataset-name ensemble_demo \\")
    print("    --ensemble-method majority_vote")
    print()
    print("  # Confidence-weighted ensemble")
    print("  python -m autolabeler.cli ensemble \\")
    print("    configs/multi_model_demo.json \\")
    print("    datasets/stacked_headline_scores_simple.csv \\")
    print("    results/ensemble_weighted.csv \\")
    print("    --text-column headline \\")
    print("    --dataset-name ensemble_demo \\")
    print("    --ensemble-method confidence_weighted")
    print()

    # Output comparison
    print("📊 Output Comparison:")
    print()
    print("Parallel Predictions Output:")
    print("""
headline,score_conservative,score_conservative_confidence,score_balanced,score_balanced_confidence,score_creative,score_creative_confidence
"FED'S POWELL SAYS RATES APPROPRIATE",1,0.85,0,0.90,1,0.75
"YELLEN CRITICIZES FED POLICY",-2,0.88,-1,0.82,-2,0.91
    """)

    print("Ensemble Output:")
    print("""
headline,ensemble_label,ensemble_confidence,agreement_score,individual_predictions
"FED'S POWELL SAYS RATES APPROPRIATE",1,0.83,0.67,"[1,0,1]"
"YELLEN CRITICIZES FED POLICY",-2,0.87,0.67,"[-2,-1,-2]"
    """)

    # Best practices
    print("💡 Best Practices:")
    print()
    print("🔬 For Research & Analysis:")
    print("  • Use parallel predictions to study model disagreements")
    print("  • Analyze which models perform best on different types of content")
    print("  • Compare confidence scores across models")
    print()
    print("🚀 For Production Deployment:")
    print("  • Use ensemble methods for more robust predictions")
    print("  • Choose ensemble method based on your accuracy/speed needs:")
    print("    - majority_vote: Fast, good for discrete labels")
    print("    - confidence_weighted: Better accuracy, considers uncertainty")
    print("    - high_agreement: Conservative, only outputs when models agree")
    print()

    # Performance considerations
    print("⚡ Performance Notes:")
    print("  • Multiple models = proportionally longer runtime")
    print("  • 3 models ≈ 3x the API costs")
    print("  • Use workers=1 to get all enhanced features")
    print("  • Ensemble post-processing is very fast")
    print()

    print("🎯 Summary:")
    print("  • Parallel: Multiple columns, detailed analysis")
    print("  • Ensemble: Single column, robust prediction")
    print("  • Both approaches store all results in output files")
    print("  • Choose based on whether you need individual model insights")


if __name__ == "__main__":
    demo_workflow()
