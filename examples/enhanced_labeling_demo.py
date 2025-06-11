#!/usr/bin/env python3
"""
Enhanced AutoLabeler Demo

This script demonstrates all the new enhanced features of the AutoLabeler:
- Train/test splits with data leakage prevention
- Dynamic rule set loading
- Metadata features for improved context
- Noise injection for example selection variability
- Test set evaluation and performance metrics
- Reproducible seeding

Usage:
    python examples/enhanced_labeling_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from autolabeler.config import Settings
from autolabeler.labeler import AutoLabeler


def create_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create demo datasets for testing enhanced features."""
    # Main dataset with labels for train/test split
    main_data = pd.DataFrame({
        "text": [
            "FED'S POWELL SAYS INFLATION REMAINS TOO HIGH",
            "FED'S YELLEN CRITICIZES CURRENT MONETARY POLICY",
            "BERNANKE PRAISES CENTRAL BANK INDEPENDENCE",
            "FED'S WILLIAMS DEFENDS RECENT RATE DECISIONS",
            "DRAGHI QUESTIONS FEDERAL RESERVE STRATEGY",
            "FED'S CLARIDA SUPPORTS ONGOING MEASURES",
            "GREENSPAN WARNS AGAINST POLITICAL INTERFERENCE",
            "FED'S BRAINARD ENDORSES CURRENT APPROACH",
            "VOLCKER CRITICIZES RECENT FED DECISIONS",
            "FED'S BOSTIC EXPLAINS POLICY RATIONALE"
        ],
        "speaker": ["powell", "yellen", "bernanke", "williams", "draghi",
                   "clarida", "greenspan", "brainard", "volcker", "bostic"],
        "independence_score": [1, -2, 2, 1, -1, 2, 3, 1, -2, 1],
        "source": ["reuters", "bloomberg", "wsj", "reuters", "ft",
                  "bloomberg", "wsj", "reuters", "ft", "bloomberg"],
        "urgency": ["high", "medium", "low", "medium", "high",
                   "low", "high", "medium", "high", "medium"],
        "category": ["monetary_policy", "criticism", "independence", "defense", "critique",
                    "support", "warning", "endorsement", "criticism", "explanation"]
    })

    # Training data (separate file)
    train_data = pd.DataFrame({
        "text": [
            "FED'S JEROME SAYS POLICY IS APPROPRIATE",
            "CENTRAL BANK CRITICIZED FOR OVERREACH",
            "INDEPENDENCE IS CRUCIAL FOR MONETARY POLICY"
        ],
        "speaker": ["jerome", "unknown", "unknown"],
        "independence_score": [1, -2, 3],
        "source": ["reuters", "wsj", "bloomberg"],
        "urgency": ["medium", "high", "low"]
    })

    # Data to exclude from training examples
    exclude_data = pd.DataFrame({
        "text": [
            "THIS TEXT SHOULD NOT BE USED AS EXAMPLE",
            "EXCLUDE THIS FROM TRAINING"
        ],
        "speaker": ["unknown", "unknown"],
        "source": ["internal", "internal"]
    })

    return main_data, train_data, exclude_data


def create_demo_ruleset() -> dict:
    """Create a demo rule set for independence scoring."""
    return {
        "task": "central_bank_independence_scoring",
        "description": "Score statements on central bank independence stance",
        "scale": {
            "range": [-3, 3],
            "labels": {
                "-3": "Strong criticism/attack on independence",
                "-2": "Moderate criticism",
                "-1": "Mild criticism or concern",
                "0": "Neutral or unrelated to independence",
                "1": "Mild support or positive mention",
                "2": "Moderate support or defense",
                "3": "Strong support or praise"
            }
        },
        "indicators": {
            "criticism": ["criticizes", "attacks", "questions", "warns against"],
            "support": ["supports", "defends", "praises", "endorses"],
            "independence_terms": ["independence", "autonomy", "political interference"]
        },
        "examples": [
            {
                "text": "Fed independence is crucial for effective monetary policy",
                "score": 3,
                "reasoning": "Strong support for independence principle"
            },
            {
                "text": "Politicians should not interfere with Fed decisions",
                "score": 2,
                "reasoning": "Defense against political interference"
            }
        ]
    }


def demo_enhanced_features():
    """Demonstrate all enhanced AutoLabeler features."""
    print("🚀 Enhanced AutoLabeler Demo\n")

    # Create demo data
    print("📊 Creating demo datasets...")
    main_df, train_df, exclude_df = create_demo_data()

    # Save datasets
    Path("demo_data").mkdir(exist_ok=True)
    main_df.to_csv("demo_data/main_data.csv", index=False)
    train_df.to_csv("demo_data/train_data.csv", index=False)
    exclude_df.to_csv("demo_data/exclude_data.csv", index=False)

    # Create and save rule set
    ruleset = create_demo_ruleset()
    with open("demo_data/independence_rules.json", "w") as f:
        json.dump(ruleset, f, indent=2)

    # Create settings (you'll need to add your API keys)
    settings = Settings(
        openrouter_api_key="your-openrouter-key",  # Add your key here
        llm_model="openai/gpt-4o-mini",
        max_examples_per_query=3
    )

    print("✅ Demo data created\n")

    # Demo 1: Basic enhanced labeling with metadata
    print("🔬 Demo 1: Enhanced labeling with metadata features")
    labeler = AutoLabeler(
        dataset_name="independence_demo",
        settings=settings,
        ruleset_file="demo_data/independence_rules.json",
        seed=42,
        noise_factor=0.15
    )

    # Use metadata columns for enhanced context
    metadata_cols = ["source", "urgency", "category"]

    # Add some training data
    labeler.add_training_data(train_df, "text", "independence_score")

    # Label with metadata features
    labeled_df = labeler.label_dataframe_enhanced(
        main_df.head(3),  # Just first 3 for demo
        text_column="text",
        metadata_columns=metadata_cols,
        confidence_threshold=0.7
    )

    print(f"Labeled {len(labeled_df)} rows with metadata features")
    print(f"Metadata columns used: {metadata_cols}\n")

    # Demo 2: Automatic train/test split with evaluation
    print("🎯 Demo 2: Automatic train/test split with evaluation")

    test_labeler = AutoLabeler(
        dataset_name="split_demo",
        settings=settings,
        ruleset_file="demo_data/independence_rules.json",
        seed=42
    )

    # This will create train/test split, label test set, and evaluate
    test_results_df, evaluation = test_labeler.label_with_train_test_split(
        main_df,
        text_column="text",
        label_column="independence_score",
        test_size=0.3,
        stratify_column="independence_score",
        metadata_columns=["source", "urgency"],
        confidence_threshold=0.6
    )

    print(f"Test set size: {len(test_results_df)}")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.3f}")
    print(f"F1 Score: {evaluation['metrics']['f1_weighted']:.3f}\n")

    # Demo 3: Explicit train/test with exclude data
    print("🛡️ Demo 3: Data leakage prevention with exclude data")

    leakage_labeler = AutoLabeler(
        dataset_name="leakage_demo",
        settings=settings,
        seed=42,
        noise_factor=0.2
    )

    # Create train/test split with excluded data
    train_split, test_split = leakage_labeler.create_train_test_split(
        main_df,
        test_size=0.4,
        text_column="text",
        exclude_from_training=[exclude_df]
    )

    # Add training data (exclude data won't be used as examples)
    leakage_labeler.add_training_data(train_split, "text", "independence_score")

    # Label with exclusions
    protected_results = leakage_labeler.label_with_exclude_data(
        test_split.head(2),
        text_column="text",
        exclude_data=[exclude_df, test_split],  # Exclude both datasets
        metadata_columns=["source"]
    )

    print(f"Labeled {len(protected_results)} rows with leakage prevention")
    print(f"Excluded {len(exclude_df)} + {len(test_split)} rows from examples\n")

    # Demo 4: Test set evaluation
    print("📈 Demo 4: Detailed test set evaluation")

    # Add predictions to test data for evaluation demo
    test_data_with_preds = test_split.copy()
    test_data_with_preds["predicted_score"] = [1, -1, 2]  # Mock predictions
    test_data_with_preds["predicted_confidence"] = [0.85, 0.92, 0.78]

    eval_results = leakage_labeler.evaluate_on_test_set(
        test_data_with_preds.head(3),
        true_label_column="independence_score",
        pred_label_column="predicted_score",
        confidence_column="predicted_confidence",
        save_results=True,
        output_dir=Path("demo_results")
    )

    print(f"Evaluation completed - results saved to demo_results/")
    print(f"Accuracy: {eval_results['metrics']['accuracy']:.3f}")
    print(f"Cohen's Kappa: {eval_results['metrics']['cohen_kappa']:.3f}\n")

    # Demo 5: Reproducibility with seeding
    print("🔄 Demo 5: Reproducible results with seeding")

    # Two labelers with same seed should give same results
    labeler_a = AutoLabeler("repro_a", settings, seed=123, noise_factor=0.1)
    labeler_b = AutoLabeler("repro_b", settings, seed=123, noise_factor=0.1)

    # Add same training data
    sample_train = main_df.head(2)
    labeler_a.add_training_data(sample_train, "text", "independence_score")
    labeler_b.add_training_data(sample_train, "text", "independence_score")

    print("Both labelers initialized with seed=123")
    print("Results should be reproducible across runs\n")

    print("🎉 Enhanced AutoLabeler Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  ✅ Metadata features for improved context")
    print("  ✅ Automatic train/test splits with evaluation")
    print("  ✅ Data leakage prevention")
    print("  ✅ Dynamic rule set loading")
    print("  ✅ Noise injection for variability")
    print("  ✅ Reproducible seeding")
    print("  ✅ Comprehensive evaluation metrics")

    print(f"\nDemo files created:")
    print(f"  📁 demo_data/ - Sample datasets and rules")
    print(f"  📁 demo_results/ - Evaluation results")


if __name__ == "__main__":
    demo_enhanced_features()
