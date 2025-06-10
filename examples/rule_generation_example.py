#!/usr/bin/env python3
"""
Example script demonstrating the rule generation feature.

This script shows how to:
1. Generate labeling rules from existing labeled data
2. Update rules with new data
3. Export rules for human annotators
4. Use the generated rules for consistent labeling
"""

import pandas as pd
from pathlib import Path

from autolabeler import Settings, RuleGenerator, AutoLabeler


def main():
    """Demonstrate rule generation workflow."""

    # Configuration
    settings = Settings(
        openrouter_api_key="your-openrouter-api-key",  # Replace with your key
        llm_model="openai/gpt-3.5-turbo",
        max_examples_per_query=5,
    )

    dataset_name = "sentiment_analysis_demo"

    # Sample labeled data for demonstration
    # In practice, this would be loaded from your existing labeled dataset
    sample_data = {
        "text": [
            "I absolutely love this product! It exceeded my expectations.",
            "This is terrible quality. I want my money back.",
            "The product is okay, nothing special but works fine.",
            "Amazing service and fast delivery. Highly recommend!",
            "Poor quality materials and bad customer service.",
            "It's an average product for the price point.",
            "Outstanding quality and beautiful design!",
            "The worst purchase I've ever made. Completely useless.",
            "Decent product but could be improved in some areas.",
            "Excellent value for money. Very satisfied!",
            "Not worth the price. Found better alternatives elsewhere.",
            "Good product overall, meets my basic needs.",
            "Incredible attention to detail. Love everything about it!",
            "Disappointed with the quality. Expected much better.",
            "It's fine for what it is. Nothing to complain about.",
        ],
        "sentiment": [
            "positive", "negative", "neutral",
            "positive", "negative", "neutral",
            "positive", "negative", "neutral",
            "positive", "negative", "neutral",
            "positive", "negative", "neutral",
        ]
    }

    df = pd.DataFrame(sample_data)

    print("🚀 Rule Generation Demo")
    print("=" * 50)

    # Step 1: Generate initial rules from labeled data
    print("\n📊 Step 1: Generating labeling rules from training data...")

    rule_generator = RuleGenerator(dataset_name, settings)

    result = rule_generator.generate_rules_from_data(
        df=df,
        text_column="text",
        label_column="sentiment",
        task_description="Classify customer review sentiment as positive, negative, or neutral",
        batch_size=10,
        min_examples_per_rule=2,
    )

    ruleset = result.ruleset
    print(f"✅ Generated {len(ruleset.rules)} rules for {len(ruleset.label_categories)} labels")

    # Show some details about the generated rules
    print("\n📋 Generated Rules Summary:")
    rules_by_label = {}
    for rule in ruleset.rules:
        if rule.label not in rules_by_label:
            rules_by_label[rule.label] = []
        rules_by_label[rule.label].append(rule)

    for label, rules in rules_by_label.items():
        print(f"\n{label.upper()} ({len(rules)} rules):")
        for i, rule in enumerate(rules[:2], 1):  # Show first 2 rules per label
            print(f"  {i}. {rule.pattern_description}")
            print(f"     Confidence: {rule.confidence:.2f}")
            print(f"     Key indicators: {', '.join(rule.indicators[:3])}")

    # Step 2: Export rules for human annotators
    print("\n📝 Step 2: Exporting annotation guidelines...")

    guidelines_path = Path("sentiment_annotation_guidelines.md")
    rule_generator.export_ruleset_for_humans(
        ruleset, guidelines_path, format="markdown"
    )
    print(f"✅ Guidelines exported to: {guidelines_path}")

    # Step 3: Simulate adding new data and updating rules
    print("\n🔄 Step 3: Updating rules with new data...")

    # New sample data
    new_data = {
        "text": [
            "This product changed my life! Absolutely revolutionary.",
            "Complete garbage. Worst company ever.",
            "It works as advertised, nothing more nothing less.",
            "Fantastic customer support and great product quality!",
            "Overpriced for what you get. Not recommended.",
        ],
        "sentiment": ["positive", "negative", "neutral", "positive", "negative"]
    }

    new_df = pd.DataFrame(new_data)

    update_result = rule_generator.update_rules_with_new_data(
        new_df=new_df,
        text_column="text",
        label_column="sentiment",
    )

    print(f"✅ Rules updated:")
    print(f"   - New rules added: {update_result.new_rules_added}")
    print(f"   - Rules modified: {update_result.rules_modified}")
    print(f"   - Rules removed: {update_result.rules_removed}")

    if update_result.changes_made:
        print("\n🔧 Key changes made:")
        for change in update_result.changes_made[:3]:  # Show first 3 changes
            print(f"   - {change}")

    # Step 4: Use the rules in practice
    print("\n🎯 Step 4: Using generated rules for consistent labeling...")

    # Create an AutoLabeler that can reference the generated rules
    labeler = AutoLabeler(dataset_name, settings)

    # Add the training data to knowledge base for RAG
    labeler.add_training_data(df, "text", "sentiment", source="human")
    labeler.add_training_data(new_df, "text", "sentiment", source="human")

    # Test labeling with the knowledge base
    test_texts = [
        "This is the best purchase I've made all year!",
        "The product broke after one week. Very disappointed.",
        "It's an okay product, does what it's supposed to do.",
    ]

    print("\n🧪 Testing labels with RAG-enhanced knowledge base:")
    for text in test_texts:
        result = labeler.label_text(text, use_rag=True)
        print(f"\nText: '{text}'")
        print(f"Label: {result.label}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")

    # Step 5: Show recommendations
    print("\n💡 Recommendations for improving the ruleset:")
    if result.recommendations:
        for rec in result.recommendations:
            print(f"   - {rec}")

    # Step 6: Coverage analysis
    print("\n📈 Rule Coverage Analysis:")
    if result.coverage_analysis:
        coverage = result.coverage_analysis
        print(f"Overall coverage: {coverage.get('overall_coverage', 0):.1%} of training data")

        if 'coverage_by_label' in coverage:
            print("\nCoverage by label:")
            for label, stats in coverage['coverage_by_label'].items():
                ratio = stats.get('coverage_ratio', 0)
                print(f"  {label}: {ratio:.1%} ({stats.get('total_examples', 0)} examples)")

    print("\n✨ Rule generation demo completed!")
    print(f"Check {guidelines_path} for human-readable annotation guidelines.")
    print("\nNext steps:")
    print("1. Review and refine the generated rules")
    print("2. Share the guidelines with your annotation team")
    print("3. Use the rules to maintain consistency in labeling")
    print("4. Periodically update rules with new examples")


if __name__ == "__main__":
    main()
