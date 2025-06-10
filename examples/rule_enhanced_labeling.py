#!/usr/bin/env python3
"""
Integration example showing how rule generation enhances AutoLabeler consistency.

This script demonstrates:
1. Using generated rules to improve annotation consistency
2. Combining rule-based guidelines with RAG examples
3. Measuring improvement in labeling quality
"""

import pandas as pd
from pathlib import Path

from autolabeler import Settings, RuleGenerator, AutoLabeler


def simulate_human_annotation_with_rules(text: str, guidelines: str) -> str:
    """
    Simulate how human annotators might perform better with clear guidelines.

    In reality, this would be actual human annotation, but for demonstration
    we'll simulate the improvement that clear rules provide.
    """
    # Simple heuristic based on the guidelines content
    text_lower = text.lower()

    if any(word in text_lower for word in ["love", "amazing", "excellent", "perfect", "outstanding"]):
        return "positive"
    elif any(word in text_lower for word in ["hate", "terrible", "awful", "worst", "horrible"]):
        return "negative"
    else:
        return "neutral"


def main():
    """Demonstrate rule-enhanced labeling workflow."""

    # Configuration
    settings = Settings(
        openrouter_api_key="your-openrouter-api-key",  # Replace with your key
        llm_model="openai/gpt-3.5-turbo",
        max_examples_per_query=5,
    )

    dataset_name = "enhanced_sentiment_demo"

    print("🚀 Rule-Enhanced AutoLabeling Demo")
    print("=" * 50)

    # Step 1: Initial labeled dataset
    print("\n📊 Step 1: Setting up initial labeled dataset...")

    initial_data = {
        "text": [
            "I absolutely love this product! It's incredible.",
            "This is terrible quality. Completely disappointed.",
            "The product is okay, works as expected.",
            "Amazing service and lightning-fast delivery!",
            "Poor build quality and overpriced.",
            "It's a decent product for the price point.",
            "Outstanding design and excellent functionality!",
            "Worst purchase I've ever made. Total garbage.",
            "Average product, nothing special but adequate.",
            "Perfect quality and amazing value for money!",
        ],
        "sentiment": [
            "positive", "negative", "neutral",
            "positive", "negative", "neutral",
            "positive", "negative", "neutral",
            "positive"
        ]
    }

    df = pd.DataFrame(initial_data)
    print(f"✅ Initial dataset: {len(df)} labeled examples")

    # Step 2: Generate annotation guidelines
    print("\n📋 Step 2: Generating annotation guidelines from existing data...")

    rule_generator = RuleGenerator(dataset_name, settings)

    result = rule_generator.generate_rules_from_data(
        df=df,
        text_column="text",
        label_column="sentiment",
        task_description="Classify customer feedback sentiment consistently",
        batch_size=10,
        min_examples_per_rule=2,
    )

    print(f"✅ Generated {len(result.ruleset.rules)} annotation rules")

    # Export guidelines for human annotators
    guidelines_path = Path("enhanced_sentiment_guidelines.md")
    rule_generator.export_ruleset_for_humans(
        result.ruleset, guidelines_path, format="markdown"
    )
    print(f"✅ Guidelines exported to: {guidelines_path}")

    # Step 3: Set up AutoLabeler with knowledge base
    print("\n🧠 Step 3: Setting up AutoLabeler with knowledge base...")

    labeler = AutoLabeler(dataset_name, settings)
    labeler.add_training_data(df, "text", "sentiment", source="human")

    print("✅ AutoLabeler initialized with training data")

    # Step 4: Test consistency without and with guidelines
    print("\n🧪 Step 4: Testing annotation consistency...")

    test_cases = [
        "This product exceeds all my expectations! Absolutely fantastic.",
        "Complete waste of money. The quality is atrocious.",
        "It works fine, nothing to write home about but does the job.",
        "Incredible attention to detail and superb customer service!",
        "Overpriced garbage with terrible build quality.",
        "Fair product for the price, meets basic requirements adequately.",
        "Mind-blowing quality and design! Best purchase ever!",
        "Horrible experience. Product broke immediately and support was useless.",
        "Standard product, performs as advertised without issues.",
        "Phenomenal value and outstanding performance across the board!",
    ]

    expected_labels = [
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive"
    ]

    # Read the generated guidelines
    guidelines_content = guidelines_path.read_text() if guidelines_path.exists() else ""

    print("\n📈 Results Comparison:")
    print("Text | Expected | AutoLabeler | Simulated Human (with rules) | Match AL | Match Human")
    print("-" * 100)

    al_matches = 0
    human_matches = 0

    for i, (text, expected) in enumerate(zip(test_cases, expected_labels)):
        # Get AutoLabeler prediction with RAG
        al_result = labeler.label_text(text, use_rag=True)
        al_label = al_result.label

        # Simulate human annotation with guidelines
        human_label = simulate_human_annotation_with_rules(text, guidelines_content)

        # Check matches
        al_match = "✓" if al_label == expected else "✗"
        human_match = "✓" if human_label == expected else "✗"

        if al_label == expected:
            al_matches += 1
        if human_label == expected:
            human_matches += 1

        # Truncate text for display
        display_text = text[:40] + "..." if len(text) > 40 else text

        print(f"{display_text:43} | {expected:8} | {al_label:11} | {human_label:25} | {al_match:8} | {human_match}")

    # Calculate accuracy
    al_accuracy = al_matches / len(test_cases)
    human_accuracy = human_matches / len(test_cases)

    print(f"\n📊 Accuracy Summary:")
    print(f"AutoLabeler (with RAG): {al_accuracy:.1%} ({al_matches}/{len(test_cases)})")
    print(f"Simulated Human (with rules): {human_accuracy:.1%} ({human_matches}/{len(test_cases)})")

    # Step 5: Demonstrate rule updating with new data
    print("\n🔄 Step 5: Updating rules with new annotated data...")

    # Simulate new data being annotated
    new_data = {
        "text": [
            "This revolutionary product has changed my life completely!",
            "Absolute trash. Don't waste your money on this junk.",
            "It's a standard product that works reasonably well.",
            "Spectacular quality and unmatched performance throughout!",
            "Disappointing quality for the premium price point.",
        ],
        "sentiment": ["positive", "negative", "neutral", "positive", "negative"]
    }

    new_df = pd.DataFrame(new_data)

    # Update the knowledge base
    labeler.add_training_data(new_df, "text", "sentiment", source="human")

    # Update the rules
    update_result = rule_generator.update_rules_with_new_data(
        new_df=new_df,
        text_column="text",
        label_column="sentiment",
    )

    print(f"✅ Rules updated:")
    print(f"   - New rules added: {update_result.new_rules_added}")
    print(f"   - Rules modified: {update_result.rules_modified}")

    # Step 6: Test with updated knowledge
    print("\n🎯 Step 6: Testing with updated knowledge base...")

    final_test_texts = [
        "Revolutionary innovation with spectacular results!",
        "Complete junk and waste of money.",
        "Standard functionality, works as intended.",
    ]

    print("\nFinal test with enhanced knowledge base:")
    for text in final_test_texts:
        result = labeler.label_text(text, use_rag=True)
        print(f"Text: '{text}'")
        print(f"Label: {result.label} (confidence: {result.confidence:.2f})")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")
        print()

    # Step 7: Show the value proposition
    print("💡 Key Benefits of Rule-Enhanced Labeling:")
    print("1. 🎯 Consistent Guidelines: Clear, evidence-based rules for annotators")
    print("2. 🔄 Adaptive Learning: Rules evolve with new data")
    print("3. 📚 Knowledge Preservation: Institutional knowledge is captured")
    print("4. 🤝 Human-AI Collaboration: Rules guide both humans and AI")
    print("5. 📈 Quality Improvement: Measurable consistency gains")

    print("\n✨ Demo completed!")
    print(f"Check {guidelines_path} for the generated annotation guidelines.")
    print("\nNext steps for production use:")
    print("1. Gather a larger, high-quality labeled dataset")
    print("2. Generate comprehensive rules covering edge cases")
    print("3. Train annotators using the generated guidelines")
    print("4. Monitor consistency and update rules periodically")
    print("5. Use rules to guide both human annotators and AI systems")


if __name__ == "__main__":
    main()
