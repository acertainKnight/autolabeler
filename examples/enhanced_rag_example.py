#!/usr/bin/env python3
"""
Enhanced RAG AutoLabeling Example

This example demonstrates the advanced autolabeling pipeline with:
- Persistent dataset-specific knowledge bases
- Incremental learning from model predictions
- Full provenance tracking for model-generated labels
- RAG retrieval with preference for human-labeled examples
"""

from pathlib import Path
import pandas as pd
from loguru import logger

from autolabeler.config import Settings
from autolabeler.labeler import AutoLabeler


def main():
    """Demonstrate enhanced RAG autolabeling workflow."""

    # Initialize settings
    settings = Settings()
    logger.info("Starting enhanced RAG autolabeling example")

    # Create AutoLabeler for sentiment analysis dataset
    labeler = AutoLabeler("sentiment_analysis", settings)

    # Step 1: Add initial human-labeled training data
    logger.info("Step 1: Adding initial human-labeled training data")
    initial_data = pd.DataFrame({
        "text": [
            "This movie was absolutely fantastic!",
            "Terrible acting, worst film ever",
            "It was okay, nothing special",
            "Amazing cinematography and storyline",
            "Boring and predictable plot"
        ],
        "sentiment": ["positive", "negative", "neutral", "positive", "negative"],
        "source": "human_expert",
        "confidence": [0.95, 0.98, 0.85, 0.92, 0.89]
    })

    labeler.add_training_data(initial_data, "text", "sentiment", source="human")

    # Check knowledge base stats
    stats = labeler.get_knowledge_base_stats()
    logger.info(f"Knowledge base stats: {stats}")

    # Step 2: Label new data using RAG
    logger.info("Step 2: Labeling new data with RAG examples")
    new_texts = [
        "This film exceeded all my expectations!",
        "Complete waste of time, don't bother watching",
        "The movie was fine, nothing extraordinary",
        "Incredible performance by the lead actor",
        "Fell asleep halfway through, very dull"
    ]

    # Label each text and show RAG examples used
    for i, text in enumerate(new_texts, 1):
        logger.info(f"\nLabeling text {i}: '{text}'")
        result = labeler.label_text(text, use_rag=True, prefer_human_examples=True)
        logger.info(f"Prediction: {result.label} (confidence: {result.confidence:.2f})")
        if result.reasoning:
            logger.info(f"Reasoning: {result.reasoning}")

    # Step 3: Batch label a DataFrame with automatic knowledge base updates
    logger.info("\nStep 3: Batch labeling with knowledge base updates")
    batch_data = pd.DataFrame({
        "review": [
            "Outstanding cinematography and acting",
            "Poorly written script, amateur direction",
            "Average movie, some good moments",
            "Masterpiece of modern cinema",
            "Couldn't finish it, too boring",
            "Decent entertainment value",
            "Revolutionary filmmaking techniques"
        ]
    })

    # Label with automatic addition of high-confidence predictions to KB
    labeled_batch = labeler.label_dataframe(
        batch_data,
        text_column="review",
        label_column="predicted_sentiment",
        use_rag=True,
        save_to_knowledge_base=True,
        confidence_threshold=0.8  # Only add high-confidence predictions
    )

    logger.info("\nBatch labeling results:")
    for _, row in labeled_batch.iterrows():
        logger.info(
            f"'{row['review'][:50]}...' -> "
            f"{row['predicted_sentiment']} "
            f"(conf: {row['predicted_sentiment_confidence']:.2f})"
        )

    # Step 4: Check updated knowledge base stats
    logger.info("\nStep 4: Updated knowledge base statistics")
    updated_stats = labeler.get_knowledge_base_stats()
    logger.info(f"Total examples: {updated_stats.get('total_examples', 0)}")
    logger.info(f"Sources: {updated_stats.get('sources', {})}")

    # Step 5: Export knowledge base for inspection
    logger.info("\nStep 5: Exporting knowledge base")
    export_path = Path("exported_sentiment_kb.csv")
    labeler.export_knowledge_base(export_path)
    logger.info(f"Knowledge base exported to {export_path}")

    # Show a sample of the exported data
    if export_path.exists():
        kb_data = pd.read_csv(export_path)
        logger.info("\nSample of knowledge base data:")
        logger.info(kb_data[['text', 'label', 'source', 'added_at']].head())

    # Step 6: Demonstrate RAG retrieval for a new query
    logger.info("\nStep 6: RAG retrieval demonstration")
    query = "This movie has incredible special effects"

    # Get similar examples manually for inspection
    similar_examples = labeler.knowledge_base.get_similar_examples(
        query, k=3, filter_source="human"
    )

    logger.info(f"\nFor query: '{query}'")
    logger.info("Similar human-labeled examples found:")
    for i, ex in enumerate(similar_examples, 1):
        logger.info(f"{i}. '{ex.page_content}' -> {ex.metadata.get('label')}")

    # Now label with RAG
    result = labeler.label_text(query, use_rag=True)
    logger.info(f"\nFinal prediction: {result.label} (confidence: {result.confidence:.2f})")

    logger.info("\nEnhanced RAG autolabeling example completed!")


if __name__ == "__main__":
    main()
