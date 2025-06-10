"""
Enhanced Context-Aware RAG Labeling Examples

This module demonstrates advanced techniques for improving RAG embeddings
and retrieval using additional metadata and context information.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any

from autolabeler import AutoLabeler
from autolabeler.config import Settings


def example_1_metadata_enhanced_knowledge_base() -> None:
    """
    Example 1: Building a knowledge base with rich metadata.

    Shows how to add training data with comprehensive metadata
    that can be used for filtering and boosting retrieval.
    """
    print("=== Example 1: Metadata-Enhanced Knowledge Base ===")

    # Initialize settings and labeler
    settings = Settings(
        openrouter_api_key="your-api-key",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        max_examples_per_query=5
    )

    labeler = AutoLabeler("enhanced_sentiment", settings)

    # Create sample training data with rich metadata
    training_data = pd.DataFrame({
        "text": [
            "This smartphone has amazing battery life",
            "The customer service was terrible",
            "Great value for money on this laptop",
            "The food quality has declined recently",
            "Excellent medical care from the staff"
        ],
        "sentiment": ["positive", "negative", "positive", "negative", "positive"],
        "domain": ["technology", "service", "technology", "food", "healthcare"],
        "category": ["electronics", "support", "electronics", "restaurant", "medical"],
        "quality_score": [0.95, 0.8, 0.9, 0.75, 0.98],
        "recency": [30, 15, 45, 60, 10],  # days ago
        "source_type": ["review", "complaint", "review", "feedback", "testimonial"]
    })

    # Add additional metadata for each domain
    additional_metadata = {
        "collection_method": "user_survey",
        "verified": True,
        "quality_checked": True
    }

    # Add training data with enhanced metadata
    labeler.add_training_data(
        training_data,
        "text",
        "sentiment",
        source="human"
    )

    # Add the same data but with additional metadata
    labeler.knowledge_base.add_labeled_data(
        training_data,
        "text",
        "sentiment",
        source="human",
        additional_metadata=additional_metadata
    )

    print(f"Knowledge base stats: {labeler.get_knowledge_base_stats()}")


def example_2_context_aware_labeling() -> None:
    """
    Example 2: Context-aware labeling with domain-specific retrieval.

    Demonstrates how to use context information to improve
    labeling accuracy by retrieving domain-relevant examples.
    """
    print("\n=== Example 2: Context-Aware Labeling ===")

    settings = Settings(
        openrouter_api_key="your-api-key",
        max_examples_per_query=3
    )

    labeler = AutoLabeler("domain_specific", settings)

    # Test text to label
    test_text = "The new feature update improved performance significantly"

    # Basic labeling without context
    basic_result = labeler.label_text(test_text, use_rag=True)
    print(f"Basic labeling: {basic_result.label} (confidence: {basic_result.confidence})")

    # Context-aware labeling for technology domain
    tech_context = {
        "domain": "technology",
        "category": "software",
        "user_intent": "feature_evaluation",
        "background": "User reviewing a software update after 1 week of use",
        "boost": {
            "data_quality_score": 1.5,  # Boost high-quality examples
            "data_recency": 1.2,  # Boost recent examples
        }
    }

    context_result = labeler.label_text_with_context(test_text, tech_context)
    print(f"Context-aware labeling: {context_result.label} (confidence: {context_result.confidence})")
    print(f"Context influence: {context_result.context_influence}")


def example_3_metadata_filtering_and_boosting() -> None:
    """
    Example 3: Advanced metadata filtering and boosting.

    Shows how to use metadata filters and boost factors
    to retrieve the most relevant examples.
    """
    print("\n=== Example 3: Metadata Filtering and Boosting ===")

    settings = Settings(openrouter_api_key="your-api-key")
    labeler = AutoLabeler("advanced_retrieval", settings)

    # Example query
    query_text = "The product exceeded my expectations"

    # Method 1: Filter by exact metadata matches
    metadata_filters = {
        "data_domain": "technology",
        "data_verified": True
    }

    if hasattr(labeler.knowledge_base, 'get_similar_examples_with_metadata_filter'):
        filtered_examples = labeler.knowledge_base.get_similar_examples_with_metadata_filter(
            query_text,
            k=5,
            metadata_filters=metadata_filters
        )
        print(f"Found {len(filtered_examples)} examples with metadata filters")

    # Method 2: Boost certain metadata fields
    boost_metadata = {
        "data_quality_score": 1.5,  # Boost examples with high quality scores
        "data_recency": 0.1,        # Slight boost for recency (assuming lower days = more recent)
        "custom_verified": 2.0      # Strong boost for verified examples
    }

    if hasattr(labeler.knowledge_base, 'get_similar_examples_with_metadata_filter'):
        boosted_examples = labeler.knowledge_base.get_similar_examples_with_metadata_filter(
            query_text,
            k=5,
            boost_metadata=boost_metadata
        )
        print(f"Found {len(boosted_examples)} examples with metadata boosting")


def example_4_hybrid_semantic_metadata_search() -> None:
    """
    Example 4: Hybrid search combining semantic similarity and metadata matching.

    Demonstrates advanced hybrid search that balances semantic
    similarity with metadata relevance.
    """
    print("\n=== Example 4: Hybrid Semantic-Metadata Search ===")

    settings = Settings(openrouter_api_key="your-api-key")
    labeler = AutoLabeler("hybrid_search", settings)

    query_text = "Outstanding customer support experience"

    # Context for enhanced embedding
    search_context = {
        "domain": "service",
        "category": "support",
        "intent": "satisfaction_review"
    }

    # Metadata filters
    filters = {
        "data_quality_score": 0.8  # Only high-quality examples
    }

    # Perform hybrid search
    if hasattr(labeler.knowledge_base, 'hybrid_search'):
        hybrid_results = labeler.knowledge_base.hybrid_search(
            query_text,
            context=search_context,
            k=5,
            alpha=0.7,  # 70% semantic, 30% metadata
            metadata_filters=filters
        )

        print(f"Hybrid search returned {len(hybrid_results)} results")
        for i, (doc, score) in enumerate(hybrid_results):
            print(f"  {i+1}. Score: {score:.3f}, Text: {doc.page_content[:50]}...")


def example_5_contextual_embeddings() -> None:
    """
    Example 5: Creating contextual embeddings.

    Shows how to create embeddings that include context information
    for better semantic matching within specific domains.
    """
    print("\n=== Example 5: Contextual Embeddings ===")

    settings = Settings(openrouter_api_key="your-api-key")
    labeler = AutoLabeler("contextual_embeddings", settings)

    # Text to embed
    text = "The treatment was very effective"

    # Different contexts for the same text
    contexts = [
        {"domain": "healthcare", "category": "medical_treatment"},
        {"domain": "beauty", "category": "skincare"},
        {"domain": "technology", "category": "software_solution"}
    ]

    print(f"Original text: '{text}'")

    for context in contexts:
        if hasattr(labeler.knowledge_base, 'create_contextual_embedding'):
            embedding = labeler.knowledge_base.create_contextual_embedding(text, context)
            print(f"Context {context['domain']}: Embedding dimension {len(embedding)}")


def example_6_comprehensive_workflow() -> None:
    """
    Example 6: Complete workflow with context-aware RAG.

    Demonstrates a full pipeline from data preparation to
    context-aware labeling with analytics.
    """
    print("\n=== Example 6: Comprehensive Context-Aware Workflow ===")

    settings = Settings(
        openrouter_api_key="your-api-key",
        max_examples_per_query=5,
        similarity_threshold=0.8
    )

    labeler = AutoLabeler("comprehensive_workflow", settings)

    # Step 1: Prepare training data with comprehensive metadata
    training_data = pd.DataFrame({
        "text": [
            "Love the new design and features",
            "Terrible user experience and bugs",
            "Great value and fast delivery",
            "Poor quality and overpriced",
            "Excellent customer service team"
        ],
        "label": ["positive", "negative", "positive", "negative", "positive"],
        "domain": ["technology", "technology", "ecommerce", "ecommerce", "service"],
        "category": ["ui_design", "software_quality", "purchase", "product_quality", "support"],
        "confidence": [0.95, 0.9, 0.85, 0.88, 0.92],
        "user_type": ["premium", "free", "new", "returning", "premium"]
    })

    # Step 2: Add training data with metadata
    domain_metadata = {
        "collection_date": "2024-01-15",
        "validation_method": "expert_review",
        "quality_threshold": 0.8
    }

    labeler.knowledge_base.add_labeled_data(
        training_data,
        "text",
        "label",
        source="human",
        additional_metadata=domain_metadata
    )

    # Step 3: Label new data with different context scenarios
    test_cases = [
        {
            "text": "The interface is intuitive and responsive",
            "context": {
                "domain": "technology",
                "category": "ui_design",
                "user_intent": "usability_feedback",
                "boost": {"data_confidence": 1.2}
            }
        },
        {
            "text": "Fast shipping and good packaging",
            "context": {
                "domain": "ecommerce",
                "category": "delivery",
                "user_intent": "purchase_satisfaction",
                "boost": {"data_user_type": 1.3}  # Boost examples from similar user types
            }
        }
    ]

    results = []
    for test_case in test_cases:
        result = labeler.label_text_with_context(
            test_case["text"],
            test_case["context"]
        )
        results.append({
            "text": test_case["text"],
            "predicted_label": result.label,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "context_influence": result.context_influence
        })

    # Step 4: Analyze results
    results_df = pd.DataFrame(results)
    print("\nLabeling Results:")
    print(results_df.to_string(index=False))

    # Step 5: Get analytics
    analytics = labeler.get_prompt_analytics()
    print(f"\nPrompt Analytics:")
    print(f"Total prompts: {analytics.get('total_prompts', 0)}")
    print(f"Success rate: {analytics.get('success_rate', 0):.2%}")
    print(f"Average confidence: {analytics.get('avg_confidence', 0):.3f}")


if __name__ == "__main__":
    """
    Run all examples to demonstrate enhanced RAG capabilities.

    Note: Replace 'your-api-key' with actual API keys before running.
    """

    # Run all examples
    example_1_metadata_enhanced_knowledge_base()
    example_2_context_aware_labeling()
    example_3_metadata_filtering_and_boosting()
    example_4_hybrid_semantic_metadata_search()
    example_5_contextual_embeddings()
    example_6_comprehensive_workflow()

    print("\n=== All Examples Completed ===")
    print("\nKey Takeaways for Enhanced RAG:")
    print("1. Use rich metadata to improve retrieval relevance")
    print("2. Apply context-aware embeddings for domain-specific matching")
    print("3. Implement metadata filtering and boosting for precision")
    print("4. Combine semantic and metadata similarity for hybrid search")
    print("5. Track context influence for better model interpretability")
    print("6. Use comprehensive analytics to optimize your pipeline")
