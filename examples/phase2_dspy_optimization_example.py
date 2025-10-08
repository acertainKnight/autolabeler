"""Example: Using DSPy prompt optimization for improved labeling accuracy.

This example demonstrates how to use DSPy to automatically optimize prompts
for better labeling performance.
"""

import pandas as pd
from autolabeler.config import Settings
from autolabeler.core.configs import LabelingConfig, DSPyOptimizationConfig
from autolabeler.core.labeling import OptimizedLabelingService


def main():
    """Demonstrate DSPy prompt optimization."""

    # Sample sentiment analysis data
    train_data = pd.DataFrame({
        'text': [
            'This product exceeded my expectations!',
            'Terrible quality, complete waste of money.',
            'Pretty decent for the price.',
            'Absolutely love it, best purchase ever!',
            'Not worth the money, very disappointed.',
            'Good enough, does what it promises.',
            'Outstanding quality and service!',
            'Would not recommend to anyone.',
            'Satisfied with my purchase overall.',
            'Horrible experience from start to finish.',
        ],
        'label': [
            'positive', 'negative', 'neutral',
            'positive', 'negative', 'neutral',
            'positive', 'negative', 'neutral',
            'negative',
        ],
    })

    val_data = pd.DataFrame({
        'text': [
            'Amazing product, highly recommend!',
            'Worst purchase I ever made.',
            'It\'s okay, nothing special.',
            'Fantastic, exceeded expectations!',
            'Very poor quality.',
        ],
        'label': [
            'positive', 'negative', 'neutral',
            'positive', 'negative',
        ],
    })

    test_data = pd.DataFrame({
        'text': [
            'Great value for money!',
            'Disappointing results.',
            'Average product, works fine.',
        ],
    })

    print('=' * 80)
    print('DSPy Prompt Optimization Example')
    print('=' * 80)
    print()

    # Initialize settings
    settings = Settings()

    # Configure labeling with RAG
    labeling_config = LabelingConfig(
        use_rag=True,
        k_examples=3,
        use_validation=True,
        allowed_labels=['positive', 'negative', 'neutral'],
    )

    # Configure DSPy optimization
    dspy_config = DSPyOptimizationConfig(
        enabled=True,
        model_name='gpt-4o-mini',
        num_candidates=5,  # Test 5 prompt variations
        num_trials=10,  # Run 10 optimization trials
        max_labeled_demos=4,  # Include up to 4 examples in prompt
        metric_threshold=0.8,  # Target 80% accuracy
        cache_optimized_prompts=True,
    )

    # Initialize optimized labeling service
    print('1. Initializing OptimizedLabelingService with DSPy...')
    service = OptimizedLabelingService(
        dataset_name='sentiment_demo',
        settings=settings,
        config=labeling_config,
        dspy_config=dspy_config,
    )

    # Add training data to knowledge base
    print('2. Adding training data to knowledge base...')
    service.knowledge_store.add_examples(
        train_data, 'text', 'label', source='human'
    )
    print(f'   Knowledge base now has {len(train_data)} examples')
    print()

    # Optimize prompts using DSPy
    print('3. Running DSPy prompt optimization...')
    print('   This may take a few minutes...')
    optimization_result = service.optimize_prompts(
        train_df=train_data,
        val_df=val_data,
        text_column='text',
        label_column='label',
    )

    print()
    print('Optimization Results:')
    print(f'  Train Accuracy:      {optimization_result.train_accuracy:.2%}')
    print(f'  Validation Accuracy: {optimization_result.validation_accuracy:.2%}')
    print(f'  Optimization Cost:   ${optimization_result.optimization_cost:.2f}')
    print(f'  Converged:           {optimization_result.converged}')
    print()
    print('  Best Prompt Instructions:')
    print(f'  {optimization_result.best_prompt[:200]}...')
    print()

    # Use optimized service for labeling
    print('4. Labeling test data with optimized prompts...')
    results = []
    for _, row in test_data.iterrows():
        response = service.label_text(row['text'])
        results.append({
            'text': row['text'],
            'predicted_label': response.label,
            'confidence': response.confidence,
            'reasoning': response.reasoning,
        })

    results_df = pd.DataFrame(results)

    print()
    print('Labeling Results:')
    print(results_df.to_string(index=False))
    print()

    # Get optimization stats
    stats = service.get_optimization_stats()
    print('5. Optimization Statistics:')
    print(f'   DSPy Enabled:        {stats["dspy_enabled"]}')
    print(f'   Optimizer Available: {stats["has_optimizer"]}')
    print(f'   Results Cached:      {stats["optimization_cached"]}')
    print()

    # Compare with non-optimized baseline (optional)
    print('6. Comparison with baseline (without optimization):')
    print('   You can compare the optimized results with a baseline')
    print('   LabelingService to see the improvement!')
    print()

    print('=' * 80)
    print('Example complete!')
    print('=' * 80)


if __name__ == '__main__':
    main()
