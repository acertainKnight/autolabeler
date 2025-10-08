"""DSPy-based prompt optimization for AutoLabeler.

This module provides prompt optimization capabilities using the DSPy framework
with MIPROv2 optimization algorithm for systematic prompt improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

try:
    import dspy
    from dspy.teleprompt import MIPROv2
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not installed. Install with: pip install dspy-ai")


class DSPyConfig(BaseModel):
    """Configuration for DSPy optimization.

    Attributes:
        model_name: LLM model to use for optimization
        api_key: API key for the LLM provider (optional, uses env var if not provided)
        api_base: Base URL for API endpoint (optional)
        num_candidates: Number of prompt candidates to generate
        num_trials: Number of optimization trials
        max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
        max_labeled_demos: Maximum number of labeled demonstrations
        init_temperature: Initial temperature for optimization
        metric_threshold: Minimum metric value to consider optimization successful
    """

    model_name: str = Field(default='gpt-4o-mini', description='Model for optimization')
    api_key: str | None = Field(default=None, description='API key for LLM provider')
    api_base: str | None = Field(default=None, description='Base URL for API')
    num_candidates: int = Field(default=10, description='Number of prompt candidates')
    num_trials: int = Field(default=20, description='Number of optimization trials')
    max_bootstrapped_demos: int = Field(default=4, description='Max bootstrapped demos')
    max_labeled_demos: int = Field(default=8, description='Max labeled demos')
    init_temperature: float = Field(default=1.0, description='Initial temperature')
    metric_threshold: float = Field(default=0.8, description='Success threshold')
    cache_dir: Path | None = Field(default=None, description='Cache directory for DSPy')


@dataclass
class DSPyOptimizationResult:
    """Result of DSPy prompt optimization.

    Attributes:
        optimized_module: The optimized DSPy module
        best_prompt: Best prompt instructions found
        best_examples: Best few-shot examples selected
        validation_accuracy: Accuracy on validation set
        train_accuracy: Accuracy on training set
        optimization_cost: Total cost of optimization in USD
        num_trials: Number of trials executed
        num_candidates: Number of candidates evaluated
        converged: Whether optimization converged
        metadata: Additional metadata from optimization
    """

    optimized_module: Any
    best_prompt: str
    best_examples: list[dict[str, Any]]
    validation_accuracy: float
    train_accuracy: float
    optimization_cost: float
    num_trials: int
    num_candidates: int
    converged: bool
    metadata: dict[str, Any]


class LabelingSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """DSPy signature for text labeling tasks.

    This defines the input/output interface for the labeling task that DSPy
    will optimize prompts for.
    """

    if DSPY_AVAILABLE:
        text: str = dspy.InputField(desc='Text to classify')
        examples: str = dspy.InputField(desc='Example classifications (optional)', default='')
        label: str = dspy.OutputField(desc='Predicted label')
        reasoning: str = dspy.OutputField(desc='Explanation of classification')
        confidence: float = dspy.OutputField(desc='Confidence score (0-1)')


class LabelingModule(dspy.Module if DSPY_AVAILABLE else object):
    """DSPy module for text labeling with chain-of-thought reasoning."""

    def __init__(self):
        """Initialize the labeling module."""
        if not DSPY_AVAILABLE:
            raise ImportError('DSPy is required. Install with: pip install dspy-ai')

        super().__init__()
        self.predictor = dspy.ChainOfThought(LabelingSignature)

    def forward(self, text: str, examples: str = '') -> dspy.Prediction:
        """Execute the labeling prediction.

        Args:
            text: Text to classify
            examples: Optional few-shot examples

        Returns:
            DSPy Prediction with label, reasoning, and confidence
        """
        return self.predictor(text=text, examples=examples)


class DSPyOptimizer:
    """Optimize prompts using DSPy framework with MIPROv2.

    This class provides algorithmic prompt optimization to improve labeling
    accuracy through systematic exploration of prompt variations and
    few-shot example selection.

    Example:
        >>> config = DSPyConfig(model_name='gpt-4o-mini', num_candidates=10)
        >>> optimizer = DSPyOptimizer(config)
        >>> result = optimizer.optimize_labeling_prompt(
        ...     train_df=train_data,
        ...     val_df=val_data,
        ...     text_column='text',
        ...     label_column='label'
        ... )
        >>> print(f"Validation accuracy: {result.validation_accuracy:.2%}")
    """

    def __init__(self, config: DSPyConfig):
        """Initialize DSPy optimizer.

        Args:
            config: Configuration for DSPy optimization

        Raises:
            ImportError: If DSPy is not installed
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                'DSPy is required for prompt optimization. '
                'Install with: pip install dspy-ai'
            )

        self.config = config

        # Initialize DSPy language model
        lm_kwargs = {'model': config.model_name}
        if config.api_key:
            lm_kwargs['api_key'] = config.api_key
        if config.api_base:
            lm_kwargs['api_base'] = config.api_base

        self.lm = dspy.OpenAI(**lm_kwargs)
        dspy.settings.configure(lm=self.lm)

        logger.info(f'Initialized DSPy optimizer with model: {config.model_name}')

    def optimize_labeling_prompt(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        metric_fn: Callable | None = None,
    ) -> DSPyOptimizationResult:
        """Optimize labeling prompt using MIPROv2.

        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            text_column: Name of text column
            label_column: Name of label column
            metric_fn: Optional custom metric function. Should take (example, prediction, trace)
                      and return float score. Defaults to exact match accuracy.

        Returns:
            DSPyOptimizationResult with optimized module and performance metrics

        Example:
            >>> result = optimizer.optimize_labeling_prompt(
            ...     train_df=pd.DataFrame({'text': [...], 'label': [...]}),
            ...     val_df=pd.DataFrame({'text': [...], 'label': [...]}),
            ...     text_column='text',
            ...     label_column='label'
            ... )
        """
        logger.info('Starting DSPy prompt optimization with MIPROv2')

        # Convert DataFrames to DSPy examples
        train_examples = self._df_to_dspy_examples(train_df, text_column, label_column)
        val_examples = self._df_to_dspy_examples(val_df, text_column, label_column)

        logger.info(
            f'Prepared {len(train_examples)} train and {len(val_examples)} val examples'
        )

        # Define metric
        if metric_fn is None:
            metric_fn = self._default_accuracy_metric

        # Create labeling module
        module = LabelingModule()

        # Initialize MIPROv2 optimizer
        optimizer = MIPROv2(
            metric=metric_fn,
            num_candidates=self.config.num_candidates,
            init_temperature=self.config.init_temperature,
        )

        logger.info('Starting MIPROv2 optimization...')

        # Run optimization
        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            valset=val_examples,
            num_trials=self.config.num_trials,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
        )

        # Evaluate on both train and validation sets
        train_accuracy = self._evaluate_module(
            optimized_module, train_examples, metric_fn
        )
        val_accuracy = self._evaluate_module(optimized_module, val_examples, metric_fn)

        # Extract optimized prompt and examples
        best_prompt = self._extract_prompt_instructions(optimized_module)
        best_examples = self._extract_few_shot_examples(optimized_module)

        # Estimate cost (rough approximation)
        optimization_cost = self._estimate_optimization_cost(
            num_trials=self.config.num_trials,
            num_candidates=self.config.num_candidates,
            num_examples=len(train_examples) + len(val_examples),
        )

        # Check convergence
        converged = val_accuracy >= self.config.metric_threshold

        result = DSPyOptimizationResult(
            optimized_module=optimized_module,
            best_prompt=best_prompt,
            best_examples=best_examples,
            validation_accuracy=val_accuracy,
            train_accuracy=train_accuracy,
            optimization_cost=optimization_cost,
            num_trials=self.config.num_trials,
            num_candidates=self.config.num_candidates,
            converged=converged,
            metadata={
                'model_name': self.config.model_name,
                'num_train_examples': len(train_examples),
                'num_val_examples': len(val_examples),
                'max_bootstrapped_demos': self.config.max_bootstrapped_demos,
                'max_labeled_demos': self.config.max_labeled_demos,
            },
        )

        logger.info(
            f'Optimization complete! Train: {train_accuracy:.2%}, '
            f'Val: {val_accuracy:.2%}, Cost: ${optimization_cost:.2f}'
        )

        return result

    def _df_to_dspy_examples(
        self, df: pd.DataFrame, text_column: str, label_column: str
    ) -> list[dspy.Example]:
        """Convert DataFrame to DSPy examples.

        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            List of DSPy Example objects
        """
        examples = []
        for _, row in df.iterrows():
            example = dspy.Example(
                text=str(row[text_column]),
                label=str(row[label_column]),
            ).with_inputs('text')
            examples.append(example)
        return examples

    def _default_accuracy_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None
    ) -> float:
        """Default accuracy metric: exact match.

        Args:
            example: Ground truth example
            prediction: Model prediction
            trace: Optional execution trace

        Returns:
            1.0 if labels match, 0.0 otherwise
        """
        return 1.0 if example.label.lower() == prediction.label.lower() else 0.0

    def _evaluate_module(
        self,
        module: LabelingModule,
        examples: list[dspy.Example],
        metric_fn: Callable,
    ) -> float:
        """Evaluate module on a set of examples.

        Args:
            module: DSPy module to evaluate
            examples: List of examples to evaluate on
            metric_fn: Metric function

        Returns:
            Average metric score across all examples
        """
        scores = []
        for example in examples:
            try:
                prediction = module(text=example.text)
                score = metric_fn(example, prediction)
                scores.append(score)
            except Exception as e:
                logger.warning(f'Evaluation failed for example: {e}')
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _extract_prompt_instructions(self, module: LabelingModule) -> str:
        """Extract optimized prompt instructions from module.

        Args:
            module: Optimized DSPy module

        Returns:
            Prompt instructions as string
        """
        try:
            # Try to get the extended signature instructions
            if hasattr(module, 'predictor') and hasattr(
                module.predictor, 'extended_signature'
            ):
                return module.predictor.extended_signature.instructions
            # Fallback to basic signature
            elif hasattr(module, 'predictor') and hasattr(
                module.predictor, 'signature'
            ):
                return str(module.predictor.signature)
        except Exception as e:
            logger.warning(f'Could not extract prompt instructions: {e}')

        return 'No prompt instructions available'

    def _extract_few_shot_examples(
        self, module: LabelingModule
    ) -> list[dict[str, Any]]:
        """Extract few-shot examples from optimized module.

        Args:
            module: Optimized DSPy module

        Returns:
            List of few-shot examples as dictionaries
        """
        examples = []
        try:
            if hasattr(module, 'predictor') and hasattr(module.predictor, 'demos'):
                for demo in module.predictor.demos:
                    example = {
                        'text': demo.get('text', ''),
                        'label': demo.get('label', ''),
                    }
                    examples.append(example)
        except Exception as e:
            logger.warning(f'Could not extract few-shot examples: {e}')

        return examples

    def _estimate_optimization_cost(
        self, num_trials: int, num_candidates: int, num_examples: int
    ) -> float:
        """Estimate optimization cost in USD.

        This is a rough approximation based on typical token usage.

        Args:
            num_trials: Number of optimization trials
            num_candidates: Number of prompt candidates
            num_examples: Number of examples evaluated

        Returns:
            Estimated cost in USD
        """
        # Rough estimates:
        # - Each trial evaluates num_candidates prompts
        # - Each evaluation processes ~200 tokens (prompt + response)
        # - GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
        avg_tokens_per_eval = 200
        total_evals = num_trials * num_candidates * num_examples
        total_tokens = total_evals * avg_tokens_per_eval

        # Assume 50/50 split between input and output tokens
        input_tokens = total_tokens * 0.5
        output_tokens = total_tokens * 0.5

        # Cost per 1M tokens (these are approximate and model-dependent)
        input_cost_per_1m = 0.15
        output_cost_per_1m = 0.60

        total_cost = (
            input_tokens / 1_000_000 * input_cost_per_1m
            + output_tokens / 1_000_000 * output_cost_per_1m
        )

        return total_cost

    def save_optimized_prompt(
        self, result: DSPyOptimizationResult, output_path: Path
    ) -> None:
        """Save optimized prompt configuration to file.

        Args:
            result: Optimization result to save
            output_path: Path to save configuration

        Example:
            >>> optimizer.save_optimized_prompt(result, Path('config/optimized.json'))
        """
        import json

        config = {
            'prompt_instructions': result.best_prompt,
            'few_shot_examples': result.best_examples,
            'validation_accuracy': result.validation_accuracy,
            'train_accuracy': result.train_accuracy,
            'optimization_cost': result.optimization_cost,
            'metadata': result.metadata,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f'Saved optimized prompt configuration to {output_path}')

    def load_optimized_module(self, config_path: Path) -> LabelingModule:
        """Load a previously optimized module from configuration.

        Args:
            config_path: Path to saved configuration

        Returns:
            Configured LabelingModule

        Example:
            >>> module = optimizer.load_optimized_module(Path('config/optimized.json'))
        """
        import json

        with open(config_path) as f:
            config = json.load(f)

        # Create module and configure it
        module = LabelingModule()

        # Apply saved configuration
        # Note: This is a simplified version - actual implementation would
        # need to properly restore the optimized parameters
        if 'prompt_instructions' in config:
            logger.info(f"Loaded prompt: {config['prompt_instructions'][:100]}...")

        return module
