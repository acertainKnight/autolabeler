"""DSPy-based prompt optimization for the labeling pipeline.

Uses MIPROv2 to algorithmically improve prompt instructions and select
the best few-shot examples, starting from the existing hand-crafted
prompts in prompts/{dataset}/.

Workflow:
    1. Load existing prompts (system.md, rules.md, etc.) as baseline
    2. Load labeled data and split into train/val
    3. MIPROv2 explores variations of the prompt instructions
    4. MIPROv2 selects the best few-shot examples from training data
    5. Evaluate optimized vs baseline on validation set
    6. Output report + optionally update prompt files
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DSPyConfig(BaseModel):
    """Configuration for DSPy optimization.

    Attributes:
        model_name: LLM model to use for optimization (litellm format).
        api_key: API key for the LLM provider (optional, uses env var if not).
        api_base: Base URL for API endpoint (optional).
        num_candidates: Number of prompt candidates MIPROv2 generates.
        num_trials: Number of optimization trials.
        max_bootstrapped_demos: Max bootstrapped demonstrations per trial.
        max_labeled_demos: Max labeled demonstrations per trial.
        init_temperature: Temperature for candidate generation.
        metric_threshold: Min metric value to consider optimization successful.
        cache_dir: Cache directory for DSPy LM responses.
    """

    model_name: str = Field(default="openai/gpt-4o-mini")
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)
    num_candidates: int = Field(default=10)
    num_trials: int = Field(default=20)
    max_bootstrapped_demos: int = Field(default=4)
    max_labeled_demos: int = Field(default=8)
    init_temperature: float = Field(default=1.0)
    metric_threshold: float = Field(default=0.8)
    cache_dir: Path | None = Field(default=None)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DSPyOptimizationResult:
    """Result of DSPy prompt optimization.

    Attributes:
        optimized_module: The optimized DSPy module (can be saved/loaded).
        optimized_instructions: The best instructions MIPROv2 found.
        selected_examples: Few-shot examples MIPROv2 chose from training data.
        baseline_accuracy: Accuracy of the original prompts on validation set.
        optimized_accuracy: Accuracy of the optimized prompts on validation set.
        improvement: Absolute accuracy improvement (optimized - baseline).
        optimization_cost: Estimated cost of the optimization run in USD.
        converged: Whether optimized accuracy meets the threshold.
        metadata: Additional metadata from the optimization run.
    """

    optimized_module: Any
    optimized_instructions: str
    selected_examples: list[dict[str, str]]
    baseline_accuracy: float
    optimized_accuracy: float
    improvement: float
    optimization_cost: float
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DSPy module that uses our existing prompts as baseline
# ---------------------------------------------------------------------------


def _build_signature(labels: list[str], baseline_instructions: str) -> type:
    """Dynamically build a DSPy Signature seeded with our existing prompts.

    Parameters:
        labels: Valid label values (e.g., ["-2", "-1", "0", "1", "2"]).
        baseline_instructions: Combined prompt text (rules + examples + mistakes)
            that MIPROv2 will use as the starting point and try to improve.

    Returns:
        A dspy.Signature subclass.
    """
    if not DSPY_AVAILABLE:
        raise ImportError("DSPy is required. Install with: pip install dspy-ai")

    label_str = ", ".join(labels)

    class LabelingSignature(dspy.Signature):
        __doc__ = baseline_instructions

        text: str = dspy.InputField(desc="The text to classify")
        label: str = dspy.OutputField(
            desc=f"Classification label. Must be one of: {label_str}"
        )
        reasoning: str = dspy.OutputField(
            desc="Brief explanation for why this label was chosen"
        )

    return LabelingSignature


class LabelingModule(dspy.Module if DSPY_AVAILABLE else object):
    """DSPy module wrapping ChainOfThought with our labeling signature."""

    def __init__(self, signature_cls: type | None = None):
        """Initialize the labeling module.

        Parameters:
            signature_cls: A dspy.Signature class built by _build_signature().
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required. Install with: pip install dspy-ai")

        super().__init__()
        if signature_cls is None:
            raise ValueError("signature_cls is required")
        self.predictor = dspy.ChainOfThought(signature_cls)

    def forward(self, text: str) -> dspy.Prediction:
        """Run classification prediction.

        Parameters:
            text: Text to classify.

        Returns:
            DSPy Prediction with label and reasoning.
        """
        return self.predictor(text=text)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class DSPyOptimizer:
    """Optimize prompts using DSPy MIPROv2 with existing prompts as baseline.

    This takes your hand-crafted prompts (rules.md, examples.md, mistakes.md)
    and uses MIPROv2 to find better phrasings and better few-shot example
    selections, evaluated against your labeled data.

    Example:
        >>> from sibyls.core.prompts.registry import PromptRegistry
        >>> from sibyls.core.dataset_config import DatasetConfig
        >>>
        >>> config = DSPyConfig(model_name="openrouter/google/gemini-2.5-flash")
        >>> optimizer = DSPyOptimizer(config)
        >>>
        >>> dataset_config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
        >>> prompts = PromptRegistry("fed_headlines")
        >>>
        >>> result = optimizer.optimize(
        ...     train_df=train_data,
        ...     val_df=val_data,
        ...     text_column="headline",
        ...     label_column="label_hawk_dove",
        ...     dataset_config=dataset_config,
        ...     prompt_registry=prompts,
        ... )
        >>> print(f"Baseline: {result.baseline_accuracy:.1%}")
        >>> print(f"Optimized: {result.optimized_accuracy:.1%}")
        >>> print(f"Improvement: +{result.improvement:.1%}")
    """

    def __init__(self, config: DSPyConfig):
        """Initialize DSPy optimizer.

        Parameters:
            config: Configuration for DSPy optimization.

        Raises:
            ImportError: If DSPy is not installed.
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for prompt optimization. "
                "Install with: pip install dspy-ai"
            )

        self.config = config

        # Build litellm model name
        model_name = config.model_name
        if config.api_base and "openrouter.ai" in config.api_base:
            if not model_name.startswith("openrouter/"):
                model_name = f"openrouter/{model_name}"

        lm_kwargs: dict[str, Any] = {"model": model_name}
        if config.api_key:
            lm_kwargs["api_key"] = config.api_key
        if config.api_base:
            lm_kwargs["api_base"] = config.api_base

        self.lm = dspy.LM(**lm_kwargs)
        dspy.configure(lm=self.lm)

        logger.info(f"Initialized DSPy optimizer with model: {model_name}")

    # -- Main entry point ---------------------------------------------------

    def optimize(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        dataset_config: Any,
        prompt_registry: Any,
        metric_fn: Callable | None = None,
    ) -> DSPyOptimizationResult:
        """Run prompt optimization using MIPROv2.

        Parameters:
            train_df: Training data with text and label columns.
            val_df: Validation data with text and label columns.
            text_column: Name of the text column.
            label_column: Name of the label column.
            dataset_config: DatasetConfig for this task (provides valid labels).
            prompt_registry: PromptRegistry for this task (provides baseline prompts).
            metric_fn: Optional custom metric. Signature: (example, prediction, trace) -> float.
                Defaults to exact match accuracy.

        Returns:
            DSPyOptimizationResult with optimized vs baseline comparison.
        """
        logger.info("=" * 60)
        logger.info("STARTING DSPy PROMPT OPTIMIZATION")
        logger.info("=" * 60)

        # Build baseline instructions from existing prompts
        baseline_instructions = self._build_baseline_instructions(prompt_registry)
        logger.info(f"Baseline instructions: {len(baseline_instructions)} chars")

        # Build signature seeded with our prompts
        labels = dataset_config.labels
        signature_cls = _build_signature(labels, baseline_instructions)

        # Convert data to DSPy examples
        train_examples = self._df_to_examples(train_df, text_column, label_column)
        val_examples = self._df_to_examples(val_df, text_column, label_column)
        logger.info(f"Train: {len(train_examples)} examples, Val: {len(val_examples)} examples")

        # Metric
        if metric_fn is None:
            metric_fn = self._make_accuracy_metric(labels)

        # ----- Baseline evaluation -----
        logger.info("Evaluating baseline (existing prompts)...")
        baseline_module = LabelingModule(signature_cls)
        baseline_accuracy = self._evaluate(baseline_module, val_examples, metric_fn)
        logger.info(f"Baseline validation accuracy: {baseline_accuracy:.1%}")

        # ----- MIPROv2 optimization -----
        logger.info(
            f"Running MIPROv2: {self.config.num_candidates} candidates, "
            f"{self.config.num_trials} trials..."
        )

        optimizer = MIPROv2(
            metric=metric_fn,
            auto=None,
            num_candidates=self.config.num_candidates,
            init_temperature=self.config.init_temperature,
        )

        optimized_module = optimizer.compile(
            LabelingModule(signature_cls),
            trainset=train_examples,
            valset=val_examples,
            num_trials=self.config.num_trials,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
        )

        # ----- Evaluate optimized -----
        optimized_accuracy = self._evaluate(optimized_module, val_examples, metric_fn)
        improvement = optimized_accuracy - baseline_accuracy

        logger.info("=" * 60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Baseline accuracy:  {baseline_accuracy:.1%}")
        logger.info(f"Optimized accuracy: {optimized_accuracy:.1%}")
        logger.info(f"Improvement:        {improvement:+.1%}")

        # Extract what MIPROv2 found
        optimized_instructions = self._extract_instructions(optimized_module)
        selected_examples = self._extract_demos(optimized_module)

        logger.info(f"Selected {len(selected_examples)} few-shot examples")

        cost = self._estimate_cost(
            self.config.num_trials,
            self.config.num_candidates,
            len(train_examples) + len(val_examples),
        )

        return DSPyOptimizationResult(
            optimized_module=optimized_module,
            optimized_instructions=optimized_instructions,
            selected_examples=selected_examples,
            baseline_accuracy=baseline_accuracy,
            optimized_accuracy=optimized_accuracy,
            improvement=improvement,
            optimization_cost=cost,
            converged=optimized_accuracy >= self.config.metric_threshold,
            metadata={
                "model_name": self.config.model_name,
                "num_train": len(train_examples),
                "num_val": len(val_examples),
                "num_candidates": self.config.num_candidates,
                "num_trials": self.config.num_trials,
                "labels": labels,
            },
        )

    # -- Backward-compatible entry point ------------------------------------

    def optimize_labeling_prompt(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        metric_fn: Callable | None = None,
    ) -> DSPyOptimizationResult:
        """Legacy interface -- runs optimization without existing prompts.

        For new code, prefer `optimize()` which uses existing prompt files.

        Parameters:
            train_df: Training data DataFrame.
            val_df: Validation data DataFrame.
            text_column: Name of text column.
            label_column: Name of label column.
            metric_fn: Optional custom metric function.

        Returns:
            DSPyOptimizationResult with optimized module and metrics.
        """
        logger.warning(
            "Using legacy optimize_labeling_prompt(). "
            "Consider using optimize() with dataset_config and prompt_registry."
        )

        # Build a generic signature without existing prompts
        labels_from_data = sorted(train_df[label_column].astype(str).unique().tolist())
        baseline_instructions = (
            f"Classify the given text into one of these labels: "
            f"{', '.join(labels_from_data)}. "
            f"Provide the label and a brief reasoning."
        )
        signature_cls = _build_signature(labels_from_data, baseline_instructions)

        train_examples = self._df_to_examples(train_df, text_column, label_column)
        val_examples = self._df_to_examples(val_df, text_column, label_column)

        if metric_fn is None:
            metric_fn = self._make_accuracy_metric(labels_from_data)

        baseline_module = LabelingModule(signature_cls)
        baseline_accuracy = self._evaluate(baseline_module, val_examples, metric_fn)

        optimizer_obj = MIPROv2(
            metric=metric_fn,
            auto=None,
            num_candidates=self.config.num_candidates,
            init_temperature=self.config.init_temperature,
        )

        optimized_module = optimizer_obj.compile(
            LabelingModule(signature_cls),
            trainset=train_examples,
            valset=val_examples,
            num_trials=self.config.num_trials,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
        )

        optimized_accuracy = self._evaluate(optimized_module, val_examples, metric_fn)

        return DSPyOptimizationResult(
            optimized_module=optimized_module,
            optimized_instructions=self._extract_instructions(optimized_module),
            selected_examples=self._extract_demos(optimized_module),
            baseline_accuracy=baseline_accuracy,
            optimized_accuracy=optimized_accuracy,
            improvement=optimized_accuracy - baseline_accuracy,
            optimization_cost=self._estimate_cost(
                self.config.num_trials,
                self.config.num_candidates,
                len(train_examples) + len(val_examples),
            ),
            converged=optimized_accuracy >= self.config.metric_threshold,
            metadata={
                "model_name": self.config.model_name,
                "num_train": len(train_examples),
                "num_val": len(val_examples),
            },
        )

    # -- Prompt construction ------------------------------------------------

    @staticmethod
    def _build_baseline_instructions(prompt_registry: Any) -> str:
        """Combine existing prompt files into baseline instructions for MIPROv2.

        Parameters:
            prompt_registry: PromptRegistry instance.

        Returns:
            Combined instructions string.
        """
        parts = []

        parts.append(prompt_registry.get("system"))
        parts.append("")
        parts.append(prompt_registry.get("rules"))
        parts.append("")

        if prompt_registry.exists("mistakes"):
            parts.append(prompt_registry.get("mistakes"))
            parts.append("")

        # We intentionally exclude examples.md here --
        # MIPROv2 will select its own few-shot examples from training data
        # and we don't want to double up.

        return "\n".join(parts)

    # -- Data conversion ----------------------------------------------------

    @staticmethod
    def _df_to_examples(
        df: pd.DataFrame, text_column: str, label_column: str
    ) -> list:
        """Convert DataFrame rows to DSPy Example objects.

        Parameters:
            df: Input DataFrame.
            text_column: Name of text column.
            label_column: Name of label column.

        Returns:
            List of dspy.Example objects.
        """
        examples = []
        for _, row in df.iterrows():
            ex = dspy.Example(
                text=str(row[text_column]),
                label=str(row[label_column]),
            ).with_inputs("text")
            examples.append(ex)
        return examples

    # -- Metrics ------------------------------------------------------------

    @staticmethod
    def _make_accuracy_metric(labels: list[str]) -> Callable:
        """Build an accuracy metric that normalizes label strings.

        Parameters:
            labels: Valid label values.

        Returns:
            Metric function compatible with DSPy.
        """
        valid = {str(l).strip().lower() for l in labels}

        def accuracy(example, prediction, trace=None) -> float:
            pred = str(prediction.label).strip().lower()
            gold = str(example.label).strip().lower()

            # Exact match
            if pred == gold:
                return 1.0

            # Partial credit for adjacent labels (ordinal scales)
            try:
                pred_int = int(float(pred))
                gold_int = int(float(gold))
                if abs(pred_int - gold_int) == 1:
                    return 0.5
            except (ValueError, TypeError):
                pass

            return 0.0

        return accuracy

    # -- Evaluation ---------------------------------------------------------

    @staticmethod
    def _evaluate(
        module: Any, examples: list, metric_fn: Callable
    ) -> float:
        """Evaluate a module on a list of examples.

        Parameters:
            module: DSPy module.
            examples: List of dspy.Example objects.
            metric_fn: Metric function.

        Returns:
            Average metric score.
        """
        scores = []
        for ex in examples:
            try:
                pred = module(text=ex.text)
                scores.append(metric_fn(ex, pred))
            except Exception as e:
                logger.debug(f"Eval failed for one example: {e}")
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    # -- Extraction ---------------------------------------------------------

    @staticmethod
    def _extract_instructions(module: Any) -> str:
        """Extract the optimized instructions from a module.

        Parameters:
            module: Optimized DSPy module.

        Returns:
            Instructions string.
        """
        try:
            if hasattr(module, "predictor"):
                pred = module.predictor
                if hasattr(pred, "extended_signature"):
                    return pred.extended_signature.instructions
                if hasattr(pred, "signature"):
                    return str(pred.signature.instructions)
        except Exception as e:
            logger.warning(f"Could not extract instructions: {e}")
        return ""

    @staticmethod
    def _extract_demos(module: Any) -> list[dict[str, str]]:
        """Extract the selected few-shot demos from a module.

        Parameters:
            module: Optimized DSPy module.

        Returns:
            List of dicts with text, label, and reasoning keys.
        """
        demos = []
        try:
            if hasattr(module, "predictor") and hasattr(module.predictor, "demos"):
                for demo in module.predictor.demos:
                    d: dict[str, str] = {}
                    if hasattr(demo, "text"):
                        d["text"] = demo.text
                    elif isinstance(demo, dict):
                        d["text"] = demo.get("text", "")
                    if hasattr(demo, "label"):
                        d["label"] = str(demo.label)
                    elif isinstance(demo, dict):
                        d["label"] = str(demo.get("label", ""))
                    if hasattr(demo, "reasoning"):
                        d["reasoning"] = demo.reasoning
                    elif isinstance(demo, dict):
                        d["reasoning"] = demo.get("reasoning", "")
                    demos.append(d)
        except Exception as e:
            logger.warning(f"Could not extract demos: {e}")
        return demos

    # -- Cost estimation ----------------------------------------------------

    @staticmethod
    def _estimate_cost(
        num_trials: int, num_candidates: int, num_examples: int
    ) -> float:
        """Rough cost estimate for the optimization run.

        Parameters:
            num_trials: Number of MIPROv2 trials.
            num_candidates: Number of prompt candidates.
            num_examples: Total examples (train + val).

        Returns:
            Estimated cost in USD.
        """
        avg_tokens = 500  # prompt + response per eval
        total_evals = num_trials * num_candidates * num_examples
        total_tokens = total_evals * avg_tokens
        # Assume ~$0.50/1M tokens average (mix of input/output)
        return total_tokens / 1_000_000 * 0.50

    # -- Save / load --------------------------------------------------------

    def save_result(self, result: DSPyOptimizationResult, path: Path) -> None:
        """Save optimization result to JSON.

        Parameters:
            result: Optimization result.
            path: Output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "optimized_instructions": result.optimized_instructions,
            "selected_examples": result.selected_examples,
            "baseline_accuracy": result.baseline_accuracy,
            "optimized_accuracy": result.optimized_accuracy,
            "improvement": result.improvement,
            "optimization_cost": result.optimization_cost,
            "converged": result.converged,
            "metadata": result.metadata,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved optimization result to {path}")

    def update_prompt_files(
        self,
        result: DSPyOptimizationResult,
        prompt_registry: Any,
    ) -> dict[str, Path]:
        """Write DSPy's improvements back to prompt files.

        Creates *_dspy.md variants alongside the originals so you can
        diff and review before replacing.

        Parameters:
            result: Optimization result.
            prompt_registry: PromptRegistry for this dataset.

        Returns:
            Dict of {name: path} for files written.
        """
        written: dict[str, Path] = {}
        dataset_dir = prompt_registry.dataset_dir

        # Write optimized instructions as a new rules file
        if result.optimized_instructions:
            path = dataset_dir / "rules_dspy.md"
            path.write_text(result.optimized_instructions, encoding="utf-8")
            written["rules_dspy"] = path
            logger.info(f"Wrote optimized rules to {path}")

        # Write selected examples
        if result.selected_examples:
            lines = ["# DSPy-Selected Examples\n"]
            lines.append(
                "These examples were algorithmically selected by MIPROv2 "
                "as the most effective few-shot demonstrations.\n"
            )
            lines.append(f"Validation accuracy: {result.optimized_accuracy:.1%} "
                         f"(baseline: {result.baseline_accuracy:.1%})\n")
            lines.append("---\n")

            for i, ex in enumerate(result.selected_examples, 1):
                text = ex.get("text", "")
                label = ex.get("label", "")
                reasoning = ex.get("reasoning", "")
                lines.append(f"### Example {i}\n")
                lines.append(f'**"{text}"**')
                lines.append(f"→ **{label}**")
                if reasoning:
                    lines.append(f": {reasoning}")
                lines.append("\n")

            path = dataset_dir / "examples_dspy.md"
            path.write_text("\n".join(lines), encoding="utf-8")
            written["examples_dspy"] = path
            logger.info(f"Wrote {len(result.selected_examples)} selected examples to {path}")

        return written
