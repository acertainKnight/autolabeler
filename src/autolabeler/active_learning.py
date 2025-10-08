"""Rule evolution service for improving classification principles through active learning."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from .config import Settings


class ErrorPattern(BaseModel):
    """Identified error pattern from uncertain predictions."""

    task_name: str
    pattern_type: str  # low_confidence, label_confusion, edge_case
    frequency: int
    examples: list[str] = Field(default_factory=list)
    suggested_rule: str | None = None


class RuleEvolutionService:
    """Improve classification rules based on feedback and error patterns."""

    def __init__(
        self,
        initial_rules: dict[str, list[str]] | None = None,
        improvement_strategy: str = "feedback_driven",
        settings: Settings | None = None,
    ):
        """Initialize rule evolution service.

        Args:
            initial_rules: Starting rules for each task
            improvement_strategy: Strategy for rule improvement
            settings: Settings for LLM-based rule generation
        """
        self.rules = initial_rules or {}
        self.improvement_strategy = improvement_strategy
        self.settings = settings or Settings()
        self.error_history: list[ErrorPattern] = []
        self.rule_generation_count = 0

        logger.info(
            f"Rule evolution service initialized with strategy: {improvement_strategy}"
        )

    def identify_error_patterns(
        self, feedback_data: pd.DataFrame, task_names: list[str]
    ) -> list[ErrorPattern]:
        """Identify patterns in uncertain or incorrect predictions.

        Args:
            feedback_data: DataFrame with uncertain predictions
            task_names: List of task names to analyze

        Returns:
            List of identified error patterns
        """
        patterns = []

        for task_name in task_names:
            label_col = f"label_{task_name}"
            confidence_col = f"confidence_{task_name}"

            if label_col not in feedback_data.columns:
                continue

            # Pattern 1: Low confidence predictions
            low_conf = feedback_data[feedback_data[confidence_col] < 0.7]
            if len(low_conf) > 0:
                label_dist = Counter(low_conf[label_col])
                for label, count in label_dist.most_common(3):
                    examples = low_conf[low_conf[label_col] == label].iloc[:3]
                    pattern = ErrorPattern(
                        task_name=task_name,
                        pattern_type="low_confidence",
                        frequency=count,
                        examples=examples.index.tolist(),
                        suggested_rule=f"Clarify criteria for '{label}' classification",
                    )
                    patterns.append(pattern)

            # Pattern 2: Borderline confidence (0.5-0.7)
            borderline = feedback_data[
                (feedback_data[confidence_col] >= 0.5)
                & (feedback_data[confidence_col] < 0.7)
            ]
            if len(borderline) > 5:
                pattern = ErrorPattern(
                    task_name=task_name,
                    pattern_type="edge_case",
                    frequency=len(borderline),
                    examples=borderline.iloc[:5].index.tolist(),
                    suggested_rule=f"Add rules for edge cases in {task_name}",
                )
                patterns.append(pattern)

        self.error_history.extend(patterns)
        logger.info(f"Identified {len(patterns)} error patterns")
        return patterns

    def _generate_rule_from_llm(
        self, task_name: str, error_pattern: ErrorPattern, current_rules: list[str]
    ) -> str | None:
        """Use LLM to generate an improved rule based on error pattern.

        Args:
            task_name: Name of the classification task
            error_pattern: Identified error pattern
            current_rules: Current rules for the task

        Returns:
            Suggested new rule or None if generation fails
        """
        try:
            from anthropic import Anthropic
            from openai import OpenAI

            prompt = f"""You are a rule generation system for classification tasks.

Task: {task_name}
Current Rules:
{chr(10).join(f'- {rule}' for rule in current_rules)}

Error Pattern Identified:
- Type: {error_pattern.pattern_type}
- Frequency: {error_pattern.frequency} cases
- Initial suggestion: {error_pattern.suggested_rule}

Generate ONE specific, actionable rule that would help reduce this error pattern.
The rule should be concise, clear, and directly address the issue.
Return ONLY the rule text, no explanation."""

            if self.settings.llm_provider == "anthropic":
                client = Anthropic(api_key=self.settings.anthropic_api_key)
                response = client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=256,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                new_rule = response.content[0].text.strip()
            else:
                client = OpenAI(api_key=self.settings.openai_api_key)
                response = client.chat.completions.create(
                    model=self.settings.llm_model,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                new_rule = response.choices[0].message.content.strip()

            self.rule_generation_count += 1
            return new_rule

        except Exception as e:
            logger.error(f"Failed to generate rule via LLM: {e}")
            return None

    def improve_rules(
        self,
        current_rules: dict[str, list[str]],
        feedback_data: pd.DataFrame,
        error_patterns: pd.DataFrame | None = None,
    ) -> dict[str, list[str]]:
        """Generate improved rules based on error patterns and feedback.

        Args:
            current_rules: Current rules for each task
            feedback_data: DataFrame with uncertain/incorrect predictions
            error_patterns: Optional pre-identified error patterns

        Returns:
            Updated rules dictionary
        """
        logger.info(f"Starting rule improvement with {len(feedback_data)} examples")

        task_names = list(current_rules.keys())
        patterns = self.identify_error_patterns(feedback_data, task_names)

        updated_rules = {task: rules.copy() for task, rules in current_rules.items()}

        # Generate new rules based on patterns
        for pattern in patterns[:5]:  # Limit to top 5 patterns per batch
            task_name = pattern.task_name

            if self.improvement_strategy == "feedback_driven":
                # Use LLM to generate sophisticated rule
                new_rule = self._generate_rule_from_llm(
                    task_name, pattern, updated_rules[task_name]
                )
            else:
                # Use pattern-based heuristic rule
                new_rule = pattern.suggested_rule

            if new_rule and new_rule not in updated_rules[task_name]:
                updated_rules[task_name].append(new_rule)
                logger.info(f"Added new rule to {task_name}: {new_rule}")

        # Report improvement summary
        for task_name, rules in updated_rules.items():
            original_count = len(current_rules[task_name])
            new_count = len(rules)
            if new_count > original_count:
                logger.info(
                    f"Task '{task_name}': {original_count} → {new_count} rules (+{new_count - original_count})"
                )

        return updated_rules

    def get_improvement_stats(self) -> dict[str, Any]:
        """Get statistics about rule improvement process.

        Returns:
            Dictionary with improvement statistics
        """
        pattern_types = Counter(p.pattern_type for p in self.error_history)

        return {
            "total_patterns_identified": len(self.error_history),
            "pattern_types": dict(pattern_types),
            "rules_generated": self.rule_generation_count,
            "improvement_strategy": self.improvement_strategy,
        }

    def reset_history(self):
        """Reset error history and statistics."""
        self.error_history = []
        self.rule_generation_count = 0
        logger.info("Rule evolution history reset")
