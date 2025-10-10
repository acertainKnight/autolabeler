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
        core_rules: dict[str, list[str]] | None = None,
        improvement_strategy: str = "feedback_driven",
        settings: Settings | None = None,
    ):
        """Initialize rule evolution service.

        Args:
            initial_rules: Starting rules for each task (includes core + any evolved)
            core_rules: Core human-written rules that cannot be modified (from config)
            improvement_strategy: Strategy for rule improvement
            settings: Settings for LLM-based rule generation
        """
        self.rules = initial_rules or {}
        self.core_rules = core_rules or {}  # Protected rules from original config
        self.improvement_strategy = improvement_strategy
        self.settings = settings or Settings()
        self.error_history: list[ErrorPattern] = []
        self.rule_generation_count = 0

        # Track which rules are AI-generated (can be modified)
        self.ai_generated_rules: dict[str, list[str]] = {
            task: [] for task in self.rules.keys()
        }

        logger.info(
            f"Rule evolution service initialized with strategy: {improvement_strategy}"
        )
        if core_rules:
            core_count = sum(len(rules) for rules in core_rules.values())
            logger.info(f"Protected {core_count} core rules from modification")

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
                        examples=[str(idx) for idx in examples.index.tolist()],
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
                    examples=[str(idx) for idx in borderline.iloc[:5].index.tolist()],
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

            if self.settings.llm_provider == "openrouter":
                from .openrouter_client import OpenRouterClient

                client = OpenRouterClient(
                    api_key=self.settings.openrouter_api_key,
                    model=self.settings.llm_model,
                    temperature=0.3,
                    use_rate_limiter=True,
                )
                response = client.create(
                    messages=[{"role": "user", "content": prompt}], max_tokens=256
                )
                new_rule = response["choices"][0]["message"]["content"].strip()
            elif self.settings.llm_provider == "anthropic":
                from anthropic import Anthropic

                client = Anthropic(api_key=self.settings.anthropic_api_key)
                response = client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=256,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                new_rule = response.content[0].text.strip()
            else:
                from openai import OpenAI

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

    def _is_duplicate_rule(self, new_rule: str, existing_rules: list[str], similarity_threshold: float = 0.8) -> bool:
        """Check if a new rule is too similar to existing rules.

        Args:
            new_rule: New rule to check
            existing_rules: List of existing rules
            similarity_threshold: Threshold for considering rules as duplicates (0-1)

        Returns:
            True if rule is duplicate, False otherwise
        """
        # Normalize for comparison
        new_normalized = new_rule.lower().strip()

        for existing_rule in existing_rules:
            existing_normalized = existing_rule.lower().strip()

            # Exact match
            if new_normalized == existing_normalized:
                return True

            # Check if one is substring of another (partial duplication)
            if new_normalized in existing_normalized or existing_normalized in new_normalized:
                return True

            # Check for high similarity (fuzzy matching)
            # Simple word-based similarity
            new_words = set(new_normalized.split())
            existing_words = set(existing_normalized.split())

            if len(new_words) > 0 and len(existing_words) > 0:
                intersection = len(new_words & existing_words)
                union = len(new_words | existing_words)
                similarity = intersection / union

                if similarity >= similarity_threshold:
                    return True

        return False

    def _can_modify_rule(self, rule: str, task_name: str) -> bool:
        """Check if a rule can be modified (i.e., it's not a core rule).

        Args:
            rule: Rule to check
            task_name: Task name

        Returns:
            True if rule can be modified, False if it's a protected core rule
        """
        if task_name not in self.core_rules:
            return True

        # Check if this rule matches any core rule
        rule_normalized = rule.lower().strip()
        for core_rule in self.core_rules[task_name]:
            if rule_normalized == core_rule.lower().strip():
                return False

        return True

    def improve_rules(
        self,
        current_rules: dict[str, list[str]],
        feedback_data: pd.DataFrame,
        error_patterns: pd.DataFrame | None = None,
        max_rules_per_task: int = 25,
        allow_modifications: bool = True,
    ) -> dict[str, list[str]]:
        """Generate improved rules based on error patterns and feedback.

        Can both ADD new rules and MODIFY existing AI-generated rules.
        Core human-written rules are protected from modification.

        Args:
            current_rules: Current rules for each task
            feedback_data: DataFrame with uncertain/incorrect predictions
            error_patterns: Optional pre-identified error patterns
            max_rules_per_task: Maximum number of rules per task (prevents unbounded growth)
            allow_modifications: If True, can modify existing AI-generated rules (not core rules)

        Returns:
            Updated rules dictionary
        """
        logger.info(f"Starting rule improvement with {len(feedback_data)} examples")

        task_names = list(current_rules.keys())
        patterns = self.identify_error_patterns(feedback_data, task_names)

        updated_rules = {task: rules.copy() for task, rules in current_rules.items()}

        # Track AI-generated rules for this task
        for task_name in task_names:
            core_count = len(self.core_rules.get(task_name, []))
            ai_rules = updated_rules[task_name][core_count:]
            if task_name not in self.ai_generated_rules:
                self.ai_generated_rules[task_name] = []
            self.ai_generated_rules[task_name] = ai_rules

        # Generate new rules or modify existing ones based on patterns
        for pattern in patterns[:5]:  # Limit to top 5 patterns per batch
            task_name = pattern.task_name

            # Check if we've hit the max rules limit
            core_count = len(self.core_rules.get(task_name, []))
            ai_rule_count = len(updated_rules[task_name]) - core_count

            if len(updated_rules[task_name]) >= max_rules_per_task:
                logger.warning(
                    f"Task '{task_name}' has reached max rules limit ({max_rules_per_task})"
                )

                # Try to modify an existing AI-generated rule instead of adding
                if allow_modifications and ai_rule_count > 0:
                    logger.info(f"Attempting to refine an existing AI-generated rule for {task_name}")
                    modified_rule = self._modify_existing_rule(
                        task_name, pattern, updated_rules[task_name]
                    )
                    if modified_rule:
                        continue
                else:
                    logger.warning(f"Skipping - cannot add or modify rules for {task_name}")
                    continue

            if self.improvement_strategy == "feedback_driven":
                # Use LLM to generate sophisticated rule
                new_rule = self._generate_rule_from_llm(
                    task_name, pattern, updated_rules[task_name]
                )
            else:
                # Use pattern-based heuristic rule
                new_rule = pattern.suggested_rule

            # Check for duplicates using fuzzy matching
            if new_rule and not self._is_duplicate_rule(new_rule, updated_rules[task_name]):
                updated_rules[task_name].append(new_rule)
                self.ai_generated_rules[task_name].append(new_rule)
                logger.info(f"Added new rule to {task_name}: {new_rule[:100]}...")
            elif new_rule:
                logger.debug(f"Skipped duplicate rule for {task_name}: {new_rule[:100]}...")

        # Report improvement summary
        for task_name, rules in updated_rules.items():
            original_count = len(current_rules[task_name])
            new_count = len(rules)
            core_count = len(self.core_rules.get(task_name, []))
            ai_count = new_count - core_count

            if new_count > original_count:
                logger.info(
                    f"Task '{task_name}': {original_count} → {new_count} rules "
                    f"(+{new_count - original_count}) [core: {core_count}, AI: {ai_count}]"
                )

        return updated_rules

    def _modify_existing_rule(
        self, task_name: str, error_pattern: ErrorPattern, current_rules: list[str]
    ) -> str | None:
        """Modify an existing AI-generated rule to address error pattern.

        Args:
            task_name: Name of the classification task
            error_pattern: Identified error pattern
            current_rules: Current rules for the task

        Returns:
            Modified rule or None if modification fails
        """
        # Find AI-generated rules (exclude core rules)
        core_count = len(self.core_rules.get(task_name, []))
        ai_rules = current_rules[core_count:]

        if not ai_rules:
            logger.debug(f"No AI-generated rules to modify for {task_name}")
            return None

        try:
            # Ask LLM to identify which rule should be refined and how
            prompt = f"""You are refining classification rules based on error patterns.

Task: {task_name}
Core Rules (PROTECTED, cannot be modified):
{chr(10).join(f'- {rule}' for rule in self.core_rules.get(task_name, []))}

AI-Generated Rules (can be modified):
{chr(10).join(f'{i+1}. {rule}' for i, rule in enumerate(ai_rules))}

Error Pattern Identified:
- Type: {error_pattern.pattern_type}
- Frequency: {error_pattern.frequency} cases
- Initial suggestion: {error_pattern.suggested_rule}

Which AI-generated rule should be refined to address this error? Provide:
1. The rule number to modify (1-{len(ai_rules)})
2. The improved version of that rule

Format: RULE_NUMBER|IMPROVED_RULE_TEXT"""

            if self.settings.llm_provider == "openrouter":
                from .openrouter_client import OpenRouterClient

                client = OpenRouterClient(
                    api_key=self.settings.openrouter_api_key,
                    model=self.settings.llm_model,
                    temperature=0.3,
                    use_rate_limiter=True,
                )
                response = client.create(
                    messages=[{"role": "user", "content": prompt}], max_tokens=512
                )
                result = response["choices"][0]["message"]["content"].strip()
            elif self.settings.llm_provider == "anthropic":
                from anthropic import Anthropic

                client = Anthropic(api_key=self.settings.anthropic_api_key)
                response = client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=512,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.content[0].text.strip()
            else:
                from openai import OpenAI

                client = OpenAI(api_key=self.settings.openai_api_key)
                response = client.chat.completions.create(
                    model=self.settings.llm_model,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.choices[0].message.content.strip()

            # Parse response
            if "|" in result:
                parts = result.split("|", 1)
                rule_num = int(parts[0].strip()) - 1
                improved_rule = parts[1].strip()

                if 0 <= rule_num < len(ai_rules):
                    old_rule = ai_rules[rule_num]
                    # Replace in the full rules list
                    rule_idx = core_count + rule_num
                    current_rules[rule_idx] = improved_rule
                    logger.info(f"Modified rule #{rule_num + 1} for {task_name}")
                    logger.info(f"  Old: {old_rule[:80]}...")
                    logger.info(f"  New: {improved_rule[:80]}...")
                    return improved_rule

        except Exception as e:
            logger.error(f"Failed to modify existing rule: {e}")

        return None

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
