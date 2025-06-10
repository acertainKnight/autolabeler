from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template
from loguru import logger

from .config import Settings
from .corporate import CorporateOpenAIClient
from .models import (
    LabelingRule,
    RuleGenerationResult,
    RuleSet,
    RuleUpdateResult,
)
from .openrouter import OpenRouterClient


class RuleGenerator:
    """
    Generates and maintains labeling rules from training data.

    Analyzes patterns in labeled data to create comprehensive annotation guidelines
    that can be used by both human annotators and LLM autolabelers for consistency.

    Args:
        dataset_name (str): Unique identifier for this dataset.
        settings (Settings): Application settings containing LLM configurations.
        storage_path (Path | None): Path to store generated rulesets. Uses default if None.

    Example:
        >>> settings = Settings(openrouter_api_key="your-key")
        >>> generator = RuleGenerator("sentiment_analysis", settings)
        >>> result = generator.generate_rules_from_data(labeled_df, "text", "sentiment")
        >>> print(f"Generated {len(result.ruleset.rules)} rules")
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        storage_path: Path | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.settings = settings
        self.storage_path = storage_path or Path("rulesets")
        self.storage_path.mkdir(exist_ok=True)

        # Initialize LLM client for rule generation
        if settings.corporate_base_url:
            base_llm = CorporateOpenAIClient(
                api_key=settings.corporate_api_key,
                base_url=settings.corporate_base_url,
                model=settings.corporate_model,
            )
        else:
            base_llm = OpenRouterClient(
                api_key=settings.openrouter_api_key,
                model=settings.llm_model,
            )

        self.llm = base_llm.with_structured_output(schema=RuleGenerationResult)
        self.update_llm = base_llm.with_structured_output(schema=RuleUpdateResult)

        # Load templates for rule generation
        template_dir = Path(__file__).parent / "templates"
        self.generation_template = Template(
            (template_dir / "rule_generation.j2").read_text()
        )
        self.update_template = Template(
            (template_dir / "rule_update.j2").read_text()
        )

        logger.info(f"Initialized RuleGenerator for dataset: {dataset_name}")

    def generate_rules_from_data(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        task_description: str | None = None,
        batch_size: int = 50,
        min_examples_per_rule: int = 3,
    ) -> RuleGenerationResult:
        """
        Generate labeling rules from a dataset of labeled examples.

        Args:
            df (pd.DataFrame): DataFrame containing labeled training data.
            text_column (str): Name of column containing text data.
            label_column (str): Name of column containing labels.
            task_description (str | None): Description of the labeling task.
            batch_size (int): Number of examples to analyze per batch.
            min_examples_per_rule (int): Minimum examples needed to create a rule.

        Returns:
            RuleGenerationResult: Generated ruleset with metadata and analysis.

        Example:
            >>> result = generator.generate_rules_from_data(
            ...     labeled_df, "review_text", "sentiment",
            ...     task_description="Classify product review sentiment"
            ... )
        """
        logger.info(f"Generating rules from {len(df)} examples")

        # Analyze the data structure
        data_analysis = self._analyze_data(df, text_column, label_column)

        # Process data in batches to handle large datasets
        rules = []
        all_examples_by_label = defaultdict(list)

        # Group examples by label
        for _, row in df.iterrows():
            text = str(row[text_column])
            label = str(row[label_column])
            all_examples_by_label[label].append(text)

        # Generate rules for each label category
        for label, examples in all_examples_by_label.items():
            if len(examples) < min_examples_per_rule:
                logger.warning(f"Skipping label '{label}': only {len(examples)} examples")
                continue

            # Process examples in batches
            for i in range(0, len(examples), batch_size):
                batch_examples = examples[i:i + batch_size]
                batch_rules = self._generate_rules_for_batch(
                    label, batch_examples, data_analysis
                )
                rules.extend(batch_rules)

        # Merge and consolidate rules
        consolidated_rules = self._consolidate_rules(rules)

        # Create the ruleset
        ruleset = RuleSet(
            dataset_name=self.dataset_name,
            task_description=task_description or f"Labeling task for {self.dataset_name}",
            label_categories=list(data_analysis["label_distribution"].keys()),
            rules=consolidated_rules,
            general_guidelines=self._generate_general_guidelines(data_analysis),
            disambiguation_rules=self._generate_disambiguation_rules(consolidated_rules),
            quality_checks=self._generate_quality_checks(data_analysis),
            version="1.0.0",
            creation_timestamp=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            statistics=self._calculate_ruleset_statistics(consolidated_rules, data_analysis),
        )

        # Generate metadata about the process
        generation_metadata = {
            "total_examples_processed": len(df),
            "batch_size": batch_size,
            "min_examples_per_rule": min_examples_per_rule,
            "rules_generated": len(consolidated_rules),
            "model_used": self.settings.corporate_model if self.settings.corporate_base_url else self.settings.llm_model,
            "generation_timestamp": datetime.now().isoformat(),
        }

        result = RuleGenerationResult(
            ruleset=ruleset,
            generation_metadata=generation_metadata,
            data_analysis=data_analysis,
            coverage_analysis=self._analyze_rule_coverage(consolidated_rules, df, text_column, label_column),
            recommendations=self._generate_recommendations(consolidated_rules, data_analysis),
        )

        # Save the ruleset
        self._save_ruleset(result.ruleset)

        logger.info(f"Generated {len(consolidated_rules)} rules for {len(data_analysis['label_distribution'])} labels")
        return result

    def update_rules_with_new_data(
        self,
        new_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        existing_ruleset: RuleSet | None = None,
    ) -> RuleUpdateResult:
        """
        Update existing rules with new labeled data.

        Args:
            new_df (pd.DataFrame): New labeled data to incorporate.
            text_column (str): Name of column containing text data.
            label_column (str): Name of column containing labels.
            existing_ruleset (RuleSet | None): Existing ruleset to update. Loads latest if None.

        Returns:
            RuleUpdateResult: Updated ruleset with change information.

        Example:
            >>> update_result = generator.update_rules_with_new_data(
            ...     new_labeled_df, "text", "label"
            ... )
            >>> print(f"Made {len(update_result.changes_made)} changes")
        """
        if existing_ruleset is None:
            existing_ruleset = self.load_latest_ruleset()

        logger.info(f"Updating ruleset with {len(new_df)} new examples")

        # Analyze new data
        new_data_analysis = self._analyze_data(new_df, text_column, label_column)

        # Generate update prompt
        rendered_prompt = self.update_template.render(
            existing_ruleset=existing_ruleset.model_dump(),
            new_examples=self._prepare_examples_for_update(new_df, text_column, label_column),
            new_data_analysis=new_data_analysis,
        )

        # Get update recommendations from LLM
        update_result = self.update_llm.invoke(rendered_prompt)

        # Apply updates and track changes
        update_result.updated_ruleset.last_updated = datetime.now().isoformat()
        update_result.updated_ruleset.version = self._increment_version(existing_ruleset.version)

        # Save updated ruleset
        self._save_ruleset(update_result.updated_ruleset)

        logger.info(f"Updated ruleset: {update_result.new_rules_added} new, {update_result.rules_modified} modified")
        return update_result

    def load_latest_ruleset(self) -> RuleSet:
        """
        Load the most recent ruleset for this dataset.

        Returns:
            RuleSet: The latest saved ruleset.

        Raises:
            FileNotFoundError: If no ruleset exists for this dataset.
        """
        ruleset_files = list(self.storage_path.glob(f"{self.dataset_name}_ruleset_*.json"))
        if not ruleset_files:
            raise FileNotFoundError(f"No ruleset found for dataset: {self.dataset_name}")

        # Sort by modification time and get the latest
        latest_file = max(ruleset_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r') as f:
            ruleset_data = json.load(f)

        return RuleSet(**ruleset_data)

    def export_ruleset_for_humans(
        self,
        ruleset: RuleSet,
        output_path: Path,
        format: str = "markdown",
    ) -> None:
        """
        Export ruleset in human-readable format for annotators.

        Args:
            ruleset (RuleSet): Ruleset to export.
            output_path (Path): Path to save the exported guidelines.
            format (str): Export format ('markdown', 'html', 'json').

        Example:
            >>> generator.export_ruleset_for_humans(
            ...     ruleset, Path("annotation_guidelines.md")
            ... )
        """
        if format == "markdown":
            self._export_markdown_guidelines(ruleset, output_path)
        elif format == "html":
            self._export_html_guidelines(ruleset, output_path)
        elif format == "json":
            self._export_json_guidelines(ruleset, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported ruleset to {output_path} in {format} format")

    def _analyze_data(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> dict[str, Any]:
        """Analyze the structure and patterns in the labeled data."""
        analysis = {
            "total_examples": len(df),
            "label_distribution": dict(df[label_column].value_counts()),
            "avg_text_length": df[text_column].str.len().mean(),
            "text_length_distribution": {
                "min": df[text_column].str.len().min(),
                "max": df[text_column].str.len().max(),
                "median": df[text_column].str.len().median(),
                "std": df[text_column].str.len().std(),
            },
            "unique_labels": df[label_column].nunique(),
            "data_hash": self._hash_dataframe(df, [text_column, label_column]),
        }

        # Analyze text patterns by label
        label_patterns = {}
        for label in df[label_column].unique():
            label_texts = df[df[label_column] == label][text_column]
            label_patterns[str(label)] = {
                "count": len(label_texts),
                "avg_length": label_texts.str.len().mean(),
                "common_words": self._extract_common_words(label_texts.tolist()),
            }

        analysis["label_patterns"] = label_patterns
        return analysis

    def _generate_rules_for_batch(
        self,
        label: str,
        examples: list[str],
        data_analysis: dict[str, Any],
    ) -> list[LabelingRule]:
        """Generate rules for a batch of examples with the same label."""
        # Prepare examples for analysis
        example_analysis = {
            "label": label,
            "examples": examples[:20],  # Limit for prompt size
            "pattern_hints": self._identify_patterns(examples),
            "common_indicators": self._extract_indicators(examples),
        }

        rendered_prompt = self.generation_template.render(
            label=label,
            examples=examples[:20],
            example_analysis=example_analysis,
            data_analysis=data_analysis,
        )

        # Generate initial rules using LLM
        try:
            response = self.llm.invoke(rendered_prompt)
            return response.ruleset.rules
        except Exception as e:
            logger.error(f"Error generating rules for label '{label}': {e}")
            # Fallback: create basic rule from patterns
            return [self._create_fallback_rule(label, examples)]

    def _consolidate_rules(self, rules: list[LabelingRule]) -> list[LabelingRule]:
        """Merge similar rules and remove duplicates."""
        # Group rules by label
        rules_by_label = defaultdict(list)
        for rule in rules:
            rules_by_label[rule.label].append(rule)

        consolidated = []
        for label, label_rules in rules_by_label.items():
            # Simple consolidation: merge rules with similar patterns
            unique_rules = []
            for rule in label_rules:
                # Check if similar rule already exists
                similar_found = False
                for existing in unique_rules:
                    if self._rules_are_similar(rule, existing):
                        # Merge the rules
                        existing.examples.extend(rule.examples)
                        existing.indicators.extend(rule.indicators)
                        existing.frequency += rule.frequency
                        existing.confidence = max(existing.confidence, rule.confidence)
                        similar_found = True
                        break

                if not similar_found:
                    unique_rules.append(rule)

            consolidated.extend(unique_rules)

        return consolidated

    def _generate_general_guidelines(self, data_analysis: dict[str, Any]) -> list[str]:
        """Generate general annotation guidelines based on data analysis."""
        guidelines = [
            f"This dataset contains {data_analysis['unique_labels']} distinct labels",
            f"Text length varies from {data_analysis['text_length_distribution']['min']} to {data_analysis['text_length_distribution']['max']} characters",
            "Consider the full context of each text before assigning a label",
            "When uncertain, refer to the specific rules and examples provided",
        ]

        # Add guidelines based on label distribution
        label_dist = data_analysis["label_distribution"]
        if len(label_dist) > 0:
            most_common = max(label_dist, key=label_dist.get)
            guidelines.append(f"The most common label is '{most_common}' ({label_dist[most_common]} examples)")

        return guidelines

    def _generate_disambiguation_rules(self, rules: list[LabelingRule]) -> list[str]:
        """Generate rules for handling ambiguous cases."""
        disambiguation = [
            "When a text could fit multiple categories, prioritize the most specific rule",
            "If indicators from multiple rules are present, choose the label with the strongest indicators",
            "Consider the primary intent or focus of the text when multiple aspects are present",
        ]

        # Add specific disambiguation based on rule conflicts
        label_indicators = defaultdict(set)
        for rule in rules:
            for indicator in rule.indicators:
                label_indicators[indicator.lower()].add(rule.label)

        # Find overlapping indicators
        conflicts = []
        for indicator, labels in label_indicators.items():
            if len(labels) > 1:
                conflicts.append(f"'{indicator}' appears in rules for: {', '.join(labels)}")

        if conflicts:
            disambiguation.append("Pay special attention to these overlapping indicators:")
            disambiguation.extend(conflicts)

        return disambiguation

    def _generate_quality_checks(self, data_analysis: dict[str, Any]) -> list[str]:
        """Generate quality assurance guidelines."""
        return [
            "Double-check labels for examples that seem to fall between categories",
            "Ensure consistency with similar examples you've already labeled",
            "If you're unsure about a label, mark it for review rather than guessing",
            f"Maintain the overall label distribution observed in training data: {data_analysis['label_distribution']}",
        ]

    def _calculate_ruleset_statistics(
        self,
        rules: list[LabelingRule],
        data_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate statistics about the generated ruleset."""
        return {
            "total_rules": len(rules),
            "rules_per_label": {
                label: len([r for r in rules if r.label == label])
                for label in data_analysis["label_distribution"].keys()
            },
            "avg_confidence": sum(rule.confidence for rule in rules) / len(rules) if rules else 0,
            "total_examples_covered": sum(rule.frequency for rule in rules),
            "avg_indicators_per_rule": sum(len(rule.indicators) for rule in rules) / len(rules) if rules else 0,
        }

    def _analyze_rule_coverage(
        self,
        rules: list[LabelingRule],
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> dict[str, Any]:
        """Analyze how well the rules cover the training data."""
        coverage_by_label = {}
        for label in df[label_column].unique():
            label_examples = df[df[label_column] == label]
            label_rules = [r for r in rules if r.label == str(label)]

            coverage_by_label[str(label)] = {
                "total_examples": len(label_examples),
                "rules_count": len(label_rules),
                "total_rule_examples": sum(r.frequency for r in label_rules),
                "coverage_ratio": sum(r.frequency for r in label_rules) / len(label_examples) if len(label_examples) > 0 else 0,
            }

        return {
            "coverage_by_label": coverage_by_label,
            "overall_coverage": sum(r.frequency for r in rules) / len(df) if len(df) > 0 else 0,
        }

    def _generate_recommendations(
        self,
        rules: list[LabelingRule],
        data_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations for improving the ruleset."""
        recommendations = []

        # Check for underrepresented labels
        for label, count in data_analysis["label_distribution"].items():
            label_rules = [r for r in rules if r.label == str(label)]
            if len(label_rules) < 2 and count > 10:
                recommendations.append(f"Consider gathering more examples for label '{label}' to create more specific rules")

        # Check for low confidence rules
        low_confidence_rules = [r for r in rules if r.confidence < 0.5]
        if low_confidence_rules:
            recommendations.append(f"Review {len(low_confidence_rules)} rules with low confidence scores")

        # Check for rules with few examples
        sparse_rules = [r for r in rules if r.frequency < 3]
        if sparse_rules:
            recommendations.append(f"Consider validating {len(sparse_rules)} rules that are based on few examples")

        return recommendations

    def _identify_patterns(self, examples: list[str]) -> list[str]:
        """Identify common patterns in a list of examples."""
        patterns = []

        # Simple pattern identification
        common_starts = self._find_common_prefixes(examples)
        common_ends = self._find_common_suffixes(examples)

        if common_starts:
            patterns.extend([f"Often starts with: {prefix}" for prefix in common_starts[:3]])
        if common_ends:
            patterns.extend([f"Often ends with: {suffix}" for suffix in common_ends[:3]])

        return patterns

    def _extract_indicators(self, examples: list[str]) -> list[str]:
        """Extract key indicators (words/phrases) from examples."""
        # Simple word frequency analysis
        all_words = []
        for example in examples:
            words = example.lower().split()
            all_words.extend(words)

        word_counts = Counter(all_words)
        # Return most common words (excluding very common ones)
        common_words = [word for word, count in word_counts.most_common(10)
                       if count > 1 and len(word) > 3]

        return common_words[:5]  # Limit to top 5

    def _extract_common_words(self, texts: list[str]) -> list[str]:
        """Extract most common words from a list of texts."""
        all_words = []
        for text in texts:
            words = str(text).lower().split()
            all_words.extend(words)

        word_counts = Counter(all_words)
        return [word for word, _ in word_counts.most_common(10)]

    def _find_common_prefixes(self, texts: list[str]) -> list[str]:
        """Find common prefixes in a list of texts."""
        if not texts:
            return []

        prefixes = []
        for length in range(1, min(20, min(len(t) for t in texts) + 1)):
            prefix_counts = Counter(text[:length].strip() for text in texts)
            common_prefixes = [prefix for prefix, count in prefix_counts.items()
                             if count > 1 and len(prefix.strip()) > 0]
            prefixes.extend(common_prefixes)

        return list(set(prefixes))[:5]

    def _find_common_suffixes(self, texts: list[str]) -> list[str]:
        """Find common suffixes in a list of texts."""
        if not texts:
            return []

        suffixes = []
        for length in range(1, min(20, min(len(t) for t in texts) + 1)):
            suffix_counts = Counter(text[-length:].strip() for text in texts)
            common_suffixes = [suffix for suffix, count in suffix_counts.items()
                             if count > 1 and len(suffix.strip()) > 0]
            suffixes.extend(common_suffixes)

        return list(set(suffixes))[:5]

    def _create_fallback_rule(self, label: str, examples: list[str]) -> LabelingRule:
        """Create a basic rule when LLM generation fails."""
        return LabelingRule(
            rule_id=f"fallback_{label}_{datetime.now().timestamp()}",
            label=label,
            pattern_description=f"Basic rule for label '{label}' based on example patterns",
            conditions=[f"Text exhibits patterns similar to {label} examples"],
            indicators=self._extract_indicators(examples),
            examples=examples[:5],
            confidence=0.3,  # Low confidence for fallback
            frequency=len(examples),
            creation_timestamp=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )

    def _rules_are_similar(self, rule1: LabelingRule, rule2: LabelingRule) -> bool:
        """Check if two rules are similar enough to merge."""
        if rule1.label != rule2.label:
            return False

        # Check indicator overlap
        indicators1 = set(rule1.indicators)
        indicators2 = set(rule2.indicators)
        overlap = len(indicators1.intersection(indicators2))
        total = len(indicators1.union(indicators2))

        return overlap / total > 0.5 if total > 0 else False

    def _hash_dataframe(self, df: pd.DataFrame, columns: list[str]) -> str:
        """Create a hash of specific DataFrame columns for tracking changes."""
        data_str = df[columns].to_string()
        return hashlib.md5(data_str.encode()).hexdigest()

    def _increment_version(self, version: str) -> str:
        """Increment version number (semantic versioning)."""
        try:
            major, minor, patch = version.split('.')
            return f"{major}.{minor}.{int(patch) + 1}"
        except (ValueError, AttributeError):
            return "1.0.1"

    def _prepare_examples_for_update(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> dict[str, list[str]]:
        """Prepare new examples grouped by label for update process."""
        examples_by_label = defaultdict(list)
        for _, row in df.iterrows():
            label = str(row[label_column])
            text = str(row[text_column])
            examples_by_label[label].append(text)

        return dict(examples_by_label)

    def _save_ruleset(self, ruleset: RuleSet) -> None:
        """Save ruleset to storage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_ruleset_{timestamp}.json"
        filepath = self.storage_path / filename

        with open(filepath, 'w') as f:
            json.dump(ruleset.model_dump(), f, indent=2)

        logger.info(f"Saved ruleset to {filepath}")

    def _export_markdown_guidelines(self, ruleset: RuleSet, output_path: Path) -> None:
        """Export ruleset as Markdown documentation."""
        md_content = f"""# Annotation Guidelines: {ruleset.dataset_name}

## Task Description
{ruleset.task_description}

## Label Categories
{', '.join(ruleset.label_categories)}

## General Guidelines
"""
        for guideline in ruleset.general_guidelines:
            md_content += f"- {guideline}\n"

        md_content += "\n## Labeling Rules\n\n"

        # Group rules by label
        rules_by_label = defaultdict(list)
        for rule in ruleset.rules:
            rules_by_label[rule.label].append(rule)

        for label, rules in rules_by_label.items():
            md_content += f"### {label}\n\n"
            for rule in rules:
                md_content += f"#### {rule.pattern_description}\n\n"
                md_content += f"**Confidence:** {rule.confidence:.2f}\n\n"

                if rule.conditions:
                    md_content += "**Conditions:**\n"
                    for condition in rule.conditions:
                        md_content += f"- {condition}\n"
                    md_content += "\n"

                if rule.indicators:
                    md_content += "**Key Indicators:**\n"
                    for indicator in rule.indicators:
                        md_content += f"- {indicator}\n"
                    md_content += "\n"

                if rule.examples:
                    md_content += "**Examples:**\n"
                    for example in rule.examples[:3]:  # Limit to 3 examples
                        md_content += f"> {example}\n\n"

        if ruleset.disambiguation_rules:
            md_content += "## Disambiguation Rules\n\n"
            for rule in ruleset.disambiguation_rules:
                md_content += f"- {rule}\n"

        if ruleset.quality_checks:
            md_content += "\n## Quality Checks\n\n"
            for check in ruleset.quality_checks:
                md_content += f"- {check}\n"

        with open(output_path, 'w') as f:
            f.write(md_content)

    def _export_html_guidelines(self, ruleset: RuleSet, output_path: Path) -> None:
        """Export ruleset as HTML documentation."""
        # This would generate a more formatted HTML version
        # For now, convert markdown to basic HTML
        from markdown import markdown

        md_path = output_path.with_suffix('.md')
        self._export_markdown_guidelines(ruleset, md_path)

        with open(md_path, 'r') as f:
            md_content = f.read()

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Annotation Guidelines: {ruleset.dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .rule {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background: #f9f9f9; }}
        .confidence {{ font-weight: bold; color: #007acc; }}
        blockquote {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-left: 3px solid #ccc; }}
    </style>
</head>
<body>
{markdown(md_content)}
</body>
</html>"""

        with open(output_path, 'w') as f:
            f.write(html_content)

        # Clean up temporary markdown file
        md_path.unlink()

    def _export_json_guidelines(self, ruleset: RuleSet, output_path: Path) -> None:
        """Export ruleset as JSON for programmatic use."""
        with open(output_path, 'w') as f:
            json.dump(ruleset.model_dump(), f, indent=2)
