"""
Constitutional AI for principled annotation consistency.

This module implements Constitutional AI with critique-revise workflows
to enforce annotation principles and reduce biases systematically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from ...config import Settings
from ..llm_providers import get_llm_client


class ConstitutionalPrinciple(BaseModel):
    """A single constitutional principle for annotations."""

    name: str = Field(description="Principle name")
    description: str = Field(description="Principle description")
    constraints: list[str] = Field(
        default_factory=list, description="Specific constraints"
    )
    severity: str = Field(
        default="medium", description="Severity level (critical, high, medium, low)"
    )
    examples: list[dict[str, Any]] = Field(
        default_factory=list, description="Example violations and corrections"
    )


class ConstitutionalConfig(BaseModel):
    """Configuration for Constitutional AI."""

    constitution_path: str = Field(description="Path to constitution JSON file")
    settings: Settings = Field(description="Application settings")
    llm_config: dict[str, Any] = Field(
        default_factory=dict, description="LLM configuration"
    )
    max_revisions: int = Field(
        2, gt=0, description="Maximum revision iterations"
    )
    critique_temperature: float = Field(
        0.2, ge=0, le=2, description="Temperature for critique generation"
    )
    revision_temperature: float = Field(
        0.3, ge=0, le=2, description="Temperature for revision generation"
    )
    require_unanimous_compliance: bool = Field(
        False, description="Require all principles to pass"
    )


class ConstitutionalAI:
    """
    Constitutional AI for principled annotation consistency.

    Implements a critique-revise workflow where initial annotations are
    evaluated against constitutional principles and revised if violations
    are detected.

    Example:
        >>> config = ConstitutionalConfig(
        ...     constitution_path="config/constitution.json",
        ...     settings=settings,
        ...     llm_config={"model": "gpt-4"}
        ... )
        >>> cai = ConstitutionalAI(config)
        >>> result = cai.annotate_with_constitution(
        ...     text="Sample text",
        ...     initial_annotation={"label": "positive", "confidence": 0.9}
        ... )
    """

    def __init__(self, config: ConstitutionalConfig):
        """
        Initialize Constitutional AI system.

        Args:
            config: Constitutional AI configuration.
        """
        self.config = config
        self.constitution = self._load_constitution(config.constitution_path)
        self.llm_client = get_llm_client(config.settings, config.llm_config)

        logger.info(
            f"Initialized Constitutional AI with {len(self.constitution)} principles"
        )

    def _load_constitution(self, path: str) -> list[ConstitutionalPrinciple]:
        """
        Load annotation constitution from file.

        Args:
            path: Path to constitution JSON file.

        Returns:
            List of constitutional principles.
        """
        constitution_path = Path(path)

        if not constitution_path.exists():
            logger.warning(f"Constitution file not found: {path}")
            return []

        try:
            with open(constitution_path) as f:
                constitution_data = json.load(f)

            principles = [
                ConstitutionalPrinciple(**principle)
                for principle in constitution_data.get("principles", [])
            ]

            logger.info(f"Loaded {len(principles)} principles from {path}")
            return principles

        except Exception as e:
            logger.error(f"Failed to load constitution: {e}")
            return []

    def annotate_with_constitution(
        self,
        text: str,
        initial_annotation: dict[str, Any],
        max_revisions: int | None = None,
    ) -> dict[str, Any]:
        """
        Annotate with constitutional critique-revise workflow.

        Args:
            text: Text to annotate.
            initial_annotation: Initial annotation result.
            max_revisions: Maximum revisions (uses config if not provided).

        Returns:
            Final annotation with constitutional metadata.
        """
        max_revisions = max_revisions or self.config.max_revisions
        current_annotation = initial_annotation.copy()
        all_critiques = []
        revision_count = 0

        for iteration in range(max_revisions):
            # Critique against each principle
            critiques = []
            for principle in self.constitution:
                critique = self._critique_annotation(
                    text, current_annotation, principle
                )
                critiques.append(critique)

            all_critiques.append(
                {"iteration": iteration, "critiques": critiques.copy()}
            )

            # Check if revision needed
            violations = [c for c in critiques if c["violates_principle"]]

            if not violations:
                # No violations, annotation is acceptable
                break

            # Revise based on critiques
            logger.info(
                f"Revision {iteration + 1}: Found {len(violations)} violations"
            )
            current_annotation = self._revise_annotation(text, current_annotation, critiques)
            revision_count += 1

        # Final validation
        final_critiques = [
            self._critique_annotation(text, current_annotation, principle)
            for principle in self.constitution
        ]

        final_violations = [c for c in final_critiques if c["violates_principle"]]

        return {
            "label": current_annotation.get("label"),
            "confidence": current_annotation.get("confidence", 0.0),
            "reasoning": current_annotation.get("reasoning", ""),
            "constitutional_metadata": {
                "num_revisions": revision_count,
                "critique_history": all_critiques,
                "final_validation": final_critiques,
                "compliant": len(final_violations) == 0,
                "violations": [v["principle_name"] for v in final_violations],
            },
        }

    def _critique_annotation(
        self,
        text: str,
        annotation: dict[str, Any],
        principle: ConstitutionalPrinciple,
    ) -> dict[str, Any]:
        """
        Critique annotation against a constitutional principle.

        Args:
            text: Original text.
            annotation: Current annotation.
            principle: Principle to check against.

        Returns:
            Critique results dictionary.
        """
        critique_prompt = self._format_critique_prompt(text, annotation, principle)

        try:
            response = self.llm_client.invoke(
                critique_prompt,
                temperature=self.config.critique_temperature,
            )

            # Parse response
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Simple parsing - look for "yes" or "violation" in response
            violates = any(
                keyword in response_text.lower()[:100]
                for keyword in ["yes", "violation", "violates"]
            )

            return {
                "principle_name": principle.name,
                "principle_severity": principle.severity,
                "violates_principle": violates,
                "explanation": response_text,
                "suggested_revision": self._extract_suggestion(response_text),
            }

        except Exception as e:
            logger.error(f"Critique failed for principle {principle.name}: {e}")
            return {
                "principle_name": principle.name,
                "principle_severity": principle.severity,
                "violates_principle": False,
                "explanation": f"Critique failed: {e}",
                "suggested_revision": None,
            }

    def _revise_annotation(
        self,
        text: str,
        annotation: dict[str, Any],
        critiques: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Revise annotation based on constitutional critiques.

        Args:
            text: Original text.
            annotation: Current annotation.
            critiques: List of critique results.

        Returns:
            Revised annotation.
        """
        revision_prompt = self._format_revision_prompt(text, annotation, critiques)

        try:
            response = self.llm_client.invoke(
                revision_prompt,
                temperature=self.config.revision_temperature,
            )

            # Parse revised annotation
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Attempt to extract structured annotation
            revised = self._parse_revised_annotation(response_text, annotation)

            # Track which principles were addressed
            revised["revisions_applied"] = [
                c["principle_name"] for c in critiques if c["violates_principle"]
            ]

            return revised

        except Exception as e:
            logger.error(f"Revision failed: {e}")
            # Return original if revision fails
            return annotation

    def _format_critique_prompt(
        self,
        text: str,
        annotation: dict[str, Any],
        principle: ConstitutionalPrinciple,
    ) -> str:
        """Format critique prompt."""
        constraints_text = "\n".join(f"- {c}" for c in principle.constraints)

        examples_text = ""
        if principle.examples:
            examples_text = "\n\nExamples of violations:\n"
            for ex in principle.examples[:2]:  # Limit to 2 examples
                examples_text += f"- Violation: {ex.get('violation', 'N/A')}\n"
                examples_text += f"  Correction: {ex.get('correction', 'N/A')}\n"

        return f"""You are evaluating an annotation against a constitutional principle.

Text: {text}

Current Annotation:
- Label: {annotation.get('label', 'N/A')}
- Confidence: {annotation.get('confidence', 'N/A')}
- Reasoning: {annotation.get('reasoning', 'N/A')}

Constitutional Principle: {principle.name}
Severity: {principle.severity}

Description: {principle.description}

Constraints:
{constraints_text}
{examples_text}

Does this annotation violate the principle? Respond with:
1. Yes or No
2. Brief explanation (2-3 sentences)
3. Suggested revision if violation detected

Response:"""

    def _format_revision_prompt(
        self,
        text: str,
        annotation: dict[str, Any],
        critiques: list[dict[str, Any]],
    ) -> str:
        """Format revision prompt."""
        violations = [c for c in critiques if c["violates_principle"]]

        critiques_text = ""
        for critique in violations:
            critiques_text += f"\n- {critique['principle_name']} ({critique['principle_severity']} severity)\n"
            critiques_text += f"  Issue: {critique['explanation'][:200]}\n"
            if critique.get("suggested_revision"):
                critiques_text += f"  Suggestion: {critique['suggested_revision']}\n"

        return f"""Revise the annotation to comply with constitutional principles.

Text: {text}

Original Annotation:
- Label: {annotation.get('label', 'N/A')}
- Confidence: {annotation.get('confidence', 'N/A')}
- Reasoning: {annotation.get('reasoning', 'N/A')}

Principle Violations:
{critiques_text}

Provide a revised annotation that addresses ALL violations while maintaining accuracy.

Format your response as:
Label: <revised label>
Confidence: <confidence score 0-1>
Reasoning: <updated reasoning>

Revised Annotation:"""

    def _parse_revised_annotation(
        self, response_text: str, fallback: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Parse revised annotation from response text.

        Args:
            response_text: Response text from LLM.
            fallback: Fallback annotation if parsing fails.

        Returns:
            Parsed annotation dictionary.
        """
        try:
            # Simple line-based parsing
            lines = response_text.strip().split("\n")
            result = fallback.copy()

            for line in lines:
                if line.lower().startswith("label:"):
                    result["label"] = line.split(":", 1)[1].strip()
                elif line.lower().startswith("confidence:"):
                    conf_str = line.split(":", 1)[1].strip()
                    try:
                        result["confidence"] = float(conf_str)
                    except ValueError:
                        pass
                elif line.lower().startswith("reasoning:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()

            return result

        except Exception as e:
            logger.warning(f"Failed to parse revised annotation: {e}")
            return fallback

    def _extract_suggestion(self, response_text: str) -> str | None:
        """Extract suggested revision from critique response."""
        # Look for common suggestion patterns
        suggestion_markers = [
            "suggested revision:",
            "suggestion:",
            "recommend:",
            "should be:",
        ]

        lower_text = response_text.lower()

        for marker in suggestion_markers:
            if marker in lower_text:
                idx = lower_text.index(marker) + len(marker)
                # Extract next 200 characters
                suggestion = response_text[idx : idx + 200].strip()
                # Clean up
                suggestion = suggestion.split("\n")[0]
                return suggestion

        return None

    def validate_compliance(
        self, text: str, annotation: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate annotation compliance without revision.

        Args:
            text: Text being annotated.
            annotation: Annotation to validate.

        Returns:
            Validation results.
        """
        critiques = [
            self._critique_annotation(text, annotation, principle)
            for principle in self.constitution
        ]

        violations = [c for c in critiques if c["violates_principle"]]

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "total_principles": len(self.constitution),
            "passed_principles": len(self.constitution) - len(violations),
        }

    def create_constitution_template(self, output_path: str) -> None:
        """
        Create a template constitution file.

        Args:
            output_path: Path for output JSON file.
        """
        template = {
            "constitution_name": "Annotation Principles",
            "version": "1.0",
            "description": "Constitutional principles for fair and consistent annotation",
            "principles": [
                {
                    "name": "No Demographic Bias",
                    "description": "Annotations must not be influenced by demographic attributes",
                    "constraints": [
                        "Do not use gender, race, age, or nationality as classification factors",
                        "Evaluate content neutrally without stereotyping",
                    ],
                    "severity": "critical",
                    "examples": [
                        {
                            "violation": "Labeled as negative due to perceived ethnicity",
                            "correction": "Focus only on content, not demographics",
                        }
                    ],
                },
                {
                    "name": "Evidence-Based Reasoning",
                    "description": "All labels must be supported by explicit evidence in the text",
                    "constraints": [
                        "Do not infer information not present in the text",
                        "Cite specific phrases or passages supporting the label",
                    ],
                    "severity": "high",
                    "examples": [],
                },
                {
                    "name": "Consistency with Guidelines",
                    "description": "Follow established annotation guidelines precisely",
                    "constraints": [
                        "Use only predefined label categories",
                        "Apply edge case rules as specified",
                    ],
                    "severity": "high",
                    "examples": [],
                },
            ],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(template, f, indent=2)

        logger.info(f"Created constitution template at {output_path}")
