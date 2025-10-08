"""Constitutional AI service for rule enforcement and principle management."""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class ConstitutionalPrinciples(BaseModel):
    """Container for constitutional principles by task."""

    principles: dict[str, list[str]] = Field(default_factory=dict)
    enforcement_level: str = "strict"  # strict, moderate, lenient


class ConstitutionalService:
    """Manage and enforce labeling principles across classification tasks."""

    def __init__(
        self,
        principles: dict[str, list[str]] | None = None,
        enforcement_level: str = "strict",
    ):
        """Initialize constitutional service.

        Args:
            principles: Initial principles dictionary mapping task names to rule lists
            enforcement_level: How strictly to enforce rules (strict, moderate, lenient)
        """
        self.principles = ConstitutionalPrinciples(
            principles=principles or {},
            enforcement_level=enforcement_level,
        )
        logger.info(
            f"Constitutional service initialized with {len(self.principles.principles)} task principle sets"
        )

    def get_principles(self, task_name: str | None = None) -> dict[str, list[str]]:
        """Get principles for a specific task or all tasks.

        Args:
            task_name: Optional task name to filter by

        Returns:
            Dictionary of principles
        """
        if task_name:
            return {task_name: self.principles.principles.get(task_name, [])}
        return self.principles.principles

    def update_principles(self, new_principles: dict[str, list[str]]):
        """Update constitutional principles for one or more tasks.

        Args:
            new_principles: Dictionary mapping task names to updated principle lists
        """
        for task_name, principles in new_principles.items():
            self.principles.principles[task_name] = principles
            logger.info(
                f"Updated {len(principles)} principles for task '{task_name}'"
            )

    def add_principle(self, task_name: str, principle: str):
        """Add a single principle to a task.

        Args:
            task_name: Name of the task
            principle: Principle text to add
        """
        if task_name not in self.principles.principles:
            self.principles.principles[task_name] = []

        if principle not in self.principles.principles[task_name]:
            self.principles.principles[task_name].append(principle)
            logger.info(f"Added principle to task '{task_name}': {principle}")
        else:
            logger.debug(f"Principle already exists for task '{task_name}'")

    def remove_principle(self, task_name: str, principle: str):
        """Remove a principle from a task.

        Args:
            task_name: Name of the task
            principle: Principle text to remove
        """
        if task_name in self.principles.principles:
            if principle in self.principles.principles[task_name]:
                self.principles.principles[task_name].remove(principle)
                logger.info(f"Removed principle from task '{task_name}': {principle}")
            else:
                logger.warning(
                    f"Principle not found in task '{task_name}': {principle}"
                )
        else:
            logger.warning(f"Task '{task_name}' not found in principles")

    def validate_label(
        self, task_name: str, label: str, text: str, confidence: float
    ) -> dict[str, Any]:
        """Validate a label against constitutional principles.

        Args:
            task_name: Name of the classification task
            label: Predicted label
            text: Original text that was classified
            confidence: Confidence score for the prediction

        Returns:
            Validation result with is_valid flag and any warnings
        """
        result = {
            "is_valid": True,
            "warnings": [],
            "confidence_adjusted": confidence,
        }

        if task_name not in self.principles.principles:
            result["warnings"].append(f"No principles defined for task '{task_name}'")
            return result

        # Enforcement level logic
        if self.enforcement_level == "strict" and confidence < 0.9:
            result["warnings"].append(
                f"Strict mode: Low confidence {confidence:.2f} for task '{task_name}'"
            )
        elif self.enforcement_level == "moderate" and confidence < 0.7:
            result["warnings"].append(
                f"Moderate mode: Low confidence {confidence:.2f} for task '{task_name}'"
            )
        elif self.enforcement_level == "lenient" and confidence < 0.5:
            result["warnings"].append(
                f"Lenient mode: Low confidence {confidence:.2f} for task '{task_name}'"
            )

        return result

    def set_enforcement_level(self, level: str):
        """Set the enforcement level for constitutional principles.

        Args:
            level: Enforcement level (strict, moderate, lenient)
        """
        if level not in ["strict", "moderate", "lenient"]:
            logger.error(f"Invalid enforcement level: {level}")
            return

        self.principles.enforcement_level = level
        logger.info(f"Enforcement level set to: {level}")

    def get_principle_summary(self) -> dict[str, int]:
        """Get summary of principles per task.

        Returns:
            Dictionary mapping task names to principle counts
        """
        return {
            task_name: len(principles)
            for task_name, principles in self.principles.principles.items()
        }

    def export_principles(self) -> dict[str, Any]:
        """Export all principles to a dictionary.

        Returns:
            Dictionary with all principles and settings
        """
        return {
            "principles": self.principles.principles,
            "enforcement_level": self.principles.enforcement_level,
            "summary": self.get_principle_summary(),
        }

    def import_principles(self, data: dict[str, Any]):
        """Import principles from a dictionary.

        Args:
            data: Dictionary with principles and settings
        """
        if "principles" in data:
            self.principles.principles = data["principles"]
            logger.info(f"Imported {len(data['principles'])} task principle sets")

        if "enforcement_level" in data:
            self.set_enforcement_level(data["enforcement_level"])
