from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def load_ruleset_for_prompt(
    ruleset_file: str | Path | None, rulesets_dir: Path = Path("rulesets")
) -> dict[str, Any] | None:
    """
    Load, validate, and format a ruleset from a JSON file for prompt injection.

    Args:
        ruleset_file: The path to the ruleset file.
        rulesets_dir: The base directory where rulesets are stored.

    Returns:
        A dictionary containing the formatted ruleset, or None if loading fails.
    """
    if not ruleset_file:
        return None

    try:
        ruleset_path = _resolve_path(ruleset_file, rulesets_dir)
        with open(ruleset_path, "r") as f:
            ruleset = json.load(f)

        if not _is_valid(ruleset):
            logger.error(f"Invalid ruleset file: {ruleset_path}")
            return None

        return _format_for_prompt(ruleset)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse ruleset {ruleset_file}: {e}")
        return None


def _resolve_path(file_path: str | Path, base_dir: Path) -> Path:
    """Resolve a file path, checking both absolute and relative paths."""
    path = Path(file_path)
    if path.is_absolute():
        return path

    relative_path = base_dir / path
    if relative_path.exists():
        return relative_path

    return path


def _is_valid(ruleset: dict[str, Any]) -> bool:
    """Validate the basic structure of a ruleset."""
    return "label_categories" in ruleset and "rules" in ruleset


def _format_for_prompt(ruleset: dict[str, Any]) -> dict[str, Any]:
    """Format the ruleset to be easily digestible by a Jinja2 template."""
    formatted_rules = [
        {
            "label": rule.get("label", ""),
            "description": rule.get("pattern_description", ""),
            "indicators": rule.get("indicators", []),
        }
        for rule in ruleset.get("rules", [])
    ]

    return {
        "task_description": ruleset.get("task_description", ""),
        "rules": sorted(formatted_rules, key=lambda x: str(x["label"])),
        "general_guidelines": ruleset.get("general_guidelines", []),
    }
