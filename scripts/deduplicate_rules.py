"""Utility script to deduplicate rules in existing rules files."""
import json
from pathlib import Path
from typing import Dict, List
from loguru import logger


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate word-based similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union


def deduplicate_rules(rules: List[str], similarity_threshold: float = 0.8) -> List[str]:
    """Remove duplicate and highly similar rules.

    Args:
        rules: List of rules to deduplicate
        similarity_threshold: Threshold for considering rules as duplicates

    Returns:
        Deduplicated list of rules
    """
    if not rules:
        return rules

    deduplicated = []
    seen_normalized = set()

    for rule in rules:
        rule_normalized = rule.lower().strip()

        # Skip exact duplicates
        if rule_normalized in seen_normalized:
            logger.debug(f"Skipping exact duplicate: {rule[:80]}...")
            continue

        # Check for partial duplicates (substring)
        is_duplicate = False
        for existing in deduplicated:
            existing_normalized = existing.lower().strip()

            # Check substring relationships
            if rule_normalized in existing_normalized:
                logger.debug(f"Skipping substring of existing rule: {rule[:80]}...")
                is_duplicate = True
                break
            elif existing_normalized in rule_normalized:
                # Keep the longer, more detailed version
                logger.debug(f"Replacing shorter rule with longer version")
                deduplicated.remove(existing)
                seen_normalized.discard(existing_normalized)
                break

            # Check similarity
            similarity = calculate_similarity(rule, existing)
            if similarity >= similarity_threshold:
                logger.debug(f"Skipping similar rule (similarity={similarity:.2f}): {rule[:80]}...")
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(rule)
            seen_normalized.add(rule_normalized)

    return deduplicated


def deduplicate_rules_file(input_file: Path, output_file: Path = None, dry_run: bool = False):
    """Deduplicate rules in a JSON rules file.

    Args:
        input_file: Path to input rules JSON file
        output_file: Path to output file (overwrites input if None)
        dry_run: If True, only report duplicates without modifying files
    """
    logger.info(f"Processing {input_file}")

    # Load rules
    with open(input_file) as f:
        rules_data = json.load(f)

    if not isinstance(rules_data, dict):
        logger.error(f"Expected dict in {input_file}, got {type(rules_data)}")
        return

    # Deduplicate each task's rules
    original_counts = {}
    deduplicated_data = {}

    for task_name, rules in rules_data.items():
        if not isinstance(rules, list):
            logger.warning(f"Task '{task_name}' rules are not a list, skipping")
            deduplicated_data[task_name] = rules
            continue

        original_counts[task_name] = len(rules)
        deduplicated_rules = deduplicate_rules(rules)
        deduplicated_data[task_name] = deduplicated_rules

        removed_count = len(rules) - len(deduplicated_rules)
        if removed_count > 0:
            logger.info(
                f"Task '{task_name}': {len(rules)} → {len(deduplicated_rules)} rules "
                f"(-{removed_count} duplicates)"
            )

    # Report summary
    total_original = sum(original_counts.values())
    total_deduplicated = sum(len(rules) for rules in deduplicated_data.values())
    total_removed = total_original - total_deduplicated

    logger.info("\n" + "=" * 80)
    logger.info("DEDUPLICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total rules before: {total_original}")
    logger.info(f"Total rules after: {total_deduplicated}")
    logger.info(f"Total duplicates removed: {total_removed}")
    logger.info(f"Reduction: {total_removed / total_original * 100:.1f}%")
    logger.info("=" * 80)

    # Save deduplicated rules
    if not dry_run:
        output_path = output_file or input_file
        with open(output_path, 'w') as f:
            json.dump(deduplicated_data, f, indent=2)
        logger.info(f"\nDeduplicated rules saved to: {output_path}")
    else:
        logger.info("\n[DRY RUN] No files were modified")


def main():
    """Find and deduplicate all rules files in outputs directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate rules in JSON files")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Specific rules file to process (processes all if not specified)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file (overwrites input if not specified)"
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Report duplicates without modifying files"
    )
    parser.add_argument(
        "--similarity-threshold",
        "-s",
        type=float,
        default=0.8,
        help="Similarity threshold for duplicate detection (0-1, default: 0.8)"
    )

    args = parser.parse_args()

    if args.input:
        # Process single file
        if not args.input.exists():
            logger.error(f"File not found: {args.input}")
            return

        deduplicate_rules_file(args.input, args.output, args.dry_run)
    else:
        # Find all rules files
        rules_files = list(Path("outputs").rglob("*rules*.json"))

        if not rules_files:
            logger.warning("No rules files found in outputs/ directory")
            return

        logger.info(f"Found {len(rules_files)} rules files to process\n")

        for rules_file in rules_files:
            deduplicate_rules_file(rules_file, dry_run=args.dry_run)
            print()


if __name__ == "__main__":
    main()
