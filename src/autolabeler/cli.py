from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd
from loguru import logger

from .config import Settings
from .ensemble import EnsembleLabeler
from .labeler import AutoLabeler
from .model_config import EnsembleMethod, ModelConfig
from .rule_generator import RuleGenerator
from .synthetic_generator import SyntheticDataGenerator


def load_json_config(config_path: Path) -> dict[str, Any]:
    """Load and validate JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        sys.exit(1)


def create_settings_from_config(config: dict[str, Any]) -> Settings:
    """Create Settings object from configuration dictionary."""
    # Extract base settings, use defaults for missing values
    settings_dict = config.get("settings", {})

    return Settings(
        openrouter_api_key=settings_dict.get("openrouter_api_key", ""),
        corporate_api_key=settings_dict.get("corporate_api_key", ""),
        corporate_base_url=settings_dict.get("corporate_base_url"),
        corporate_model=settings_dict.get("corporate_model", "gpt-4"),
        llm_model=settings_dict.get("llm_model", "openai/gpt-3.5-turbo"),
        embedding_model=settings_dict.get("embedding_model", "all-MiniLM-L6-v2"),
        max_examples_per_query=settings_dict.get("max_examples_per_query", 5),
        similarity_threshold=settings_dict.get("similarity_threshold", 0.8),
        knowledge_base_dir=Path(settings_dict.get("knowledge_base_dir", "knowledge_bases")),
    )


def create_model_configs_from_config(config: dict[str, Any]) -> list[ModelConfig]:
    """Create ModelConfig objects from configuration dictionary."""
    model_configs = []

    for model_config in config.get("models", []):
        model_config_obj = ModelConfig(
            model_name=model_config["model_name"],
            provider=model_config.get("provider", "openrouter"),
            temperature=model_config.get("temperature", 0.1),
            seed=model_config.get("seed"),
            max_tokens=model_config.get("max_tokens"),
            description=model_config.get("description", ""),
            tags=model_config.get("tags", []),
            custom_params=model_config.get("custom_params", {}),
        )
        model_configs.append(model_config_obj)

    return model_configs


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
def cli(log_level: str):
    """AutoLabeler CLI - Advanced text labeling with LLMs, RAG, and ensembles."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())


@cli.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--text-column', required=True, help='Name of column containing text to label')
@click.option('--dataset-name', required=True, help='Name of the dataset for knowledge base')
@click.option('--label-column', default='predicted_label', help='Name of column to store predictions')
@click.option('--use-rag/--no-rag', default=True, help='Whether to use RAG examples')
@click.option('--save-to-kb/--no-save-to-kb', default=True, help='Whether to save predictions to knowledge base')
@click.option('--confidence-threshold', type=float, default=0.0, help='Minimum confidence to save to KB')
def label(
    config_file: Path,
    input_file: Path,
    output_file: Path,
    text_column: str,
    dataset_name: str,
    label_column: str,
    use_rag: bool,
    save_to_kb: bool,
    confidence_threshold: float
):
    """Label text data using multiple model configurations."""
    config = load_json_config(config_file)
    settings = create_settings_from_config(config)
    model_configs = create_model_configs_from_config(config)

    if not model_configs:
        logger.error("No model configurations found in config file")
        sys.exit(1)

    # Load input data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Process with each model configuration
    all_results = []

    for i, model_config in enumerate(model_configs):
        logger.info(f"Processing with model config {i+1}/{len(model_configs)}: {model_config.description}")

        # Create labeler for this configuration
        labeler = AutoLabeler(dataset_name, settings)

        # Label the dataframe
        try:
            labeled_df = labeler.label_dataframe(
                df=df.copy(),
                text_column=text_column,
                label_column=f"{label_column}_{model_config.model_id}",
                use_rag=use_rag,
                save_to_knowledge_base=save_to_kb,
                confidence_threshold=confidence_threshold
            )

            # Add model configuration info
            labeled_df[f"{label_column}_{model_config.model_id}_model"] = model_config.model_name
            labeled_df[f"{label_column}_{model_config.model_id}_temp"] = model_config.temperature
            labeled_df[f"{label_column}_{model_config.model_id}_config_id"] = model_config.model_id

            all_results.append(labeled_df)

        except Exception as e:
            logger.error(f"Failed to process with model config {model_config.model_id}: {e}")
            continue

    if not all_results:
        logger.error("No successful labeling runs")
        sys.exit(1)

    # Merge all results
    final_df = all_results[0]
    for result_df in all_results[1:]:
        # Merge on original columns
        original_cols = df.columns.tolist()
        final_df = final_df.merge(
            result_df[original_cols + [col for col in result_df.columns if col not in original_cols]],
            on=original_cols,
            how='outer'
        )

    # Save results
    logger.info(f"Saving results to {output_file}")
    final_df.to_csv(output_file, index=False)

    # Print summary
    logger.info(f"Labeling complete. Processed {len(final_df)} rows with {len(model_configs)} model configurations.")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--text-column', required=True, help='Name of column containing text to label')
@click.option('--dataset-name', required=True, help='Name of the dataset for ensemble')
@click.option('--ensemble-method', default='majority_vote',
              type=click.Choice(['majority_vote', 'confidence_weighted', 'high_agreement']),
              help='Ensemble consolidation method')
@click.option('--save-individual/--no-save-individual', default=True,
              help='Whether to save individual model results')
def ensemble(
    config_file: Path,
    input_file: Path,
    output_file: Path,
    text_column: str,
    dataset_name: str,
    ensemble_method: str,
    save_individual: bool
):
    """Label text data using ensemble of multiple models."""
    config = load_json_config(config_file)
    settings = create_settings_from_config(config)
    model_configs = create_model_configs_from_config(config)

    if not model_configs:
        logger.error("No model configurations found in config file")
        sys.exit(1)

    # Load input data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Create ensemble labeler
    ensemble_labeler = EnsembleLabeler(dataset_name, settings)

    # Add model configurations
    model_ids = []
    for model_config in model_configs:
        model_id = ensemble_labeler.add_model_config(model_config)
        model_ids.append(model_id)

    # Create ensemble method
    if ensemble_method == "majority_vote":
        method = EnsembleMethod.majority_vote()
    elif ensemble_method == "confidence_weighted":
        method = EnsembleMethod.confidence_weighted()
    elif ensemble_method == "high_agreement":
        method = EnsembleMethod.high_agreement()
    else:
        method = EnsembleMethod.majority_vote()

    # Run ensemble labeling
    logger.info(f"Running ensemble labeling with {len(model_configs)} models")
    try:
        ensemble_df = ensemble_labeler.label_dataframe_ensemble(
            df=df,
            text_column=text_column,
            model_ids=model_ids,
            ensemble_method=method,
            save_individual_results=save_individual
        )

        # Save results
        logger.info(f"Saving ensemble results to {output_file}")
        ensemble_df.to_csv(output_file, index=False)

        # Print summary
        logger.info(f"Ensemble labeling complete. Processed {len(ensemble_df)} rows.")

        # Print ensemble summary
        summary = ensemble_labeler.get_ensemble_summary()
        logger.info(f"Ensemble summary: {summary}")

    except Exception as e:
        logger.error(f"Ensemble labeling failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.option('--dataset-name', required=True, help='Name of the dataset for synthetic generation')
@click.option('--target-label', required=True, help='Target label for synthetic examples')
@click.option('--num-examples', type=int, default=5, help='Number of synthetic examples to generate')
@click.option('--strategy', default='mixed',
              type=click.Choice(['paraphrase', 'interpolate', 'extrapolate', 'transform', 'mixed']),
              help='Generation strategy')
@click.option('--output-file', type=click.Path(path_type=Path), help='Output file for synthetic examples')
@click.option('--add-to-kb/--no-add-to-kb', default=True, help='Whether to add to knowledge base')
@click.option('--confidence-threshold', type=float, default=0.7, help='Minimum confidence to add to KB')
def generate(
    config_file: Path,
    dataset_name: str,
    target_label: str,
    num_examples: int,
    strategy: str,
    output_file: Path | None,
    add_to_kb: bool,
    confidence_threshold: float
):
    """Generate synthetic examples for a specific label."""
    config = load_json_config(config_file)
    settings = create_settings_from_config(config)
    model_configs = create_model_configs_from_config(config)

    if not model_configs:
        logger.error("No model configurations found in config file")
        sys.exit(1)

    # Generate with each model configuration
    all_synthetic_examples = []

    for i, model_config in enumerate(model_configs):
        logger.info(f"Generating with model config {i+1}/{len(model_configs)}: {model_config.description}")

        try:
            # Create synthetic generator for this configuration
            generator = SyntheticDataGenerator(dataset_name, settings)

            # Generate examples
            examples = generator.generate_examples_for_label(
                target_label=target_label,
                num_examples=num_examples,
                strategy=strategy,
                add_to_knowledge_base=add_to_kb,
                confidence_threshold=confidence_threshold
            )

            # Convert to DataFrame format
            for example in examples:
                example_dict = {
                    "text": example.text,
                    "label": example.label,
                    "confidence": example.confidence,
                    "reasoning": example.reasoning,
                    "strategy": strategy,
                    "model_config_id": model_config.model_id,
                    "model_name": model_config.model_name,
                    "temperature": model_config.temperature,
                }
                if example.generation_metadata:
                    for k, v in example.generation_metadata.items():
                        example_dict[f"meta_{k}"] = v

                all_synthetic_examples.append(example_dict)

        except Exception as e:
            logger.error(f"Failed to generate with model config {model_config.model_id}: {e}")
            continue

    if not all_synthetic_examples:
        logger.error("No successful synthetic generation runs")
        sys.exit(1)

    # Create DataFrame
    synthetic_df = pd.DataFrame(all_synthetic_examples)

    # Save to file if specified
    if output_file:
        logger.info(f"Saving synthetic examples to {output_file}")
        synthetic_df.to_csv(output_file, index=False)

    # Print summary
    logger.info(f"Generated {len(all_synthetic_examples)} synthetic examples for label '{target_label}'")
    logger.info(f"Average confidence: {synthetic_df['confidence'].mean():.3f}")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.option('--dataset-name', required=True, help='Name of the dataset for balancing')
@click.option('--target-balance', default='equal', help='Target balance: "equal" or JSON dict like {"pos":100,"neg":100}')
@click.option('--max-per-label', type=int, default=50, help='Maximum synthetic examples per label')
@click.option('--confidence-threshold', type=float, default=0.7, help='Minimum confidence threshold')
@click.option('--output-file', type=click.Path(path_type=Path), help='Output file for synthetic examples')
def balance(
    config_file: Path,
    dataset_name: str,
    target_balance: str,
    max_per_label: int,
    confidence_threshold: float,
    output_file: Path | None
):
    """Generate synthetic examples to balance dataset labels."""
    config = load_json_config(config_file)
    settings = create_settings_from_config(config)

    # Parse target balance
    if target_balance != "equal":
        try:
            target_balance = json.loads(target_balance)
        except json.JSONDecodeError:
            logger.error("Invalid target-balance format. Use 'equal' or JSON dict like '{\"pos\":100,\"neg\":100}'")
            sys.exit(1)

    # Create synthetic generator
    generator = SyntheticDataGenerator(dataset_name, settings)

    # Analyze current balance
    logger.info("Analyzing current label distribution...")
    analysis = generator.analyze_class_imbalance()
    logger.info(f"Current distribution: {analysis.get('distribution', {})}")

    # Generate balanced examples
    logger.info("Generating synthetic examples for balancing...")
    balanced_examples = generator.balance_dataset(
        target_balance=target_balance,
        max_synthetic_per_label=max_per_label,
        confidence_threshold=confidence_threshold
    )

    # Convert to DataFrame if output file specified
    if output_file:
        all_examples = []
        for label, examples in balanced_examples.items():
            for example in examples:
                example_dict = {
                    "text": example.text,
                    "label": example.label,
                    "confidence": example.confidence,
                    "reasoning": example.reasoning,
                    "target_label": label,
                }
                if example.generation_metadata:
                    for k, v in example.generation_metadata.items():
                        example_dict[f"meta_{k}"] = v
                all_examples.append(example_dict)

        if all_examples:
            df = pd.DataFrame(all_examples)
            logger.info(f"Saving balanced synthetic examples to {output_file}")
            df.to_csv(output_file, index=False)

    # Print summary
    total_generated = sum(len(examples) for examples in balanced_examples.values())
    logger.info(f"Generated {total_generated} synthetic examples for dataset balancing")

    for label, examples in balanced_examples.items():
        logger.info(f"  {label}: {len(examples)} examples")


@cli.command()
@click.argument('dataset_name')
@click.option('--config-file', type=click.Path(exists=True, path_type=Path),
              help='Configuration file for settings')
def stats(dataset_name: str, config_file: Path | None):
    """Show statistics for a dataset's knowledge base."""
    if config_file:
        config = load_json_config(config_file)
        settings = create_settings_from_config(config)
    else:
        settings = Settings()  # Use defaults

    # Create labeler to access knowledge base
    labeler = AutoLabeler(dataset_name, settings)

    # Get and display stats
    kb_stats = labeler.get_knowledge_base_stats()
    prompt_stats = labeler.get_prompt_analytics()

    click.echo(f"\n=== Knowledge Base Stats for '{dataset_name}' ===")
    click.echo(f"Total examples: {kb_stats.get('total_examples', 0)}")
    click.echo(f"Sources: {kb_stats.get('sources', {})}")
    click.echo(f"Label distribution: {kb_stats.get('label_distribution', {})}")
    click.echo(f"Last updated: {kb_stats.get('last_updated', 'Never')}")

    click.echo(f"\n=== Prompt Analytics ===")
    click.echo(f"Total prompts: {prompt_stats.get('total_prompts', 0)}")
    click.echo(f"Success rate: {prompt_stats.get('success_rate', 0):.2%}")
    click.echo(f"Average confidence: {prompt_stats.get('average_confidence', 0):.3f}")


@cli.command()
def create_config():
    """Create a sample configuration file."""
    sample_config = {
        "settings": {
            "openrouter_api_key": "your-openrouter-api-key",
            "corporate_api_key": "your-corporate-api-key",
            "corporate_base_url": "https://your-corporate-api.com/v1",
            "corporate_model": "gpt-4",
            "llm_model": "openai/gpt-3.5-turbo",
            "embedding_model": "all-MiniLM-L6-v2",
            "max_examples_per_query": 5,
            "similarity_threshold": 0.8,
            "knowledge_base_dir": "knowledge_bases"
        },
        "models": [
            {
                "model_name": "openai/gpt-3.5-turbo",
                "provider": "openrouter",
                "temperature": 0.1,
                "seed": 42,
                "description": "Conservative GPT-3.5",
                "tags": ["conservative", "low-temp"],
                "custom_params": {}
            },
            {
                "model_name": "openai/gpt-3.5-turbo",
                "provider": "openrouter",
                "temperature": 0.7,
                "seed": 42,
                "description": "Creative GPT-3.5",
                "tags": ["creative", "high-temp"],
                "custom_params": {}
            },
            {
                "model_name": "anthropic/claude-3-haiku",
                "provider": "openrouter",
                "temperature": 0.3,
                "seed": 123,
                "description": "Claude Haiku",
                "tags": ["claude", "mid-temp"],
                "custom_params": {}
            }
        ]
    }

    config_path = Path("autolabeler_config.json")
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    click.echo(f"Sample configuration created at: {config_path}")
    click.echo("Please edit the file to add your API keys and adjust model configurations.")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--text-column', required=True, help='Name of column containing text data')
@click.option('--label-column', required=True, help='Name of column containing labels')
@click.option('--dataset-name', required=True, help='Name of the dataset for rule generation')
@click.option('--task-description', help='Description of the labeling task')
@click.option('--batch-size', type=int, default=50, help='Number of examples to analyze per batch')
@click.option('--min-examples', type=int, default=3, help='Minimum examples needed to create a rule')
@click.option('--output-format', default='markdown',
              type=click.Choice(['markdown', 'html', 'json']),
              help='Output format for human-readable guidelines')
@click.option('--guidelines-file', type=click.Path(path_type=Path),
              help='Output file for human-readable annotation guidelines')
def generate_rules(
    config_file: Path,
    input_file: Path,
    text_column: str,
    label_column: str,
    dataset_name: str,
    task_description: str | None,
    batch_size: int,
    min_examples: int,
    output_format: str,
    guidelines_file: Path | None,
):
    """Generate labeling rules from labeled training data."""
    config = load_json_config(config_file)
    settings = create_settings_from_config(config)

    # Load input data
    logger.info(f"Loading labeled data from {input_file}")
    df = pd.read_csv(input_file)

    # Validate columns
    if text_column not in df.columns:
        logger.error(f"Text column '{text_column}' not found in data")
        sys.exit(1)
    if label_column not in df.columns:
        logger.error(f"Label column '{label_column}' not found in data")
        sys.exit(1)

    # Create rule generator
    rule_generator = RuleGenerator(dataset_name, settings)

    # Generate rules
    logger.info("Analyzing data and generating labeling rules...")
    result = rule_generator.generate_rules_from_data(
        df=df,
        text_column=text_column,
        label_column=label_column,
        task_description=task_description,
        batch_size=batch_size,
        min_examples_per_rule=min_examples,
    )

    # Print summary
    ruleset = result.ruleset
    logger.info(f"Generated {len(ruleset.rules)} rules for {len(ruleset.label_categories)} labels")

    click.echo(f"\n=== Rule Generation Summary ===")
    click.echo(f"Dataset: {ruleset.dataset_name}")
    click.echo(f"Task: {ruleset.task_description}")
    click.echo(f"Total rules: {len(ruleset.rules)}")
    click.echo(f"Label categories: {', '.join(ruleset.label_categories)}")

    # Show rules per label
    rules_per_label = {}
    for rule in ruleset.rules:
        rules_per_label[rule.label] = rules_per_label.get(rule.label, 0) + 1

    click.echo("\nRules per label:")
    for label, count in rules_per_label.items():
        click.echo(f"  {label}: {count} rules")

    # Export human-readable guidelines if requested
    if guidelines_file:
        rule_generator.export_ruleset_for_humans(
            ruleset, guidelines_file, format=output_format
        )
        click.echo(f"\nAnnotation guidelines exported to: {guidelines_file}")

    # Show recommendations
    if result.recommendations:
        click.echo("\nRecommendations:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")

    # Show coverage analysis
    if result.coverage_analysis:
        coverage = result.coverage_analysis
        click.echo(f"\nRule coverage: {coverage.get('overall_coverage', 0):.2%} of training data")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--text-column', required=True, help='Name of column containing text data')
@click.option('--label-column', required=True, help='Name of column containing labels')
@click.option('--dataset-name', required=True, help='Name of the dataset for rule updates')
@click.option('--output-format', default='markdown',
              type=click.Choice(['markdown', 'html', 'json']),
              help='Output format for updated guidelines')
@click.option('--guidelines-file', type=click.Path(path_type=Path),
              help='Output file for updated annotation guidelines')
def update_rules(
    config_file: Path,
    input_file: Path,
    text_column: str,
    label_column: str,
    dataset_name: str,
    output_format: str,
    guidelines_file: Path | None,
):
    """Update existing labeling rules with new training data."""
    config = load_json_config(config_file)
    settings = create_settings_from_config(config)

    # Load new training data
    logger.info(f"Loading new labeled data from {input_file}")
    df = pd.read_csv(input_file)

    # Validate columns
    if text_column not in df.columns:
        logger.error(f"Text column '{text_column}' not found in data")
        sys.exit(1)
    if label_column not in df.columns:
        logger.error(f"Label column '{label_column}' not found in data")
        sys.exit(1)

    # Create rule generator
    rule_generator = RuleGenerator(dataset_name, settings)

    # Check if existing ruleset exists
    try:
        existing_ruleset = rule_generator.load_latest_ruleset()
        logger.info(f"Found existing ruleset version {existing_ruleset.version}")
    except FileNotFoundError:
        logger.error(f"No existing ruleset found for dataset '{dataset_name}'")
        logger.error("Use 'generate-rules' command to create an initial ruleset first")
        sys.exit(1)

    # Update rules
    logger.info("Updating rules with new training data...")
    update_result = rule_generator.update_rules_with_new_data(
        new_df=df,
        text_column=text_column,
        label_column=label_column,
        existing_ruleset=existing_ruleset,
    )

    # Print update summary
    click.echo(f"\n=== Rule Update Summary ===")
    click.echo(f"Dataset: {update_result.updated_ruleset.dataset_name}")
    click.echo(f"Version: {existing_ruleset.version} -> {update_result.updated_ruleset.version}")
    click.echo(f"New rules added: {update_result.new_rules_added}")
    click.echo(f"Rules modified: {update_result.rules_modified}")
    click.echo(f"Rules removed: {update_result.rules_removed}")

    # Show changes made
    if update_result.changes_made:
        click.echo("\nChanges made:")
        for change in update_result.changes_made[:10]:  # Limit to first 10
            click.echo(f"  - {change}")
        if len(update_result.changes_made) > 10:
            click.echo(f"  ... and {len(update_result.changes_made) - 10} more changes")

    # Export updated guidelines if requested
    if guidelines_file:
        rule_generator.export_ruleset_for_humans(
            update_result.updated_ruleset, guidelines_file, format=output_format
        )
        click.echo(f"\nUpdated annotation guidelines exported to: {guidelines_file}")

    logger.info("Rule update completed successfully")


@cli.command()
@click.argument('dataset_name')
@click.option('--config-file', type=click.Path(exists=True, path_type=Path),
              help='Configuration file for settings')
@click.option('--output-format', default='markdown',
              type=click.Choice(['markdown', 'html', 'json']),
              help='Output format for guidelines')
@click.option('--output-file', type=click.Path(path_type=Path), required=True,
              help='Output file for annotation guidelines')
def export_rules(
    dataset_name: str,
    config_file: Path | None,
    output_format: str,
    output_file: Path,
):
    """Export the latest ruleset as human-readable annotation guidelines."""
    if config_file:
        config = load_json_config(config_file)
        settings = create_settings_from_config(config)
    else:
        settings = Settings()  # Use defaults

    # Create rule generator
    rule_generator = RuleGenerator(dataset_name, settings)

    # Load latest ruleset
    try:
        ruleset = rule_generator.load_latest_ruleset()
        logger.info(f"Found ruleset version {ruleset.version}")
    except FileNotFoundError:
        logger.error(f"No ruleset found for dataset '{dataset_name}'")
        logger.error("Use 'generate-rules' command to create a ruleset first")
        sys.exit(1)

    # Export guidelines
    rule_generator.export_ruleset_for_humans(
        ruleset, output_file, format=output_format
    )

    click.echo(f"Annotation guidelines exported to: {output_file}")
    click.echo(f"Format: {output_format}")
    click.echo(f"Ruleset version: {ruleset.version}")
    click.echo(f"Total rules: {len(ruleset.rules)}")


if __name__ == "__main__":
    cli()
