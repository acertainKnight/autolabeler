from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import click
import pandas as pd
from loguru import logger

from .autolabeler_v2 import AutoLabelerV2 as AutoLabeler
from .config import Settings
from .core.configs import (
    BatchConfig,
    DataSplitConfig,
    EnsembleConfig,
    EvaluationConfig,
    GenerationConfig,
    LabelingConfig,
    RuleGenerationConfig,
)

CONFIG_TYPE_MAP = {
    "labeling_config": LabelingConfig,
    "batch_config": BatchConfig,
    "split_config": DataSplitConfig,
    "eval_config": EvaluationConfig,
    "generation_config": GenerationConfig,
    "rule_config": RuleGenerationConfig,
    "ensemble_config": EnsembleConfig,
}

TASK_CONFIG_MAP = {
    "label": LabelingConfig,
    "split_data": DataSplitConfig,
    "evaluate": EvaluationConfig,
    "generate_synthetic": GenerationConfig,
    "generate_rules": RuleGenerationConfig,
    "label_ensemble": EnsembleConfig,
}


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    help="Set the logging level (e.g., DEBUG, INFO, WARNING).",
)
def cli(log_level: str):
    """
    AutoLabeler v2 CLI: A modular, configuration-driven tool for advanced
    data labeling, generation, and analysis.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())
    logger.info(f"Log level set to {log_level.upper()}")


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a JSON configuration file for the pipeline.",
)
def run(config_path: Path):
    """Run a pipeline of tasks from a configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    project_name = config.get("project_name")
    if not project_name:
        logger.error("Configuration must include a 'project_name'.")
        sys.exit(1)

    settings = Settings()
    labeler = AutoLabeler(project_name, settings)
    data_context: dict[str, Any] = {
        name: _load_data(Path(path))
        for name, path in config.get("data_context", {}).items()
    }

    task_dispatcher = {
        "split_data": labeler.split_data,
        "add_training_data": labeler.add_training_data,
        "label": labeler.label,
        "label_ensemble": labeler.label_ensemble,
        "evaluate": labeler.evaluate,
        "generate_rules": labeler.generate_rules,
        "save_data": lambda df, output_file: _save_data(df, Path(output_file)),
    }

    for i, task in enumerate(config.get("tasks", [])):
        task_name = task.get("name", f"Task {i+1}")
        task_type = task.get("type")
        params = task.get("params", {}).copy()
        logger.info(f"Running task '{task_name}' of type '{task_type}'")

        if task_type not in task_dispatcher:
            logger.error(f"Unknown task type '{task_type}' in task '{task_name}'.")
            continue

        # Prepare parameters by resolving keys from context
        resolved_params = {}
        for key, value in params.items():
            if key.endswith("_key"):
                if key == "df_key":
                    if value not in data_context:
                        logger.error(f"Data key '{value}' not found for task '{task_name}'.")
                        sys.exit(1)
                    resolved_params["df"] = data_context[value]
                elif key == "ruleset_key":
                    if value not in data_context:
                        logger.error(f"Ruleset key '{value}' not found for task '{task_name}'.")
                        sys.exit(1)
                    resolved_params["ruleset"] = data_context[value]
            elif key.endswith("_path"):
                if key == "ruleset_path":
                    from .core.utils import ruleset_utils
                    resolved_params["ruleset"] = ruleset_utils.load_ruleset_for_prompt(Path(value))
            elif key.endswith("_config"):
                config_model = _resolve_config(key, value)
                if config_model:
                    resolved_params[key] = config_model
            elif key == "config" and task_type in TASK_CONFIG_MAP:
                # Handle generic "config" key by mapping to task-specific config
                config_class = TASK_CONFIG_MAP[task_type]
                try:
                    config_model = config_class(**value)
                    # Map to the expected parameter name
                    if task_type == "label":
                        resolved_params["labeling_config"] = config_model
                    elif task_type == "split_data":
                        resolved_params["split_config"] = config_model
                    elif task_type == "evaluate":
                        resolved_params["eval_config"] = config_model
                    elif task_type == "generate_synthetic":
                        resolved_params["generation_config"] = config_model
                    elif task_type == "generate_rules":
                        resolved_params["rule_config"] = config_model
                    elif task_type == "label_ensemble":
                        resolved_params["ensemble_config"] = config_model
                except Exception as e:
                    logger.error(f"Failed to parse config for task '{task_type}': {e}")
            else:
                resolved_params[key] = value

        method = task_dispatcher[task_type]
        result = method(**resolved_params)

        # Store outputs for subsequent tasks
        if isinstance(result, pd.DataFrame):
            key = params.get("output_df_key")
            if key:
                data_context[key] = result
        elif isinstance(result, tuple) and all(
            isinstance(x, pd.DataFrame) for x in result
        ):
            if task_type == "split_data":
                data_context[params.get("output_train_df_key")] = result[0]
                data_context[params.get("output_test_df_key")] = result[1]
        elif isinstance(result, dict):
            key = params.get("output_ruleset_key")
            if key:
                data_context[key] = result
        elif result is not None:
            logger.warning(
                f"Task '{task_name}' returned an unsupported type: {type(result)}"
            )

    logger.info("Pipeline run complete.")


def _resolve_config(key: str, value: dict) -> Any:
    """Parse a config dictionary into its Pydantic model."""
    config_class = CONFIG_TYPE_MAP.get(key)
    if not config_class:
        logger.warning(f"No Pydantic model found for config key: {key}")
        return value
    if config_class == "NotImplemented":
        logger.warning(f"Parsing for '{key}' is not yet implemented.")
        return value
    try:
        return config_class(**value)
    except Exception as e:
        logger.error(f"Failed to parse config for {key}: {e}")
        return None


def _load_data(file_path: Path) -> pd.DataFrame:
    """Load data and add a persistent unique ID column if it's missing."""
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)
    logger.info(f"Loading data from {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    # Add a unique ID to each row for robust progress tracking if it doesn't exist
    if "autolabeler_id" not in df.columns:
        logger.info(
            f"'autolabeler_id' column not found in {file_path}. "
            f"Adding unique IDs and saving back to the original file."
        )
        df["autolabeler_id"] = [uuid.uuid4().hex for _ in range(len(df))]

        try:
            # Save the DataFrame with the new ID column back to the original file
            if file_path.suffix == ".parquet":
                df.to_parquet(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
            logger.info(f"Successfully saved data with unique IDs to {file_path}.")
        except Exception as e:
            logger.error(f"Failed to save data with unique IDs back to {file_path}: {e}")
            logger.warning("Continuing without persistent IDs. Progress may not be saved correctly across runs.")

    return df


def _save_data(df: pd.DataFrame, file_path: Path):
    """Save data to a CSV or Parquet file."""
    logger.info(f"Saving {len(df)} rows to {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)
    logger.info("Save complete.")


@cli.command()
@click.option(
    "--dataset-name", required=True, help="A unique name for your labeling project."
)
@click.option(
    "--input-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to the input CSV or Parquet file containing text data.",
)
@click.option(
    "--output-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to save the labeled output data.",
)
@click.option("--text-column", required=True, help="The name of the column with text to label.")
@click.option(
    "--train-file",
    type=click.Path(path_type=Path),
    help="Optional path to a file with pre-labeled training examples.",
)
@click.option(
    "--label-column",
    help="The name of the label column in the training data. Required if --train-file is used.",
)
@click.option(
    "--model-name", help="The LLM to use for labeling (e.g., 'gpt-3.5-turbo')."
)
@click.option(
    "--temperature",
    type=float,
    default=0.1,
    help="LLM temperature for sampling.",
)
@click.option(
    "--batch-size",
    type=int,
    default=50,
    show_default=True,
    help="Number of items to process in a concurrent batch.",
)
def label(
    dataset_name: str,
    input_file: Path,
    output_file: Path,
    text_column: str,
    train_file: Path | None,
    label_column: str | None,
    model_name: str | None,
    temperature: float,
    batch_size: int,
):
    """Label a dataset using a single model with RAG."""
    settings = Settings()
    if model_name:
        settings.llm_model = model_name
    if temperature:
        settings.temperature = temperature

    labeler = AutoLabeler(dataset_name, settings)

    if train_file:
        if not label_column:
            logger.error("--label-column is required when using --train-file.")
            sys.exit(1)
        train_df = _load_data(train_file)
        labeler.add_training_data(train_df, text_column, label_column)

    df = _load_data(input_file)
    labeling_config = LabelingConfig(
        use_rag=True if train_file else False,
        model_name=settings.llm_model,
        temperature=settings.temperature,
    )
    batch_config = BatchConfig(batch_size=batch_size, resume=True)

    labeled_df = labeler.label(
        df, text_column, labeling_config=labeling_config, batch_config=batch_config
    )
    _save_data(labeled_df, output_file)


@cli.command()
@click.option("--dataset-name", required=True, help="A unique name for your project.")
@click.option(
    "--input-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to the input file.",
)
@click.option(
    "--output-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to save the labeled output.",
)
@click.option("--text-column", required=True, help="Name of the text column.")
@click.option(
    "--train-file",
    type=click.Path(path_type=Path),
    help="Optional path to training data.",
)
@click.option(
    "--label-column",
    help="Name of label column. Required if --train-file is used.",
)
@click.option(
    "--models",
    required=True,
    help="Comma-separated list of models to use (e.g., 'gpt-4,claude-3').",
)
@click.option(
    "--method",
    default="majority_vote",
    type=click.Choice(["majority_vote", "confidence_weighted", "high_agreement"]),
    help="Ensemble consolidation method.",
)
@click.option(
    "--batch-size",
    type=int,
    default=20,
    show_default=True,
    help="Number of items to process in a concurrent batch.",
)
def ensemble(
    dataset_name: str,
    input_file: Path,
    output_file: Path,
    text_column: str,
    train_file: Path | None,
    label_column: str | None,
    models: str,
    method: str,
    batch_size: int,
):
    """Label a dataset using a multi-model ensemble."""
    settings = Settings()
    labeler = AutoLabeler(dataset_name, settings)
    model_list = [{'model_name': m.strip()} for m in models.split(",")]

    if train_file:
        if not label_column:
            logger.error("--label-column is required when using --train-file.")
            sys.exit(1)
        train_df = _load_data(train_file)
        labeler.add_training_data(train_df, text_column, label_column)

    df = _load_data(input_file)
    ensemble_config = EnsembleConfig(method=method)
    batch_config = BatchConfig(batch_size=batch_size, resume=True)

    labeled_df = labeler.label_ensemble(
        df,
        text_column,
        model_configs=model_list,
        ensemble_config=ensemble_config,
        batch_config=batch_config,
    )
    _save_data(labeled_df, output_file)


@cli.command()
@click.option("--dataset-name", required=True, help="A unique name for your project.")
@click.option(
    "--labeled-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to a file with labeled examples.",
)
@click.option("--text-column", required=True, help="Name of the text column.")
@click.option("--label-column", required=True, help="Name of the label column.")
@click.option(
    "--output-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to save the generated synthetic data.",
)
@click.option(
    "--strategy",
    default="balanced",
    help="Generation strategy: 'balanced' or a JSON distribution like '{\"pos\":100,\"neg\":50}'.",
)
@click.option(
    "--max-per-class",
    type=int,
    default=100,
    help="Maximum examples to generate per class for 'balanced' strategy.",
)
def generate(
    dataset_name: str,
    labeled_file: Path,
    text_column: str,
    label_column: str,
    output_file: Path,
    strategy: str,
    max_per_class: int,
):
    """Generate synthetic data to balance a dataset."""
    settings = Settings()
    labeler = AutoLabeler(dataset_name, settings)

    # Load labeled data to understand distribution
    labeled_df = _load_data(labeled_file)
    labeler.add_training_data(labeled_df, text_column, label_column)

    target_dist: Any
    if strategy == "balanced":
        # Calculate target distribution to balance all classes
        current_dist = labeled_df[label_column].value_counts().to_dict()
        if not current_dist:
            logger.error("Could not determine label distribution. Ensure data is labeled.")
        sys.exit(1)
        max_count = max(current_dist.values())
        target_dist = {
            label: max(0, min(max_per_class, max_count - count))
            for label, count in current_dist.items()
        }
        logger.info(f"Targeting generation distribution: {target_dist}")
    else:
        try:
            target_dist = json.loads(strategy)
        except json.JSONDecodeError:
            logger.error(
                "Invalid strategy format. Use 'balanced' or JSON like '{\"pos\":100}'."
            )
        sys.exit(1)

    config = GenerationConfig(add_to_knowledge_base=False)
    synthetic_df = labeler.generate_synthetic_data(
        target_distribution=target_dist, config=config
    )

    if not synthetic_df.empty:
        _save_data(synthetic_df, output_file)
    else:
        logger.info("No synthetic data was generated.")


@cli.command()
@click.option("--dataset-name", required=True, help="A unique name for your project.")
@click.option(
    "--labeled-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to a file with labeled examples.",
)
@click.option("--text-column", required=True, help="Name of the text column.")
@click.option("--label-column", required=True, help="Name of the label column.")
@click.option(
    "--output-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to save the generated rules (e.g., 'rules.md').",
)
@click.option(
    "--task-description",
    required=True,
    help="A clear description of the labeling task.",
)
@click.option(
    "--output-format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format for the rules.",
)
def rules(
    dataset_name: str,
    labeled_file: Path,
    text_column: str,
    label_column: str,
    output_file: Path,
    task_description: str,
    output_format: str,
):
    """Generate human-readable labeling rules from data."""
    settings = Settings()
    labeler = AutoLabeler(dataset_name, settings)

    labeled_df = _load_data(labeled_file)

    config = RuleGenerationConfig(
        task_description=task_description, export_format=output_format
    )

    # The service returns a list of rules, but the main interface can handle exporting.
    labeler.generate_rules(
        labeled_df,
        text_column,
        label_column,
        config=config,
        output_file=output_file,
    )
    logger.info(f"Rules saved to {output_file}")


@cli.command()
@click.option("--dataset-name", required=True, help="The project to evaluate.")
@click.option(
    "--predictions-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to a file with model predictions.",
)
@click.option(
    "--true-label-column", required=True, help="Column name for the ground truth labels."
)
@click.option(
    "--pred-label-column", required=True, help="Column name for the predicted labels."
)
@click.option(
    "--report-file",
    type=click.Path(path_type=Path),
    help="Optional path to save an HTML evaluation report.",
)
def evaluate(
    dataset_name: str,
    predictions_file: Path,
    true_label_column: str,
    pred_label_column: str,
    report_file: Path | None,
):
    """Evaluate prediction accuracy and generate a report."""
    settings = Settings()
    labeler = AutoLabeler(dataset_name, settings)

    df = _load_data(predictions_file)

    metrics = labeler.evaluate(
        df,
        true_label_column,
        pred_label_column,
        output_report_path=report_file,
    )

    logger.info("Evaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    cli()
