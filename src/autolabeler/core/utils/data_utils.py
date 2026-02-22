from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_FORMAT_READERS: dict[str, callable] = {
    "csv": pd.read_csv,
    "jsonl": lambda p: pd.read_json(p, lines=True),
}

_FORMAT_GLOBS: dict[str, list[str]] = {
    "csv": ["*.csv"],
    "jsonl": ["*.jsonl", "*.ndjson"],
}


def _read_single_file(path: Path, input_format: str) -> pd.DataFrame:
    """Read a single data file into a DataFrame.

    Args:
        path: Path to the file.
        input_format: Expected format ("csv" or "jsonl").

    Returns:
        pd.DataFrame loaded from the file.

    Raises:
        ValueError: If the format is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    reader = _FORMAT_READERS.get(input_format)
    if reader is None:
        raise ValueError(
            f"Unsupported input_format '{input_format}'. "
            f"Supported: {list(_FORMAT_READERS)}"
        )

    return reader(path)


def load_input_data(
    path: str | Path,
    input_format: str = "csv",
    text_column: str | None = None,
) -> pd.DataFrame:
    """Load data from a single file or a directory of files.

    When *path* is a directory every file matching the ``input_format``
    extension is loaded and concatenated into one DataFrame.  A
    ``_source_file`` column is added so downstream code can trace each
    row back to its origin file.

    Args:
        path: File path or directory containing data files.
        input_format: File format — ``"csv"`` or ``"jsonl"``.
        text_column: If provided, validates that this column exists in
            every loaded file.

    Returns:
        A single concatenated DataFrame with an additional ``_source_file``
        column when multiple files are loaded.

    Raises:
        FileNotFoundError: If the path does not exist or the directory
            contains no matching files.
        ValueError: If ``text_column`` is missing from any file.

    Example:
        >>> df = load_input_data("datasets/headlines/", input_format="jsonl")
        >>> df._source_file.unique()
        array(['split_01.jsonl', 'split_02.jsonl'], dtype=object)
    """
    path = Path(path)

    if path.is_dir():
        globs = _FORMAT_GLOBS.get(input_format, [f"*.{input_format}"])
        files = sorted(
            f for pattern in globs for f in path.glob(pattern)
        )
        if not files:
            raise FileNotFoundError(
                f"No {input_format} files found in directory: {path}"
            )

        frames: list[pd.DataFrame] = []
        for file_path in files:
            df = _read_single_file(file_path, input_format)
            df["_source_file"] = file_path.name
            if text_column and text_column not in df.columns:
                raise ValueError(
                    f"Text column '{text_column}' not found in {file_path.name}. "
                    f"Available columns: {df.columns.tolist()}"
                )
            frames.append(df)
            logger.debug(f"  Loaded {len(df)} rows from {file_path.name}")

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            f"Loaded {len(combined)} total rows from {len(files)} "
            f"{input_format} files in {path}"
        )
        return combined

    df = _read_single_file(path, input_format)
    df["_source_file"] = path.name
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def filter_exclude_data(
    df: pd.DataFrame,
    exclude_df: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    """
    Remove rows from a DataFrame that exist in an exclusion DataFrame.

    Matching is performed based on the content of the `text_column`.

    Args:
        df: The DataFrame to filter.
        exclude_df: DataFrame containing data to exclude.
        text_column: The name of the column to match on.

    Returns:
        A filtered DataFrame with excluded rows removed.
    """
    if exclude_df.empty or text_column not in df.columns or text_column not in exclude_df.columns:
        return df

    exclude_texts = set(exclude_df[text_column].astype(str))
    return df[~df[text_column].astype(str).isin(exclude_texts)]


class ExampleSelector:
    """
    Selects examples for RAG with optional noise injection for variability.

    This helps in selecting a more diverse set of examples for few-shot prompting.
    """

    def __init__(self, noise_factor: float = 0.0, seed: int = 42):
        self.noise_factor = noise_factor
        self.rng = np.random.RandomState(seed)

    def select_with_noise(
        self, similarity_scores: list[float], examples: list[dict], k: int
    ) -> list[dict]:
        """
        Add noise to similarity scores to select a more diverse set of examples.

        Args:
            similarity_scores: A list of similarity scores (higher is better).
            examples: The list of candidate examples corresponding to the scores.
            k: The number of examples to select.

        Returns:
            A list of k selected examples.
        """
        if self.noise_factor == 0.0 or len(similarity_scores) <= k:
            # No noise or not enough examples, return top k
            top_indices = np.argsort(similarity_scores)[-k:]
            return [examples[i] for i in top_indices]

        scores = np.array(similarity_scores)
        noise = self.rng.normal(0, self.noise_factor * np.std(scores), len(scores))
        noisy_scores = scores + noise

        selected_indices = np.argsort(noisy_scores)[-k:]
        return [examples[i] for i in selected_indices]
