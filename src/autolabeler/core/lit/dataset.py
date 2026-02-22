"""LIT Dataset wrapper for autolabeler pipeline outputs.

Wraps any labeled CSV produced by LabelingPipeline.label_dataframe() into a
LIT-compatible Dataset. The spec is built dynamically from the DatasetConfig
and the actual columns present in the CSV, so this works for any dataset --
not just fed_headlines.

Column classification logic:
    - text_column          -> TextSegment (the input to be analysed)
    - 'label'              -> CategoryLabel(vocab=config.labels) (gold label)
    - known numeric cols   -> Scalar (confidence, training_weight, etc.)
    - low-cardinality str  -> CategoryLabel() (tier, agreement, speaker, etc.)
    - timestamps / high-cardinality str -> excluded (too many values for LIT UI)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

try:
    from lit_nlp.api import dataset as lit_dataset
    from lit_nlp.api import types as lit_types
except ImportError as exc:
    raise ImportError(
        "lit-nlp is not installed. Install it with: pip install 'autolabeler[lit]'"
    ) from exc


# Columns whose values are always treated as Scalar regardless of dtype
_SCALAR_COLUMNS: frozenset[str] = frozenset(
    {'confidence', 'training_weight', 'suspicion_score'}
)

# Columns that are always excluded from the LIT spec (too noisy or redundant)
_EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {'soft_label', 'jury_labels', 'jury_confidences', 'reasoning'}
)

# Maximum unique values for a string column to be treated as CategoryLabel
_MAX_CATEGORY_VALUES: int = 200


class LabeledDataset(lit_dataset.Dataset):
    """LIT Dataset built from an autolabeler pipeline output CSV.

    Automatically maps all columns to appropriate LIT types, so every metadata
    column (tier, agreement, speaker, time period, etc.) becomes a sliceable
    facet in the LIT UI -- with no per-dataset configuration required.

    Args:
        data_path: Path to the labeled CSV (output from LabelingPipeline).
        config: DatasetConfig for this dataset.
        max_rows: Optional row limit for large datasets. Defaults to all rows.

    Example:
        >>> from autolabeler.core.dataset_config import DatasetConfig
        >>> from autolabeler.core.lit.dataset import LabeledDataset
        >>> config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
        >>> ds = LabeledDataset("outputs/fed_headlines/labeled.csv", config)
        >>> print(len(ds))
    """

    def __init__(
        self,
        data_path: str | Path,
        config: Any,  # DatasetConfig -- avoid circular import at module level
        max_rows: int | None = None,
    ) -> None:
        """Load labeled CSV and build the LIT spec.

        Args:
            data_path: Path to labeled CSV.
            config: DatasetConfig instance.
            max_rows: If set, only load this many rows (useful for large files).
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Labeled data not found: {data_path}")

        df = pd.read_csv(data_path)
        if max_rows is not None:
            df = df.head(max_rows)

        logger.info(
            f"Loaded {len(df)} rows from {data_path} "
            f"({len(df.columns)} columns)"
        )

        self._text_col: str = config.text_column
        self._label_vocab: list[str] = [str(l) for l in config.labels]
        self._spec: dict[str, lit_types.LitType] = self._build_spec(df, config)
        self._examples: list[dict[str, Any]] = self._build_examples(df)

        logger.info(
            f"LabeledDataset ready: {len(self._examples)} examples, "
            f"{len(self._spec)} spec fields"
        )

    def _build_spec(
        self,
        df: pd.DataFrame,
        config: Any,
    ) -> dict[str, lit_types.LitType]:
        """Construct the LIT spec from the DataFrame columns.

        Args:
            df: Loaded labeled DataFrame.
            config: DatasetConfig instance.

        Returns:
            Dict mapping column names to LIT types.
        """
        spec: dict[str, lit_types.LitType] = {}

        for col in df.columns:
            if col in _EXCLUDED_COLUMNS:
                continue

            if col == self._text_col:
                spec[col] = lit_types.TextSegment()
                continue

            if col == 'label':
                spec[col] = lit_types.CategoryLabel(vocab=self._label_vocab)
                continue

            if col in _SCALAR_COLUMNS:
                spec[col] = lit_types.Scalar()
                continue

            # Infer type from dtype + cardinality
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                spec[col] = lit_types.Scalar()
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= _MAX_CATEGORY_VALUES:
                    spec[col] = lit_types.CategoryLabel()
                else:
                    logger.debug(
                        f"Excluding high-cardinality column '{col}' "
                        f"({len(unique_vals)} unique values)"
                    )
            # datetime or other types: skip

        logger.debug(f"Built spec with fields: {list(spec.keys())}")
        return spec

    def _build_examples(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert the DataFrame rows to LIT example dicts.

        Args:
            df: Labeled DataFrame.

        Returns:
            List of dicts, one per row, with only spec-defined fields.
        """
        spec_cols = set(self._spec.keys())
        examples = []
        for _, row in df.iterrows():
            example: dict[str, Any] = {}
            for col in spec_cols:
                if col not in df.columns:
                    continue
                val = row[col]
                # Convert NaN to None so LIT handles it gracefully
                if pd.isna(val) if not isinstance(val, (list, dict)) else False:
                    example[col] = None
                else:
                    example[col] = str(val) if isinstance(self._spec[col], lit_types.CategoryLabel) else val
            examples.append(example)
        return examples

    def spec(self) -> dict[str, lit_types.LitType]:
        """Return the LIT spec for this dataset.

        Returns:
            Dict mapping field names to LIT types.
        """
        return self._spec
