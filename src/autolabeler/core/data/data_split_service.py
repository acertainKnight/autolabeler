"""Service for managing data splits with leakage prevention."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from autolabeler.config import Settings

from ..base import ConfigurableComponent
from ..configs import DataSplitConfig


class DataSplitService(ConfigurableComponent):
    """
    Service for creating and managing train/test/validation splits.

    Provides functionality for creating reproducible splits with data leakage
    prevention and exclusion of specific datasets from training.

    Example:
        >>> config = DataSplitConfig(test_size=0.2, stratify_column="label")
        >>> service = DataSplitService("sentiment", settings)
        >>> train, test = service.create_split(df, config)
    """

    def __init__(self, dataset_name: str, settings: Settings, config: DataSplitConfig | None = None):
        """Initialize the data split service."""
        super().__init__(
            component_type="data_split",
            dataset_name=dataset_name,
            settings=settings,
            config=config or DataSplitConfig()
        )
        self.config = config or DataSplitConfig()
        self._split_cache = {}

    def create_split(
        self,
        df: pd.DataFrame,
        config: DataSplitConfig | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """
        Create train/test(/validation) split with data leakage prevention.

        Args:
            df: DataFrame to split
            config: Split configuration

        Returns:
            Tuple of (train_df, test_df, val_df) where val_df is None if no validation split
        """
        config = config or DataSplitConfig()

        # Create hash of data for caching
        data_hash = self._compute_data_hash(df)
        cache_key = f"{data_hash}_{config.model_dump_json()}"

        if cache_key in self._split_cache:
            logger.info("Using cached split")
            return self._split_cache[cache_key]

        # Filter out excluded data
        filtered_df = df.copy()
        for exclude_df in config.exclude_from_training:
            filtered_df = self._filter_exclude_data(filtered_df, exclude_df)

        logger.info(f"Creating split with {len(filtered_df)} rows (excluded {len(df) - len(filtered_df)})")

        # Determine stratification
        stratify = None
        if config.stratify_column and config.stratify_column in filtered_df.columns:
            stratify = filtered_df[config.stratify_column]

        # Create initial train/test split
        train_df, test_df = train_test_split(
            filtered_df,
            test_size=config.test_size,
            stratify=stratify,
            random_state=config.random_state,
        )

        # Create validation split if requested
        val_df = None
        if config.validation_size:
            # Adjust train size for validation split
            val_size_from_train = config.validation_size / (1 - config.test_size)

            train_df, val_df = train_test_split(
                train_df,
                test_size=val_size_from_train,
                stratify=train_df[config.stratify_column] if config.stratify_column else None,
                random_state=config.random_state,
            )

        # Cache the split
        self._split_cache[cache_key] = (train_df, test_df, val_df)

        # Log split statistics
        self._log_split_stats(train_df, test_df, val_df, config.stratify_column)

        return train_df, test_df, val_df

    def ensure_no_leakage(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ensure no data leakage between train and test sets.

        Removes any exact text matches that appear in both sets.

        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            text_column: Column containing text

        Returns:
            Tuple of (cleaned_train_df, cleaned_test_df)
        """
        # Find overlapping texts
        train_texts = set(train_df[text_column].str.strip())
        test_texts = set(test_df[text_column].str.strip())
        overlapping = train_texts & test_texts

        if overlapping:
            logger.warning(f"Found {len(overlapping)} overlapping texts between train/test")

            # Remove overlapping from train (keep in test for evaluation)
            train_mask = ~train_df[text_column].str.strip().isin(overlapping)
            train_df = train_df[train_mask].copy()

            logger.info(f"Removed {sum(~train_mask)} overlapping texts from training set")

        return train_df, test_df

    def create_k_fold_splits(
        self,
        df: pd.DataFrame,
        k: int = 5,
        stratify_column: str | None = None,
        random_state: int = 42,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create k-fold cross-validation splits.

        Args:
            df: DataFrame to split
            k: Number of folds
            stratify_column: Column to stratify on
            random_state: Random seed

        Returns:
            List of (train_df, val_df) tuples for each fold
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        if stratify_column:
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
            split_iter = kfold.split(df, df[stratify_column])
        else:
            kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
            split_iter = kfold.split(df)

        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            splits.append((train_df, val_df))

            logger.info(
                f"Fold {fold_idx + 1}/{k}: "
                f"train={len(train_df)}, val={len(val_df)}"
            )

        return splits

    def _filter_exclude_data(
        self,
        df: pd.DataFrame,
        exclude_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter out rows that appear in exclude_df."""
        # Use index-based filtering if possible
        if df.index.intersection(exclude_df.index).any():
            return df[~df.index.isin(exclude_df.index)].copy()

        # Otherwise, use content-based filtering
        # This is a simplified version - could be enhanced with fuzzy matching
        df_hash = df.apply(lambda x: hashlib.md5(str(x.values).encode()).hexdigest(), axis=1)
        exclude_hash = exclude_df.apply(lambda x: hashlib.md5(str(x.values).encode()).hexdigest(), axis=1)

        return df[~df_hash.isin(exclude_hash)].copy()

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the DataFrame for caching."""
        content = pd.util.hash_pandas_object(df).sum()
        return hashlib.md5(str(content).encode()).hexdigest()[:16]

    def _log_split_stats(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        stratify_column: str | None,
    ) -> None:
        """Log statistics about the data split."""
        total = len(train_df) + len(test_df) + (len(val_df) if val_df is not None else 0)

        logger.info(f"Data split complete:")
        logger.info(f"  Total: {total} rows")
        logger.info(f"  Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
        logger.info(f"  Test: {len(test_df)} ({len(test_df)/total*100:.1f}%)")

        if val_df is not None:
            logger.info(f"  Val: {len(val_df)} ({len(val_df)/total*100:.1f}%)")

        # Log stratification statistics if applicable
        if stratify_column:
            self._log_stratification_stats(train_df, test_df, val_df, stratify_column)

    def _log_stratification_stats(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        stratify_column: str,
    ) -> None:
        """Log stratification statistics."""
        train_dist = train_df[stratify_column].value_counts(normalize=True)
        test_dist = test_df[stratify_column].value_counts(normalize=True)

        logger.info(f"  Stratification on '{stratify_column}':")

        for label in train_dist.index:
            train_pct = train_dist.get(label, 0) * 100
            test_pct = test_dist.get(label, 0) * 100

            log_msg = f"    {label}: train={train_pct:.1f}%, test={test_pct:.1f}%"

            if val_df is not None:
                val_dist = val_df[stratify_column].value_counts(normalize=True)
                val_pct = val_dist.get(label, 0) * 100
                log_msg += f", val={val_pct:.1f}%"

            logger.info(log_msg)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about cached splits."""
        return {
            "cached_splits": len(self._split_cache),
            "dataset_name": self.dataset_name,
        }
