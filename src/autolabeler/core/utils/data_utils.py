from __future__ import annotations

import numpy as np
import pandas as pd


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
