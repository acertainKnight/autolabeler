"""Composite suspicion scoring: combines all diagnostic signals into a single score.

Each sample receives a suspicion score in [0, 1] that reflects how likely it
is to contain a labeling error. Higher = more suspicious = higher review priority.

Signals combined:
- embedding_outlier: z-score from intra-class centroid distance
- nli_mismatch: NLI entailment gap (1 - entailment for assigned label)
- low_confidence: (1 - mean_jury_confidence)
- jury_disagreement: fraction of jury members who disagreed with consensus
- rationale_inconsistency: rationale cluster mismatch margin (inverted)

Weights are configured in DiagnosticsConfig.suspicion_weights and must sum to ~1.0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .config import DiagnosticsConfig


class SuspicionScorer:
    """Combine diagnostic signals into a per-sample suspicion score.

    Takes the output DataFrames from the other diagnostic modules and merges
    them into a single ranked table. Scores are normalized to [0, 1].

    Args:
        config: DiagnosticsConfig with suspicion_weights.

    Example:
        >>> scorer = SuspicionScorer(config)
        >>> scores = scorer.compute_scores(signals_df, weights)
        >>> top_suspects = scores.nlargest(50, "suspicion_score")
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize with diagnostics configuration.

        Args:
            config: DiagnosticsConfig instance.
        """
        self.config = config

    def build_signals_dataframe(
        self,
        n_samples: int,
        diagnostic_results: dict[str, Any],
    ) -> pd.DataFrame:
        """Assemble per-sample signal values from all diagnostic results.

        Extracts scalar signal values for each sample index and stores them
        in a wide DataFrame ready for weighted combination.

        Args:
            n_samples: Total number of labeled samples.
            diagnostic_results: Dict keyed by module name containing the raw
                diagnostic output dicts (from run_all() on each module).

        Returns:
            DataFrame indexed 0..n_samples-1 with one column per signal:
                - embedding_outlier: z-score (higher = more anomalous)
                - nli_mismatch: 1 - entailment_score (higher = worse coherence)
                - low_confidence: 1 - mean_confidence
                - jury_disagreement: fraction who disagreed with consensus
                - rationale_inconsistency: 1 - rationale_sim (higher = more divergent)

        Example:
            >>> signals = scorer.build_signals_dataframe(len(df), results)
        """
        signals = pd.DataFrame(
            {
                'embedding_outlier': 0.0,
                'nli_mismatch': 0.0,
                'low_confidence': 0.0,
                'jury_disagreement': 0.0,
                'rationale_inconsistency': 0.0,
            },
            index=range(n_samples),
        )

        # --- Embedding outlier signal ---
        emb_results = diagnostic_results.get('embedding', {})
        centroid_df = emb_results.get('centroid_violations')
        if centroid_df is not None and not centroid_df.empty and 'index' in centroid_df.columns:
            # Clip margin to [min_margin, 0], then scale to [0, 1]
            # More negative margin = higher suspicion
            min_margin = centroid_df['margin'].min()
            max_margin = centroid_df['margin'].max()
            if min_margin < max_margin:
                norm_margin = (max_margin - centroid_df['margin']) / (max_margin - min_margin)
            else:
                norm_margin = centroid_df['margin'].apply(lambda x: 1.0 if x < 0 else 0.0)
            for i, idx in enumerate(centroid_df['index']):
                if idx < n_samples:
                    signals.at[idx, 'embedding_outlier'] = float(norm_margin.iloc[i])

        # --- NLI mismatch signal ---
        nli_results = diagnostic_results.get('nli', {})
        entailment_df = nli_results.get('entailment')
        if entailment_df is not None and not entailment_df.empty and 'index' in entailment_df.columns:
            for _, row in entailment_df.iterrows():
                idx = int(row['index'])
                if idx < n_samples:
                    signals.at[idx, 'nli_mismatch'] = float(
                        1.0 - max(0.0, min(1.0, row.get('entailment_score', 0.5)))
                    )

        # --- Low confidence signal (from labeled DataFrame) ---
        labeled_df = diagnostic_results.get('labeled_df')
        if labeled_df is not None and 'jury_confidences' in labeled_df.columns:
            import json

            def mean_conf(val: Any) -> float:
                if pd.isna(val):
                    return 0.5
                try:
                    lst = json.loads(val) if isinstance(val, str) else val
                    if isinstance(lst, list) and lst:
                        return float(np.mean([x for x in lst if x is not None]))
                except (json.JSONDecodeError, TypeError):
                    pass
                return 0.5

            confs = labeled_df['jury_confidences'].apply(mean_conf).values
            for i in range(min(n_samples, len(confs))):
                signals.at[i, 'low_confidence'] = float(1.0 - confs[i])

        # --- Jury disagreement signal ---
        if labeled_df is not None and 'agreement' in labeled_df.columns:
            # Map agreement type to disagreement level
            disagreement_map = {
                'unanimous': 0.0,
                'cascade_single': 0.1,
                'majority_adjacent': 0.4,
                'candidate_confident': 0.5,
                'candidate_ambiguous': 0.7,
                'verified': 0.2,
                'unresolved': 1.0,
                'jury_failure': 1.0,
                'error': 1.0,
            }
            for i, val in enumerate(labeled_df['agreement'].values[:n_samples]):
                signals.at[i, 'jury_disagreement'] = disagreement_map.get(str(val), 0.5)

        # --- Rationale inconsistency signal ---
        rationale_results = diagnostic_results.get('rationale', {})
        divergence_df = rationale_results.get('reasoning_divergence')
        if divergence_df is not None and not divergence_df.empty and 'index' in divergence_df.columns:
            for _, row in divergence_df.iterrows():
                idx = int(row['index'])
                if idx < n_samples:
                    # 1 - rationale_sim: low similarity = high inconsistency
                    sim = float(row.get('rationale_sim', 1.0))
                    signals.at[idx, 'rationale_inconsistency'] = 1.0 - max(0.0, min(1.0, sim))

        return signals

    def compute_scores(
        self,
        signals: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """Compute weighted composite suspicion score for each sample.

        Args:
            signals: Wide DataFrame with one column per signal, indexed by sample position.
            weights: Per-signal weights. Defaults to config.suspicion_weights.

        Returns:
            Series of suspicion scores in [0, 1], indexed like signals.

        Example:
            >>> scores = scorer.compute_scores(signals_df)
            >>> top_100 = scores.nlargest(100)
        """
        if weights is None:
            weights = self.config.suspicion_weights

        score = pd.Series(0.0, index=signals.index)
        total_weight = 0.0

        for signal_name, weight in weights.items():
            if signal_name in signals.columns:
                score += weight * signals[signal_name].fillna(0.0)
                total_weight += weight

        # Normalize by actual total weight applied (in case some signals are missing)
        if total_weight > 0:
            score = score / total_weight

        return score.clip(0.0, 1.0)

    def score_dataset(
        self,
        diagnostic_results: dict[str, Any],
    ) -> pd.DataFrame:
        """Build the full signal DataFrame and compute suspicion scores.

        Convenience method that combines build_signals_dataframe and
        compute_scores into a single call.

        Args:
            diagnostic_results: Combined dict from all diagnostic modules,
                plus 'labeled_df' key with the original labeled DataFrame.

        Returns:
            DataFrame with all signal columns plus 'suspicion_score',
            sorted descending by suspicion_score.

        Example:
            >>> scored = scorer.score_dataset(all_diagnostic_results)
            >>> scored.head(20)
        """
        labeled_df = diagnostic_results.get('labeled_df')
        n_samples = len(labeled_df) if labeled_df is not None else 0

        if n_samples == 0:
            logger.warning('No labeled_df in diagnostic_results -- cannot compute suspicion scores')
            return pd.DataFrame()

        signals = self.build_signals_dataframe(n_samples, diagnostic_results)
        scores = self.compute_scores(signals)
        signals['suspicion_score'] = scores

        # Attach original text and label for readability
        if labeled_df is not None:
            text_col = labeled_df.columns[0] if 'text' not in labeled_df.columns else 'text'
            if text_col in labeled_df.columns:
                signals['text'] = labeled_df[text_col].values[:n_samples]
            if 'label' in labeled_df.columns:
                signals['label'] = labeled_df['label'].values[:n_samples]
            if 'tier' in labeled_df.columns:
                signals['tier'] = labeled_df['tier'].values[:n_samples]

        logger.info(
            f'Suspicion scoring complete: '
            f'mean={scores.mean():.3f}, '
            f'top-10% threshold={scores.quantile(0.9):.3f}'
        )

        return signals.sort_values('suspicion_score', ascending=False)
