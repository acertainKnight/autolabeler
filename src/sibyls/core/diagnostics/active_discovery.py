"""Active error discovery: prioritised selection of samples for human review.

Uses composite suspicion scores to maximise the number of actual errors found
per human-reviewed sample. Research shows 3-5x higher error discovery rate
versus random sampling when review is guided by anomaly signals.

Three components:
1. Stratified audit sampling -- proportional sampling from suspicion tiers.
2. Quarantine report -- structured diagnostic summary per QUARANTINE sample.
3. Review batch suggestion -- top-k highest suspicion samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .config import DiagnosticsConfig


class ActiveDiscovery:
    """Select samples for human review to maximise labeling error discovery.

    Args:
        config: DiagnosticsConfig with top_k_suspects setting.

    Example:
        >>> discovery = ActiveDiscovery(config)
        >>> audit_batch = discovery.suggest_review_batch(scored_df, budget=100)
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize with diagnostics configuration.

        Args:
            config: DiagnosticsConfig instance.
        """
        self.config = config

    def stratified_audit_sample(
        self,
        scored_df: pd.DataFrame,
        n: int | None = None,
        tier_thresholds: tuple[float, float] = (0.3, 0.7),
    ) -> pd.DataFrame:
        """Sample proportionally from low, medium, and high suspicion tiers.

        Allocates review budget across suspicion tiers to simultaneously:
        - Confirm high-suspicion samples (quality signal validation)
        - Monitor medium-suspicion samples (catch less obvious errors)
        - Include some low-suspicion samples (verify true negatives)

        Allocation: 60% from high-suspicion tier, 30% medium, 10% low.

        Args:
            scored_df: DataFrame with 'suspicion_score' column.
            n: Total number of samples to select. Defaults to config.top_k_suspects.
            tier_thresholds: Tuple (low_cutoff, high_cutoff) defining tiers.
                Scores below low_cutoff = low tier, above high_cutoff = high tier.

        Returns:
            Subset DataFrame with a '_suspicion_tier' column added.

        Example:
            >>> audit = discovery.stratified_audit_sample(scored_df, n=100)
            >>> audit['_suspicion_tier'].value_counts()
        """
        n = n or self.config.top_k_suspects
        low_cut, high_cut = tier_thresholds

        scored_df = scored_df.copy()
        scored_df['_suspicion_tier'] = pd.cut(
            scored_df['suspicion_score'],
            bins=[-0.001, low_cut, high_cut, 1.001],
            labels=['low', 'medium', 'high'],
        )

        low_df = scored_df[scored_df['_suspicion_tier'] == 'low']
        med_df = scored_df[scored_df['_suspicion_tier'] == 'medium']
        high_df = scored_df[scored_df['_suspicion_tier'] == 'high']

        n_high = min(int(n * 0.60), len(high_df))
        n_med = min(int(n * 0.30), len(med_df))
        n_low = min(n - n_high - n_med, len(low_df))

        parts = []
        if n_high > 0:
            parts.append(high_df.nlargest(n_high, 'suspicion_score'))
        if n_med > 0:
            parts.append(med_df.nlargest(n_med, 'suspicion_score'))
        if n_low > 0:
            parts.append(low_df.sample(n_low, random_state=42))

        if not parts:
            return pd.DataFrame()

        result = pd.concat(parts).sort_values('suspicion_score', ascending=False)

        logger.info(
            f'Stratified audit sample: {len(result)} samples '
            f'(high={n_high}, medium={n_med}, low={n_low})'
        )
        return result

    def quarantine_report(
        self,
        labeled_df: pd.DataFrame,
        diagnostic_results: dict[str, Any],
        scored_df: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """Generate structured diagnostic reports for QUARANTINE tier samples.

        For each QUARANTINE sample, produces a report summarising:
        - Which jury models disagreed and their specific votes
        - Embedding geometry (centroid distance, centroid violation)
        - NLI coherence scores
        - Suspicion score breakdown

        This makes human review of quarantined samples much faster.

        Args:
            labeled_df: Original labeled DataFrame.
            diagnostic_results: Combined dict from all diagnostic modules.
            scored_df: Optional pre-computed suspicion-scored DataFrame.

        Returns:
            List of report dicts, one per QUARANTINE sample, sorted by
            suspicion score descending.

        Example:
            >>> reports = discovery.quarantine_report(df, results)
            >>> for r in reports[:5]:
            ...     print(r["text"], r["jury_votes"], r["suspicion_score"])
        """
        if 'tier' not in labeled_df.columns:
            logger.warning('No tier column in labeled_df -- cannot generate quarantine reports')
            return []

        quarantine_mask = labeled_df['tier'] == 'QUARANTINE'
        quarantine_df = labeled_df[quarantine_mask].copy()

        if quarantine_df.empty:
            logger.info('No QUARANTINE samples found')
            return []

        # Build lookup from index to suspicion score
        suspicion_lookup: dict[int, float] = {}
        signal_lookup: dict[int, dict[str, float]] = {}
        if scored_df is not None and 'suspicion_score' in scored_df.columns:
            signal_cols = [
                'embedding_outlier', 'nli_mismatch', 'low_confidence',
                'jury_disagreement', 'rationale_inconsistency', 'suspicion_score'
            ]
            for i in scored_df.index:
                suspicion_lookup[i] = float(scored_df.at[i, 'suspicion_score'])
                signal_lookup[i] = {
                    col: float(scored_df.at[i, col])
                    for col in signal_cols
                    if col in scored_df.columns
                }

        # Build lookup for centroid violations
        centroid_lookup: dict[int, dict[str, Any]] = {}
        emb_results = diagnostic_results.get('embedding', {})
        centroid_df = emb_results.get('centroid_violations')
        if centroid_df is not None and not centroid_df.empty and 'index' in centroid_df.columns:
            for _, row in centroid_df.iterrows():
                centroid_lookup[int(row['index'])] = {
                    'margin': float(row['margin']),
                    'nearest_other_label': row.get('nearest_other_label', ''),
                    'is_violation': bool(row.get('is_violation', False)),
                }

        # Build lookup for NLI scores
        nli_lookup: dict[int, float] = {}
        nli_results = diagnostic_results.get('nli', {})
        entailment_df = nli_results.get('entailment')
        if entailment_df is not None and not entailment_df.empty and 'index' in entailment_df.columns:
            for _, row in entailment_df.iterrows():
                nli_lookup[int(row['index'])] = float(row.get('entailment_score', 0.5))

        reports = []
        for i, row in quarantine_df.iterrows():
            jury_labels_raw = row.get('jury_labels', '[]')
            try:
                import json
                jury_votes = json.loads(jury_labels_raw) if isinstance(jury_labels_raw, str) else jury_labels_raw
            except (json.JSONDecodeError, TypeError):
                jury_votes = []

            report: dict[str, Any] = {
                'sample_index': int(i),
                'text': str(row.get('text', '')),
                'assigned_label': str(row.get('label', '')),
                'agreement': str(row.get('agreement', '')),
                'jury_votes': jury_votes,
                'jury_confidences': row.get('jury_confidences', '[]'),
                'suspicion_score': suspicion_lookup.get(int(i), 0.5),
                'signal_breakdown': signal_lookup.get(int(i), {}),
                'centroid_info': centroid_lookup.get(int(i), {}),
                'nli_entailment_score': nli_lookup.get(int(i)),
                'error': row.get('error'),
            }
            reports.append(report)

        reports.sort(key=lambda r: r['suspicion_score'], reverse=True)
        logger.info(f'Generated quarantine reports for {len(reports)} samples')
        return reports

    def suggest_review_batch(
        self,
        scored_df: pd.DataFrame,
        budget: int | None = None,
    ) -> pd.DataFrame:
        """Return the top-k most suspicious samples for immediate human review.

        Unlike stratified_audit_sample, this is a pure top-k selection for
        when you want maximum error density in your review batch (not diversity).

        Args:
            scored_df: DataFrame with 'suspicion_score' column.
            budget: Number of samples to return. Defaults to config.top_k_suspects.

        Returns:
            Top-k rows sorted by suspicion_score descending.

        Example:
            >>> batch = discovery.suggest_review_batch(scored_df, budget=50)
        """
        budget = budget or self.config.top_k_suspects
        result = scored_df.nlargest(budget, 'suspicion_score')

        logger.info(
            f'Review batch: top {len(result)} suspects '
            f'(score range {result["suspicion_score"].min():.3f} - '
            f'{result["suspicion_score"].max():.3f})'
        )
        return result
