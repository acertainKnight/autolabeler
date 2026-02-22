"""Temporal and batch-level diagnostics for detecting issues that emerge over time.

Detectors:
1. Per-model label distribution drift across batches.
2. Cascade bias: escalation rate per label class.
3. API error correlation: whether retried samples differ systematically.

All methods operate on the labeled output DataFrame and require no ground truth.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency

from .config import DiagnosticsConfig


class BatchDiagnostics:
    """Detect quality issues that emerge across processing batches over time.

    Operates on the output DataFrame produced by LabelingPipeline.label_dataframe().
    Expected columns: label, jury_labels (JSON), jury_confidences (JSON),
    cascade_models_called (optional), cascade_early_exit (optional), error (optional).

    Args:
        config: DiagnosticsConfig with threshold settings.

    Example:
        >>> diag = BatchDiagnostics(config)
        >>> results = diag.run_all(labeled_df)
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize with diagnostics configuration.

        Args:
            config: DiagnosticsConfig instance.
        """
        self.config = config

    def detect_model_drift(
        self,
        df: pd.DataFrame,
        n_batches: int = 10,
        jury_labels_col: str = 'jury_labels',
        label_col: str = 'label',
    ) -> dict[str, Any]:
        """Track per-model label distribution shift across sequential batches.

        Extracts individual model votes from jury_labels and checks whether
        any model's distribution shifts significantly across batches.
        Sudden shifts may indicate provider-side model updates, rate-limit
        degradation, or context window issues.

        Args:
            df: Labeled DataFrame in processing order.
            n_batches: Number of equal-sized batches to split into.
            jury_labels_col: Column with JSON-encoded list of jury member labels.
            label_col: Column with final consensus label.

        Returns:
            Dictionary with:
                - per_batch_distributions: batch -> model -> label -> count
                - drift_detected: whether any model showed significant drift
                - flagged_models: models with distribution shift (chi2 p < 0.05)
                - overall_model_distributions: full-run distribution per model

        Example:
            >>> result = diag.detect_model_drift(df)
            >>> result["flagged_models"]
        """
        if jury_labels_col not in df.columns:
            logger.info(f'Column {jury_labels_col!r} not found -- skipping model drift detection')
            return {
                'per_batch_distributions': [],
                'drift_detected': False,
                'flagged_models': [],
                'overall_model_distributions': {},
            }

        if len(df) < n_batches * 2:
            n_batches = max(2, len(df) // 2)

        batch_size = len(df) // n_batches
        all_labels = sorted(df[label_col].dropna().astype(str).unique())

        # Parse jury_labels to extract per-model votes
        # jury_labels is a JSON list like [label_model0, label_model1, label_model2]
        parsed_votes: list[list[str | None]] = []
        max_models = 0
        for _, row in df.iterrows():
            raw = row.get(jury_labels_col)
            if pd.isna(raw) or raw is None:
                parsed_votes.append([])
                continue
            try:
                votes = json.loads(raw) if isinstance(raw, str) else raw
                votes = [str(v) if v is not None else None for v in votes]
                parsed_votes.append(votes)
                max_models = max(max_models, len(votes))
            except (json.JSONDecodeError, TypeError):
                parsed_votes.append([])

        if max_models == 0:
            return {
                'per_batch_distributions': [],
                'drift_detected': False,
                'flagged_models': [],
                'overall_model_distributions': {},
            }

        # Per-batch, per-model distributions
        per_batch_distributions = []
        for batch_i in range(n_batches):
            start = batch_i * batch_size
            end = (batch_i + 1) * batch_size
            batch_votes = parsed_votes[start:end]

            batch_model_dist: dict[str, dict[str, int]] = {}
            for model_i in range(max_models):
                model_key = f'model_{model_i}'
                counts: dict[str, int] = {lbl: 0 for lbl in all_labels}
                for vote_list in batch_votes:
                    if model_i < len(vote_list) and vote_list[model_i] is not None:
                        lbl = vote_list[model_i]
                        counts[lbl] = counts.get(lbl, 0) + 1
                batch_model_dist[model_key] = counts

            per_batch_distributions.append(batch_model_dist)

        # Chi-squared test per model across all batches
        flagged_models = []
        for model_i in range(max_models):
            model_key = f'model_{model_i}'
            contingency = np.array([
                [batch_dist[model_key].get(lbl, 0) for lbl in all_labels]
                for batch_dist in per_batch_distributions
            ])
            # Only test if we have non-zero data
            if contingency.sum() == 0 or contingency.shape[0] < 2:
                continue
            try:
                chi2, p_value, _, _ = chi2_contingency(contingency)
                if p_value < 0.05:
                    flagged_models.append({
                        'model': model_key,
                        'chi2': float(chi2),
                        'p_value': float(p_value),
                    })
            except ValueError:
                pass

        # Overall distribution per model
        overall: dict[str, dict[str, int]] = {}
        for model_i in range(max_models):
            model_key = f'model_{model_i}'
            counts: dict[str, int] = {}
            for vote_list in parsed_votes:
                if model_i < len(vote_list) and vote_list[model_i] is not None:
                    lbl = vote_list[model_i]
                    counts[lbl] = counts.get(lbl, 0) + 1
            overall[model_key] = counts

        drift_detected = len(flagged_models) > 0
        if drift_detected:
            logger.warning(f'Model drift detected in {len(flagged_models)} model(s): {[m["model"] for m in flagged_models]}')
        else:
            logger.info('No significant per-model drift detected')

        return {
            'per_batch_distributions': per_batch_distributions,
            'drift_detected': drift_detected,
            'flagged_models': flagged_models,
            'overall_model_distributions': overall,
        }

    def cascade_bias_analysis(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        cascade_early_exit_col: str = 'cascade_early_exit',
        cascade_models_called_col: str = 'cascade_models_called',
    ) -> dict[str, Any]:
        """Analyse whether cascade escalation rate correlates with label class.

        If the cheap model systematically accepts (early-exits) for some labels
        more than others, it has a directional blind spot. Escalation rate that
        correlates strongly with label class suggests the cheap model cannot
        handle those classes.

        Skipped gracefully if cascade columns are absent.

        Args:
            df: Labeled DataFrame.
            label_col: Column with final label.
            cascade_early_exit_col: Boolean column from cascade mode.
            cascade_models_called_col: Integer column with models called count.

        Returns:
            Dictionary with:
                - available: True if cascade columns exist
                - escalation_rate_by_class: dict mapping label -> escalation rate
                - overall_escalation_rate: fraction of samples that escalated
                - biased_classes: classes with escalation rate > 1 sigma above mean

        Example:
            >>> result = diag.cascade_bias_analysis(df)
            >>> result["biased_classes"]
        """
        if cascade_early_exit_col not in df.columns:
            logger.info('Cascade columns not found -- skipping cascade bias analysis')
            return {'available': False}

        overall_escalation = float(~df[cascade_early_exit_col].fillna(True)).mean()

        escalation_by_class: dict[str, float] = {}
        for lbl, group in df.groupby(label_col):
            escalated = ~group[cascade_early_exit_col].fillna(True)
            escalation_by_class[str(lbl)] = float(escalated.mean())

        if escalation_by_class:
            rates = np.array(list(escalation_by_class.values()))
            mean_rate = rates.mean()
            std_rate = rates.std() or 1e-9
            biased_classes = [
                lbl for lbl, rate in escalation_by_class.items()
                if rate > mean_rate + std_rate
            ]
        else:
            biased_classes = []

        if biased_classes:
            logger.warning(f'Cascade bias detected: high escalation rate in classes {biased_classes}')

        return {
            'available': True,
            'escalation_rate_by_class': escalation_by_class,
            'overall_escalation_rate': overall_escalation,
            'biased_classes': biased_classes,
        }

    def api_error_correlation(
        self,
        df: pd.DataFrame,
        error_col: str = 'error',
        label_col: str = 'label',
        tier_col: str = 'tier',
    ) -> dict[str, Any]:
        """Check whether samples that had API errors differ from clean samples.

        If the error column is absent or always null, this is skipped.
        Partial jury results (some models failed) may have lower quality even
        if they pass tier thresholds.

        Args:
            df: Labeled DataFrame.
            error_col: Column containing error messages (null = clean).
            label_col: Column with final label.
            tier_col: Column with tier assignment.

        Returns:
            Dictionary with:
                - available: True if error column exists and has non-null values
                - n_error_samples: count of samples with errors
                - error_label_distribution: label distribution in error samples
                - clean_label_distribution: label distribution in clean samples
                - tier_difference: fraction QUARANTINE in error vs clean samples

        Example:
            >>> result = diag.api_error_correlation(df)
            >>> result["tier_difference"]
        """
        if error_col not in df.columns or df[error_col].isna().all():
            return {'available': False}

        has_error = df[error_col].notna() & (df[error_col] != '')
        n_errors = int(has_error.sum())

        if n_errors == 0:
            return {'available': True, 'n_error_samples': 0}

        error_df = df[has_error]
        clean_df = df[~has_error]

        error_label_dist = error_df[label_col].astype(str).value_counts(normalize=True).to_dict()
        clean_label_dist = clean_df[label_col].astype(str).value_counts(normalize=True).to_dict()

        error_quarantine_rate = float((error_df[tier_col] == 'QUARANTINE').mean()) if tier_col in error_df.columns else 0.0
        clean_quarantine_rate = float((clean_df[tier_col] == 'QUARANTINE').mean()) if tier_col in clean_df.columns else 0.0

        tier_difference = error_quarantine_rate - clean_quarantine_rate

        if abs(tier_difference) > 0.1:
            logger.warning(
                f'API error correlation: error samples have {tier_difference:+.1%} '
                f'different QUARANTINE rate vs clean samples'
            )

        return {
            'available': True,
            'n_error_samples': n_errors,
            'n_clean_samples': int((~has_error).sum()),
            'error_label_distribution': error_label_dist,
            'clean_label_distribution': clean_label_dist,
            'error_quarantine_rate': error_quarantine_rate,
            'clean_quarantine_rate': clean_quarantine_rate,
            'tier_difference': tier_difference,
        }

    def run_all(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run all batch diagnostics and return combined results.

        Args:
            df: Labeled DataFrame produced by LabelingPipeline.

        Returns:
            Dictionary with keys for each diagnostic's results plus a summary.

        Example:
            >>> results = diag.run_all(df)
            >>> results["summary"]
        """
        logger.info(f'BatchDiagnostics: analysing {len(df)} samples')

        model_drift = self.detect_model_drift(df)
        cascade_bias = self.cascade_bias_analysis(df)
        api_errors = self.api_error_correlation(df)

        summary = {
            'model_drift_detected': model_drift.get('drift_detected', False),
            'n_flagged_models': len(model_drift.get('flagged_models', [])),
            'cascade_available': cascade_bias.get('available', False),
            'n_cascade_biased_classes': len(cascade_bias.get('biased_classes', [])),
            'n_api_error_samples': api_errors.get('n_error_samples', 0),
        }

        return {
            'model_drift': model_drift,
            'cascade_bias': cascade_bias,
            'api_errors': api_errors,
            'summary': summary,
        }
