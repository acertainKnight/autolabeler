"""Distribution-level diagnostics for detecting systematic labeling biases.

Detectors:
1. Exact duplicate texts with different labels -- definite errors.
2. Label distribution shift across sequential batches -- catches prompt drift.
3. Class confusion profile -- directional bias from jury disagreements.
4. Confidence distribution by class -- flags systematically under-confident classes.
5. Label-length correlation -- detects known LLM text-length bias.

All methods operate on the labeled output DataFrame and require no ground truth.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency, f_oneway, pointbiserialr

from .config import DiagnosticsConfig


class DistributionDiagnostics:
    """Detect systematic distributional anomalies in labeled datasets.

    Operates on the output DataFrame produced by LabelingPipeline.label_dataframe().
    Expected columns: text, label, tier, training_weight, agreement,
    jury_labels (JSON list), jury_confidences (JSON list).

    Args:
        config: DiagnosticsConfig with threshold settings.

    Example:
        >>> diag = DistributionDiagnostics(config)
        >>> results = diag.run_all(labeled_df)
        >>> results["exact_duplicates"]
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize with diagnostics configuration.

        Args:
            config: DiagnosticsConfig instance.
        """
        self.config = config

    def detect_exact_duplicates(
        self,
        df: pd.DataFrame,
        text_col: str = 'text',
        label_col: str = 'label',
    ) -> pd.DataFrame:
        """Find identical texts that received different labels.

        Exact duplicates with label disagreement are definite errors: the same
        text must always map to the same label. Zero false positives.

        Args:
            df: Labeled DataFrame.
            text_col: Name of the text column.
            label_col: Name of the label column.

        Returns:
            DataFrame of conflicting duplicate groups with columns:
                - text: the duplicated text
                - labels: set of labels assigned to this text
                - n_occurrences: how many times this text appears
                - n_unique_labels: number of distinct labels assigned

        Example:
            >>> dups = diag.detect_exact_duplicates(df)
            >>> print(f"Found {len(dups)} texts with conflicting labels")
        """
        grouped = df.groupby(text_col)[label_col].agg(list).reset_index()
        grouped.columns = ['text', 'labels']
        grouped['n_occurrences'] = grouped['labels'].apply(len)
        grouped['n_unique_labels'] = grouped['labels'].apply(lambda ls: len(set(str(l) for l in ls)))

        conflicts = grouped[grouped['n_unique_labels'] > 1].copy()
        conflicts['labels'] = conflicts['labels'].apply(
            lambda ls: sorted(set(str(l) for l in ls))
        )

        n = len(conflicts)
        if n > 0:
            logger.warning(f'Exact duplicate conflict: {n} texts have inconsistent labels')
        else:
            logger.info('No exact duplicate label conflicts found')

        return conflicts.sort_values('n_unique_labels', ascending=False)

    def label_distribution_shift(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        n_batches: int = 10,
    ) -> dict[str, Any]:
        """Detect label distribution shifts across sequential processing batches.

        Splits the DataFrame into equal-sized batches (preserving processing order)
        and computes KL divergence + chi-squared test between adjacent batches.
        Significant shifts suggest prompt drift, data ordering effects, or API
        behaviour changes.

        Args:
            df: Labeled DataFrame in processing order.
            label_col: Name of the label column.
            n_batches: Number of equal-sized batches to split into.

        Returns:
            Dictionary with:
                - batch_distributions: per-batch label distribution
                - kl_divergences: KL divergence between adjacent batches
                - chi2_results: chi-squared test results per adjacent pair
                - flagged_shifts: pairs exceeding the KL threshold
                - overall_distribution: full-dataset marginal distribution

        Example:
            >>> result = diag.label_distribution_shift(df)
            >>> result["flagged_shifts"]
        """
        if len(df) < n_batches * 5:
            n_batches = max(2, len(df) // 5)

        batch_size = len(df) // n_batches
        all_labels = sorted(df[label_col].dropna().astype(str).unique())
        batch_dists: list[dict[str, float]] = []

        for i in range(n_batches):
            batch = df.iloc[i * batch_size:(i + 1) * batch_size]
            counts = batch[label_col].astype(str).value_counts()
            total = len(batch)
            dist = {lbl: counts.get(lbl, 0) / total for lbl in all_labels}
            batch_dists.append(dist)

        # KL divergence: KL(P || Q) -- only defined where Q > 0
        def kl_div(p: dict, q: dict, labels: list[str]) -> float:
            eps = 1e-9
            p_arr = np.array([p.get(l, 0) + eps for l in labels])
            q_arr = np.array([q.get(l, 0) + eps for l in labels])
            p_arr /= p_arr.sum()
            q_arr /= q_arr.sum()
            return float(np.sum(p_arr * np.log(p_arr / q_arr)))

        kl_divs = []
        chi2_results = []
        flagged_shifts = []

        for i in range(len(batch_dists) - 1):
            p = batch_dists[i]
            q = batch_dists[i + 1]
            kl = kl_div(p, q, all_labels)
            kl_divs.append({'batch_a': i, 'batch_b': i + 1, 'kl_divergence': kl})

            # Chi-squared test on raw counts
            batch_a_counts = df.iloc[i * batch_size:(i + 1) * batch_size][label_col].astype(str).value_counts()
            batch_b_counts = df.iloc[(i + 1) * batch_size:(i + 2) * batch_size][label_col].astype(str).value_counts()

            contingency = np.array([
                [batch_a_counts.get(lbl, 0) for lbl in all_labels],
                [batch_b_counts.get(lbl, 0) for lbl in all_labels],
            ])

            try:
                chi2, p_value, _, _ = chi2_contingency(contingency)
                chi2_results.append({
                    'batch_a': i,
                    'batch_b': i + 1,
                    'chi2': float(chi2),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                })
            except ValueError:
                chi2_results.append({'batch_a': i, 'batch_b': i + 1, 'chi2': None, 'p_value': None})

            if kl > self.config.batch_drift_kl_threshold:
                flagged_shifts.append({'batch_a': i, 'batch_b': i + 1, 'kl_divergence': kl})

        overall_counts = df[label_col].astype(str).value_counts(normalize=True).to_dict()

        if flagged_shifts:
            logger.warning(f'Label distribution shift: {len(flagged_shifts)} batch pairs exceed KL threshold')
        else:
            logger.info('No significant label distribution shifts detected')

        return {
            'batch_distributions': batch_dists,
            'kl_divergences': kl_divs,
            'chi2_results': chi2_results,
            'flagged_shifts': flagged_shifts,
            'overall_distribution': overall_counts,
        }

    def class_confusion_profile(
        self,
        df: pd.DataFrame,
        jury_labels_col: str = 'jury_labels',
        final_label_col: str = 'label',
    ) -> dict[str, Any]:
        """Build a directional confusion matrix from jury disagreements.

        For each pair of classes (A, B), counts how often model votes split
        between A and B. Asymmetric confusion patterns reveal systematic biases
        without requiring ground truth. This differs from agreement metrics --
        it identifies *which* classes models confuse with *which*.

        Args:
            df: Labeled DataFrame with jury_labels column (JSON list of per-model labels).
            jury_labels_col: Column containing JSON-encoded list of jury member labels.
            final_label_col: Column with final assigned label.

        Returns:
            Dictionary with:
                - confusion_pairs: list of (label_a, label_b, count) sorted by count
                - confusion_matrix: label x label disagreement counts
                - most_confused_pairs: top 5 most confused label pairs

        Example:
            >>> result = diag.class_confusion_profile(df)
            >>> result["most_confused_pairs"]
        """
        confusion_counts: dict[tuple[str, str], int] = Counter()

        for _, row in df.iterrows():
            raw = row.get(jury_labels_col)
            if not raw or pd.isna(raw):
                continue
            try:
                jury = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue

            jury_str = [str(l) for l in jury if l is not None]
            if len(set(jury_str)) < 2:
                continue  # Unanimous -- no confusion signal

            # All unique ordered pairs (A, B) where A != B
            unique = sorted(set(jury_str))
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    confusion_counts[(unique[i], unique[j])] += 1

        if not confusion_counts:
            return {
                'confusion_pairs': [],
                'most_confused_pairs': [],
                'n_disagreements': 0,
            }

        all_labels = sorted(set(
            lbl for pair in confusion_counts for lbl in pair
        ))
        label_idx = {lbl: i for i, lbl in enumerate(all_labels)}
        matrix = np.zeros((len(all_labels), len(all_labels)), dtype=int)

        confusion_pairs = []
        for (a, b), count in confusion_counts.items():
            matrix[label_idx[a], label_idx[b]] += count
            matrix[label_idx[b], label_idx[a]] += count
            confusion_pairs.append({'label_a': a, 'label_b': b, 'count': count})

        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

        return {
            'confusion_pairs': confusion_pairs,
            'confusion_matrix': matrix.tolist(),
            'confusion_labels': all_labels,
            'most_confused_pairs': confusion_pairs[:5],
            'n_disagreements': int(sum(confusion_counts.values())),
        }

    def confidence_by_class(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        confidence_col: str = 'jury_confidences',
    ) -> dict[str, Any]:
        """Identify label classes with systematically low confidence.

        Classes where mean confidence is more than 1 sigma below the dataset
        mean are harder to classify and more error-prone.

        Args:
            df: Labeled DataFrame.
            label_col: Name of the label column.
            confidence_col: Column with JSON-encoded list of jury confidences,
                or a single float confidence column.

        Returns:
            Dictionary with:
                - per_class: dict mapping label -> mean_confidence, std, n_samples
                - dataset_mean: overall mean confidence
                - dataset_std: overall std
                - low_confidence_classes: classes flagged as >1 sigma below mean

        Example:
            >>> result = diag.confidence_by_class(df)
            >>> result["low_confidence_classes"]
        """
        # Extract mean confidence per row (handle JSON list or scalar)
        def parse_confidence(val: Any) -> float | None:
            if pd.isna(val):
                return None
            if isinstance(val, (int, float)):
                return float(val)
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
                if isinstance(parsed, list) and parsed:
                    return float(np.mean([x for x in parsed if x is not None]))
            except (json.JSONDecodeError, TypeError):
                pass
            return None

        df = df.copy()
        df['_conf'] = df[confidence_col].apply(parse_confidence)
        valid = df.dropna(subset=['_conf', label_col])

        if valid.empty:
            return {'per_class': {}, 'dataset_mean': None, 'dataset_std': None, 'low_confidence_classes': []}

        dataset_mean = float(valid['_conf'].mean())
        dataset_std = float(valid['_conf'].std()) or 1e-9

        per_class: dict[str, dict[str, float]] = {}
        for lbl, group in valid.groupby(label_col):
            per_class[str(lbl)] = {
                'mean_confidence': float(group['_conf'].mean()),
                'std_confidence': float(group['_conf'].std()),
                'n_samples': len(group),
            }

        low_confidence_classes = [
            lbl for lbl, stats in per_class.items()
            if stats['mean_confidence'] < dataset_mean - dataset_std
        ]

        if low_confidence_classes:
            logger.warning(f'Low-confidence classes: {low_confidence_classes}')

        return {
            'per_class': per_class,
            'dataset_mean': dataset_mean,
            'dataset_std': dataset_std,
            'low_confidence_classes': low_confidence_classes,
        }

    def label_length_correlation(
        self,
        df: pd.DataFrame,
        text_col: str = 'text',
        label_col: str = 'label',
    ) -> dict[str, Any]:
        """Test whether label assignment correlates with text length.

        LLMs are known to exhibit length biases. Significant correlation
        (p < 0.01) suggests the model is using text length as a proxy for
        the actual label, which is a systematic error.

        Uses point-biserial correlation for binary tasks and one-way ANOVA
        for multiclass tasks.

        Args:
            df: Labeled DataFrame.
            text_col: Name of the text column.
            label_col: Name of the label column.

        Returns:
            Dictionary with:
                - method: 'point_biserial' or 'anova'
                - statistic: correlation or F-statistic
                - p_value: significance p-value
                - significant: True if p < 0.01
                - mean_length_by_class: average text length per label class
                - interpretation: human-readable summary

        Example:
            >>> result = diag.label_length_correlation(df)
            >>> if result["significant"]:
            ...     print("Length bias detected!")
        """
        df = df.copy()
        df['_length'] = df[text_col].astype(str).apply(len)
        df = df.dropna(subset=[label_col])

        unique_labels = df[label_col].astype(str).unique()
        mean_length_by_class = (
            df.groupby(label_col)['_length'].mean().to_dict()
        )
        mean_length_by_class = {str(k): float(v) for k, v in mean_length_by_class.items()}

        if len(unique_labels) == 2:
            # Point-biserial correlation for binary
            binary = (df[label_col].astype(str) == str(unique_labels[0])).astype(int)
            stat, p_value = pointbiserialr(binary, df['_length'])
            method = 'point_biserial'
            statistic = float(stat)
        else:
            # One-way ANOVA for multiclass
            groups = [
                df[df[label_col].astype(str) == lbl]['_length'].values
                for lbl in unique_labels
                if len(df[df[label_col].astype(str) == lbl]) > 1
            ]
            if len(groups) < 2:
                return {
                    'method': 'anova',
                    'statistic': None,
                    'p_value': None,
                    'significant': False,
                    'mean_length_by_class': mean_length_by_class,
                    'interpretation': 'insufficient_data',
                }
            stat, p_value = f_oneway(*groups)
            method = 'anova'
            statistic = float(stat)

        p_value = float(p_value)
        significant = p_value < 0.01

        if significant:
            logger.warning(f'Length-label correlation detected: {method} p={p_value:.4f}')
        else:
            logger.info(f'No significant length-label correlation: {method} p={p_value:.4f}')

        interpretation = (
            f'Significant length bias ({method}, p={p_value:.4f}) -- model may use length as a proxy label.'
            if significant
            else f'No length bias detected ({method}, p={p_value:.4f})'
        )

        return {
            'method': method,
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'mean_length_by_class': mean_length_by_class,
            'interpretation': interpretation,
        }

    def run_all(
        self,
        df: pd.DataFrame,
        text_col: str = 'text',
        label_col: str = 'label',
    ) -> dict[str, Any]:
        """Run all distribution diagnostics and return combined results.

        Args:
            df: Labeled DataFrame produced by LabelingPipeline.
            text_col: Name of the text column.
            label_col: Name of the label column.

        Returns:
            Dictionary with keys for each diagnostic's results plus a summary.

        Example:
            >>> results = diag.run_all(df)
            >>> results["summary"]
        """
        logger.info(f'DistributionDiagnostics: analysing {len(df)} samples')

        exact_dups = self.detect_exact_duplicates(df, text_col, label_col)
        dist_shift = self.label_distribution_shift(df, label_col)

        if 'jury_labels' in df.columns:
            confusion = self.class_confusion_profile(df, final_label_col=label_col)
        else:
            logger.info('Skipping class confusion profile: jury_labels column not present')
            confusion = {'confusion_pairs': [], 'most_confused_pairs': [], 'n_disagreements': 0}

        if 'jury_confidences' in df.columns:
            conf_by_class = self.confidence_by_class(df, label_col)
        else:
            logger.info('Skipping confidence-by-class: jury_confidences column not present')
            conf_by_class = {'per_class': {}, 'dataset_mean': None, 'dataset_std': None, 'low_confidence_classes': []}

        length_corr = self.label_length_correlation(df, text_col, label_col)

        summary = {
            'n_exact_duplicate_conflicts': len(exact_dups),
            'n_batch_distribution_shifts': len(dist_shift.get('flagged_shifts', [])),
            'n_most_confused_pairs': len(confusion.get('most_confused_pairs', [])),
            'n_low_confidence_classes': len(conf_by_class.get('low_confidence_classes', [])),
            'length_bias_detected': length_corr.get('significant', False),
        }

        return {
            'exact_duplicates': exact_dups,
            'distribution_shift': dist_shift,
            'class_confusion': confusion,
            'confidence_by_class': conf_by_class,
            'length_correlation': length_corr,
            'summary': summary,
        }
