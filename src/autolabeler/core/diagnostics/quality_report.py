"""Aggregate quality reporting: generates a comprehensive post-labeling diagnostic report.

Aggregates all diagnostic module outputs into:
1. Per-class quality scorecard.
2. Model reliability matrix.
3. Prioritised, actionable recommendations.

Exports to Markdown (human-readable) and JSON (programmatic consumption).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .config import DiagnosticsConfig


class QualityReportGenerator:
    """Generate a comprehensive quality report from all diagnostic module outputs.

    Args:
        config: DiagnosticsConfig with threshold settings.

    Example:
        >>> reporter = QualityReportGenerator(config)
        >>> reporter.generate(all_results, output_dir=Path("outputs/diagnostics"))
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize with diagnostics configuration.

        Args:
            config: DiagnosticsConfig instance.
        """
        self.config = config

    def per_class_scorecard(
        self,
        labeled_df: pd.DataFrame,
        diagnostic_results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build a quality scorecard for each label class.

        Combines sample count, mean confidence, embedding cohesion, NLI
        coherence, and fragmentation data into a per-class summary.

        Args:
            labeled_df: Original labeled DataFrame.
            diagnostic_results: Combined dict from all diagnostic modules.

        Returns:
            List of per-class score dicts sorted by estimated error risk descending.

        Example:
            >>> scorecard = reporter.per_class_scorecard(df, results)
            >>> for entry in scorecard:
            ...     print(entry["label"], entry["estimated_error_risk"])
        """
        import json as _json

        unique_labels = sorted(labeled_df['label'].dropna().astype(str).unique())
        scorecard = []

        conf_results = diagnostic_results.get('distribution', {}).get('confidence_by_class', {})
        per_class_conf = conf_results.get('per_class', {})
        dataset_mean_conf = conf_results.get('dataset_mean') or 0.7

        fragmentation = diagnostic_results.get('embedding', {}).get('cluster_fragmentation', {})
        centroid_df = diagnostic_results.get('embedding', {}).get('centroid_violations')
        nli_entailment = diagnostic_results.get('nli', {}).get('entailment')

        for lbl in unique_labels:
            lbl_mask = labeled_df['label'].astype(str) == lbl
            n_samples = int(lbl_mask.sum())

            # Mean confidence
            conf_stats = per_class_conf.get(lbl, {})
            mean_conf = conf_stats.get('mean_confidence', dataset_mean_conf)

            # Centroid violation rate
            centroid_violation_rate = 0.0
            if centroid_df is not None and not centroid_df.empty and 'label' in centroid_df.columns:
                lbl_centroid = centroid_df[centroid_df['label'].astype(str) == lbl]
                if len(lbl_centroid) > 0:
                    centroid_violation_rate = float(lbl_centroid['is_violation'].mean())

            # NLI mean entailment score
            mean_nli = None
            if nli_entailment is not None and not nli_entailment.empty and 'label' in nli_entailment.columns:
                lbl_nli = nli_entailment[nli_entailment['label'].astype(str) == lbl]
                if not lbl_nli.empty:
                    mean_nli = float(lbl_nli['entailment_score'].mean())

            # Fragmentation
            frag_info = fragmentation.get(lbl, {})
            n_clusters = frag_info.get('n_clusters', 1)
            is_fragmented = frag_info.get('fragmented', False)

            # Tier distribution (only for pipeline-labeled data)
            if 'tier' in labeled_df.columns:
                lbl_tiers = labeled_df[lbl_mask]['tier'].value_counts(normalize=True).to_dict()
            else:
                lbl_tiers = {}

            # Heuristic error risk estimate
            risk_factors = []
            if mean_conf is not None and mean_conf < 0.6:
                risk_factors.append('low_confidence')
            if centroid_violation_rate > 0.1:
                risk_factors.append('centroid_violations')
            if mean_nli is not None and mean_nli < self.config.nli_entailment_threshold:
                risk_factors.append('low_nli_coherence')
            if is_fragmented:
                risk_factors.append('cluster_fragmentation')

            estimated_error_risk = len(risk_factors) / 4.0

            scorecard.append({
                'label': lbl,
                'n_samples': n_samples,
                'mean_confidence': round(mean_conf, 4) if mean_conf is not None else None,
                'centroid_violation_rate': round(centroid_violation_rate, 4),
                'mean_nli_entailment': round(mean_nli, 4) if mean_nli is not None else None,
                'n_embedding_clusters': n_clusters,
                'is_fragmented': is_fragmented,
                'tier_distribution': {str(k): round(v, 3) for k, v in lbl_tiers.items()},
                'risk_factors': risk_factors,
                'estimated_error_risk': round(estimated_error_risk, 3),
            })

        scorecard.sort(key=lambda x: x['estimated_error_risk'], reverse=True)
        return scorecard

    def model_reliability_matrix(
        self,
        labeled_df: pd.DataFrame,
        diagnostic_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Cross-tabulate model × class reliability metrics.

        For each model, computes:
        - Agreement with majority vote per class
        - Whether model drift was detected

        Args:
            labeled_df: Original labeled DataFrame.
            diagnostic_results: Combined dict from all diagnostic modules.

        Returns:
            Dictionary with model x class agreement matrix and drift flags.

        Example:
            >>> matrix = reporter.model_reliability_matrix(df, results)
            >>> matrix["drift_flags"]
        """
        import json as _json

        batch_results = diagnostic_results.get('batch', {})
        model_drift = batch_results.get('model_drift', {})
        drift_flags = {m['model']: True for m in model_drift.get('flagged_models', [])}
        overall_model_dists = model_drift.get('overall_model_distributions', {})

        if 'label' not in labeled_df.columns:
            return {'per_model_class_agreement': {}, 'drift_flags': drift_flags, 'overall_model_distributions': overall_model_dists}

        unique_labels = sorted(labeled_df['label'].dropna().astype(str).unique())

        # Build per-model agreement with final label
        per_model_agreement: dict[str, dict[str, float]] = {}
        jury_labels_col = 'jury_labels'

        if jury_labels_col in labeled_df.columns:
            # Collect per-model votes
            all_model_votes: dict[int, list[str | None]] = {}
            for i, row in labeled_df.iterrows():
                raw = row.get(jury_labels_col)
                if pd.isna(raw):
                    continue
                try:
                    votes = _json.loads(raw) if isinstance(raw, str) else raw
                    all_model_votes[int(i)] = [str(v) if v is not None else None for v in votes]
                except (json.JSONDecodeError, TypeError):
                    pass

            max_models = max((len(v) for v in all_model_votes.values()), default=0)

            for model_i in range(max_models):
                model_key = f'model_{model_i}'
                per_class_agreement: dict[str, float] = {}

                for lbl in unique_labels:
                    lbl_rows = labeled_df[labeled_df['label'].astype(str) == lbl]
                    n_agree = 0
                    n_total = 0
                    for idx, row in lbl_rows.iterrows():
                        votes = all_model_votes.get(int(idx), [])
                        if model_i < len(votes) and votes[model_i] is not None:
                            n_total += 1
                            if votes[model_i] == lbl:
                                n_agree += 1
                    per_class_agreement[lbl] = round(n_agree / n_total, 3) if n_total > 0 else None

                per_model_agreement[model_key] = per_class_agreement

        return {
            'per_model_class_agreement': per_model_agreement,
            'drift_flags': drift_flags,
            'overall_model_distributions': overall_model_dists,
        }

    def generate_recommendations(
        self,
        labeled_df: pd.DataFrame,
        diagnostic_results: dict[str, Any],
        scorecard: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Auto-generate prioritised actionable recommendations.

        Translates diagnostic findings into human-readable action items,
        sorted by estimated impact.

        Args:
            labeled_df: Original labeled DataFrame.
            diagnostic_results: Combined dict from all diagnostic modules.
            scorecard: Output from per_class_scorecard().

        Returns:
            Sorted list of recommendation dicts with keys:
                - priority: 'high', 'medium', 'low'
                - finding: what was detected
                - action: what to do about it
                - evidence: supporting metrics

        Example:
            >>> recs = reporter.generate_recommendations(df, results, scorecard)
            >>> for r in recs:
            ...     print(f"[{r['priority']}] {r['finding']}")
        """
        recommendations = []

        # Near-duplicate conflicts (highest priority -- zero false positives)
        dist_results = diagnostic_results.get('distribution', {})
        exact_dups = dist_results.get('exact_duplicates')
        if exact_dups is not None and not exact_dups.empty:
            n = len(exact_dups)
            recommendations.append({
                'priority': 'high',
                'finding': f'{n} identical texts have conflicting labels',
                'action': 'Review these samples immediately -- same text must always receive same label.',
                'evidence': {'n_conflicts': n},
            })

        # Centroid violations (high priority)
        emb_results = diagnostic_results.get('embedding', {})
        centroid_df = emb_results.get('centroid_violations')
        if centroid_df is not None and not centroid_df.empty:
            n_violations = int(centroid_df['is_violation'].sum())
            if n_violations > 0:
                top_violated = centroid_df[centroid_df['is_violation']].head(3)[['label', 'nearest_other_label', 'margin']].to_dict('records')
                recommendations.append({
                    'priority': 'high',
                    'finding': f'{n_violations} samples are geometrically closer to a different class centroid',
                    'action': 'Inspect centroid violations sorted by margin -- negative margin samples are near-certain errors.',
                    'evidence': {'n_violations': n_violations, 'worst_examples': top_violated},
                })

        # NLI contrastive violations (high priority)
        nli_results = diagnostic_results.get('nli', {})
        contrastive_df = nli_results.get('contrastive')
        if contrastive_df is not None and not contrastive_df.empty and 'is_contrastive_violation' in contrastive_df.columns:
            n_cv = int(contrastive_df['is_contrastive_violation'].sum())
            if n_cv > 0:
                recommendations.append({
                    'priority': 'high',
                    'finding': f'{n_cv} samples have higher NLI entailment for an alternative label than their assigned label',
                    'action': 'Review NLI contrastive violations -- NLI is orthogonal to jury, so overlap is especially suspicious.',
                    'evidence': {'n_violations': n_cv},
                })

        # Fragmented classes (medium priority)
        fragmentation = emb_results.get('cluster_fragmentation', {})
        for lbl, info in fragmentation.items():
            if info.get('fragmented'):
                n_clusters = info['n_clusters']
                recommendations.append({
                    'priority': 'medium',
                    'finding': f'Class "{lbl}" fragments into {n_clusters} HDBSCAN clusters',
                    'action': f'Review class "{lbl}" definition in rules.md -- it may need sub-categories or clearer criteria.',
                    'evidence': {'label': lbl, 'n_clusters': n_clusters, 'cluster_sizes': info.get('cluster_sizes', [])},
                })

        # Low-confidence classes (medium priority)
        low_conf_classes = dist_results.get('confidence_by_class', {}).get('low_confidence_classes', [])
        if low_conf_classes:
            per_class = dist_results.get('confidence_by_class', {}).get('per_class', {})
            class_details = {c: per_class.get(c, {}).get('mean_confidence') for c in low_conf_classes}
            recommendations.append({
                'priority': 'medium',
                'finding': f'Classes {low_conf_classes} have systematically low jury confidence',
                'action': 'Add more calibration examples for these classes to the examples.md prompt.',
                'evidence': {'class_mean_confidence': class_details},
            })

        # Model drift (medium priority)
        batch_results = diagnostic_results.get('batch', {})
        flagged_models = batch_results.get('model_drift', {}).get('flagged_models', [])
        if flagged_models:
            recommendations.append({
                'priority': 'medium',
                'finding': f'{len(flagged_models)} jury model(s) show label distribution drift across batches',
                'action': 'Check if provider updated the model mid-run. Consider re-labeling batches where drift was detected.',
                'evidence': {'flagged_models': [m['model'] for m in flagged_models]},
            })

        # Length bias (low priority)
        length_corr = dist_results.get('length_correlation', {})
        if length_corr.get('significant'):
            recommendations.append({
                'priority': 'low',
                'finding': f'Label assignment correlates with text length ({length_corr.get("method")}, p={length_corr.get("p_value", 0):.4f})',
                'action': 'Add length-diverse calibration examples to break the length-label correlation.',
                'evidence': {
                    'method': length_corr.get('method'),
                    'p_value': length_corr.get('p_value'),
                    'mean_length_by_class': length_corr.get('mean_length_by_class', {}),
                },
            })

        # Distribution shift (low priority)
        flagged_shifts = dist_results.get('distribution_shift', {}).get('flagged_shifts', [])
        if flagged_shifts:
            recommendations.append({
                'priority': 'low',
                'finding': f'Label distribution shifted significantly between {len(flagged_shifts)} adjacent batch pairs',
                'action': 'Check processing order and prompt consistency across batches.',
                'evidence': {'n_flagged_shifts': len(flagged_shifts), 'shifts': flagged_shifts[:3]},
            })

        # Sort: high > medium > low
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda r: priority_order.get(r['priority'], 3))

        return recommendations

    def generate(
        self,
        diagnostic_results: dict[str, Any],
        output_dir: Path,
        formats: list[str] | None = None,
        dataset_name: str = 'dataset',
    ) -> dict[str, Any]:
        """Generate and save the full quality report.

        Args:
            diagnostic_results: Combined output from all diagnostic modules,
                including 'labeled_df' key with original labeled DataFrame.
            output_dir: Directory to write report files.
            formats: List of output formats: 'markdown', 'json' (default both).
            dataset_name: Dataset name for report header.

        Returns:
            Dictionary containing the complete report data.

        Example:
            >>> report = reporter.generate(results, Path("outputs/diagnostics"))
            >>> print(report["summary"]["total_suspects"])
        """
        if formats is None:
            formats = ['markdown', 'json']

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        labeled_df = diagnostic_results.get('labeled_df')
        if labeled_df is None:
            logger.error('No labeled_df in diagnostic_results -- cannot generate report')
            return {}

        logger.info(f'Generating quality report for {dataset_name} ({len(labeled_df)} samples)...')

        scorecard = self.per_class_scorecard(labeled_df, diagnostic_results)
        model_matrix = self.model_reliability_matrix(labeled_df, diagnostic_results)
        recommendations = self.generate_recommendations(labeled_df, diagnostic_results, scorecard)

        scored_df = diagnostic_results.get('scored_df')
        top_suspects = None
        if scored_df is not None and not scored_df.empty and 'suspicion_score' in scored_df.columns:
            top_k = min(self.config.top_k_suspects, len(scored_df))
            top_suspects = scored_df.nlargest(top_k, 'suspicion_score')[
                [c for c in ['text', 'label', 'tier', 'suspicion_score'] if c in scored_df.columns]
            ].to_dict('records')

        # Summary statistics
        tier_counts = labeled_df['tier'].value_counts().to_dict() if 'tier' in labeled_df.columns else {}
        label_counts = (
            labeled_df['label'].astype(str).value_counts().to_dict()
            if 'label' in labeled_df.columns
            else {}
        )

        summary = {
            'dataset_name': dataset_name,
            'n_samples': len(labeled_df),
            'tier_distribution': tier_counts,
            'label_distribution': label_counts,
            'n_recommendations': len(recommendations),
            'n_high_priority_recommendations': sum(1 for r in recommendations if r['priority'] == 'high'),
            'total_suspects': len(top_suspects) if top_suspects else 0,
            'generated_at': datetime.now().isoformat(),
        }

        report = {
            'summary': summary,
            'scorecard': scorecard,
            'model_matrix': model_matrix,
            'recommendations': recommendations,
            'top_suspects': top_suspects,
        }

        if 'json' in formats:
            json_path = output_dir / 'quality_report.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f'Quality report (JSON) saved to {json_path}')

        if 'markdown' in formats:
            md_path = output_dir / 'quality_report.md'
            self._write_markdown(report, md_path, dataset_name)
            logger.info(f'Quality report (Markdown) saved to {md_path}')

        return report

    def _write_markdown(
        self,
        report: dict[str, Any],
        path: Path,
        dataset_name: str,
    ) -> None:
        """Write the quality report as a Markdown document.

        Args:
            report: Full report dictionary.
            path: Output file path.
            dataset_name: Dataset name for title.
        """
        summary = report['summary']
        scorecard = report['scorecard']
        recommendations = report['recommendations']
        top_suspects = report.get('top_suspects') or []

        lines = [
            f'# Quality Report: {dataset_name}',
            f'',
            f'Generated: {summary.get("generated_at", "")}',
            f'',
            f'## Summary',
            f'',
            f'| Metric | Value |',
            f'|--------|-------|',
            f'| Total samples | {summary.get("n_samples", 0):,} |',
            f'| Recommendations | {summary.get("n_recommendations", 0)} ({summary.get("n_high_priority_recommendations", 0)} high priority) |',
            f'| Top suspects identified | {summary.get("total_suspects", 0)} |',
            f'',
        ]

        # Tier distribution
        tier_dist = summary.get('tier_distribution', {})
        if tier_dist:
            n = summary.get('n_samples', 1)
            lines += ['## Tier Distribution', '']
            for tier, count in sorted(tier_dist.items()):
                pct = count / n * 100
                lines.append(f'- **{tier}**: {count:,} ({pct:.1f}%)')
            lines.append('')

        # Recommendations
        lines += ['## Actionable Recommendations', '']
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(rec['priority'], '⚪')
                lines += [
                    f'### {i}. {priority_emoji} [{rec["priority"].upper()}] {rec["finding"]}',
                    f'',
                    f'**Action:** {rec["action"]}',
                    f'',
                    f'**Evidence:** `{json.dumps(rec.get("evidence", {}), default=str)}`',
                    f'',
                ]
        else:
            lines.append('No significant issues detected.')
            lines.append('')

        # Per-class scorecard
        lines += ['## Per-Class Quality Scorecard', '']
        if scorecard:
            lines += [
                '| Label | Samples | Mean Confidence | Centroid Violation Rate | NLI Entailment | Error Risk |',
                '|-------|---------|-----------------|------------------------|----------------|------------|',
            ]
            for entry in scorecard:
                nli = f'{entry["mean_nli_entailment"]:.3f}' if entry.get('mean_nli_entailment') is not None else 'N/A'
                conf = f'{entry["mean_confidence"]:.3f}' if entry.get('mean_confidence') is not None else 'N/A'
                lines.append(
                    f'| {entry["label"]} '
                    f'| {entry["n_samples"]:,} '
                    f'| {conf} '
                    f'| {entry["centroid_violation_rate"]:.3f} '
                    f'| {nli} '
                    f'| {entry["estimated_error_risk"]:.2f} |'
                )
            lines.append('')

        # Top suspects
        if top_suspects:
            lines += [
                f'## Top {len(top_suspects)} Suspects for Review',
                '',
                '| # | Label | Tier | Suspicion | Text (truncated) |',
                '|---|-------|------|-----------|-----------------|',
            ]
            for i, suspect in enumerate(top_suspects[:20], 1):
                text_preview = str(suspect.get('text', ''))[:80].replace('|', '\\|')
                lines.append(
                    f'| {i} '
                    f'| {suspect.get("label", "")} '
                    f'| {suspect.get("tier", "")} '
                    f'| {suspect.get("suspicion_score", 0):.3f} '
                    f'| {text_preview}... |'
                )
            lines.append('')

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
