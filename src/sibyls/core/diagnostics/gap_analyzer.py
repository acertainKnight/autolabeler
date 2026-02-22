"""LLM-powered gap analysis: identify systematic weaknesses in training data.

Takes error signals from the diagnostics pipeline (suspicion scores, centroid
violations, cluster fragmentation) and groups them into interpretable themes.
For each theme, an LLM diagnoses what the classifier is missing and optionally
generates synthetic examples to fill the gap.

Pipeline stages:
    1. Error pool construction -- collect top suspects + centroid violations.
    2. Semantic clustering -- UMAP + HDBSCAN on pool embeddings to group by topic.
    3. Cluster summarisation -- TF-IDF topic terms, label distribution, severity.
    4. LLM diagnosis -- per-cluster ambiguity analysis & synthetic data generation.

Outputs written to output_dir:
    - gap_report.json   -- full structured report with cluster details
    - gap_report.md     -- human-readable Markdown version
    - synthetic_examples.csv  -- generated training examples (if enabled)

Usage:
    Typically run as part of the diagnostics pipeline (step 10 in __init__.py)
    when ``diagnostics.gap_analysis.enabled: true`` is set in the dataset config.

    Can also be invoked standalone:

    >>> from sibyls.core.diagnostics.gap_analyzer import GapAnalyzer
    >>> from sibyls.core.diagnostics.config import DiagnosticsConfig
    >>> config = DiagnosticsConfig.from_dict({
    ...     "gap_analysis": {"enabled": True, "generate_synthetic": True},
    ... })
    >>> analyzer = GapAnalyzer(config)
    >>> results = analyzer.run_all(
    ...     labeled_df, diagnostic_results, output_dir=Path("outputs/gaps"),
    ...     label_definitions={"1": "hawkish", "0": "neutral", "-1": "dovish"},
    ... )
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .config import DiagnosticsConfig, GapAnalysisConfig


class GapAnalyzer:
    """Cluster classifier errors and diagnose systematic gaps with an LLM.

    Args:
        config: DiagnosticsConfig containing a gap_analysis sub-config.

    Example:
        >>> analyzer = GapAnalyzer(diag_config)
        >>> results = analyzer.run_all(
        ...     labeled_df, diagnostic_results, output_dir=Path("outputs/gaps")
        ... )
        >>> print(results["gap_report"]["n_clusters"])
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialise with parent DiagnosticsConfig.

        Args:
            config: Full DiagnosticsConfig; gap_analysis sub-config is read from it.
        """
        self.config = config
        self.gap_cfg: GapAnalysisConfig = config.gap_analysis or GapAnalysisConfig()

    # ------------------------------------------------------------------
    # Stage 1: Build the error pool
    # ------------------------------------------------------------------

    def build_error_pool(
        self,
        labeled_df: pd.DataFrame,
        diagnostic_results: dict[str, Any],
    ) -> pd.DataFrame:
        """Collect high-suspicion samples and centroid violations.

        Combines the top-N suspicious samples with any centroid-violation
        samples that aren't already in that set.  The result is a DataFrame
        with columns: index, text, label, suspicion_score, is_violation.

        Args:
            labeled_df: Original labeled DataFrame (must contain 'text' and 'label').
            diagnostic_results: Combined diagnostics output from run_diagnostics().

        Returns:
            DataFrame of error-pool samples, deduplicated by original index.

        Example:
            >>> pool = analyzer.build_error_pool(df, results)
            >>> len(pool)
            350
        """
        rows: list[dict[str, Any]] = []
        seen_indices: set[int] = set()

        # Pull top-N from scored_df
        scored_df: pd.DataFrame | None = diagnostic_results.get('scored_df')
        if scored_df is not None and not scored_df.empty and 'suspicion_score' in scored_df.columns:
            top_n = min(self.gap_cfg.top_n_suspicious, len(scored_df))
            top_df = scored_df.nlargest(top_n, 'suspicion_score').reset_index(drop=True)

            text_col = 'text' if 'text' in top_df.columns else labeled_df.columns[0]
            for _, row in top_df.iterrows():
                idx = int(row.get('index', row.name))
                seen_indices.add(idx)
                rows.append({
                    'index': idx,
                    'text': str(row.get(text_col, '')),
                    'label': str(row.get('label', '')),
                    'suspicion_score': float(row.get('suspicion_score', 0.0)),
                    'is_violation': False,
                })

        # Augment with centroid violations not already in the pool
        centroid_df: pd.DataFrame | None = (
            diagnostic_results.get('embedding', {}).get('centroid_violations')
        )
        if centroid_df is not None and not centroid_df.empty and 'is_violation' in centroid_df.columns:
            violations = centroid_df[centroid_df['is_violation']].copy()
            text_col = 'text' if 'text' in labeled_df.columns else labeled_df.columns[0]

            for _, row in violations.iterrows():
                idx = int(row['index'])
                if idx in seen_indices or idx >= len(labeled_df):
                    continue
                seen_indices.add(idx)
                orig = labeled_df.iloc[idx]
                rows.append({
                    'index': idx,
                    'text': str(orig.get(text_col, '')),
                    'label': str(orig.get('label', '')),
                    'suspicion_score': float(row.get('margin', 0.0)),
                    'is_violation': True,
                })

        if not rows:
            logger.warning('GapAnalyzer: empty error pool -- no suspicious samples found')
            return pd.DataFrame(columns=['index', 'text', 'label', 'suspicion_score', 'is_violation'])

        pool = pd.DataFrame(rows).drop_duplicates(subset=['index']).reset_index(drop=True)
        logger.info(f'GapAnalyzer: error pool has {len(pool)} samples')
        return pool

    # ------------------------------------------------------------------
    # Stage 2: Semantic clustering
    # ------------------------------------------------------------------

    def cluster_errors(
        self,
        pool_df: pd.DataFrame,
        diagnostic_results: dict[str, Any],
    ) -> pd.DataFrame:
        """UMAP + HDBSCAN cluster the error pool by semantic similarity.

        Reuses embeddings computed by EmbeddingAnalyzer (already cached) to
        avoid re-encoding. Falls back to empty cluster column if embeddings
        are unavailable.

        Args:
            pool_df: DataFrame from build_error_pool().
            diagnostic_results: Combined diagnostics dict with 'embedding' key.

        Returns:
            pool_df with an additional 'cluster' column (-1 = noise).

        Example:
            >>> clustered = analyzer.cluster_errors(pool_df, results)
            >>> clustered['cluster'].value_counts()
        """
        try:
            import hdbscan
            import umap
        except ImportError as exc:
            logger.warning(f'GapAnalyzer: hdbscan/umap not installed -- skipping clustering: {exc}')
            pool_df = pool_df.copy()
            pool_df['cluster'] = -1
            return pool_df

        all_embeddings: np.ndarray | None = (
            diagnostic_results.get('embedding', {}).get('embeddings')
        )

        if all_embeddings is None:
            logger.warning('GapAnalyzer: no embeddings in diagnostic_results -- skipping clustering')
            pool_df = pool_df.copy()
            pool_df['cluster'] = -1
            return pool_df

        # Extract embeddings for the error pool indices
        pool_indices = pool_df['index'].tolist()
        valid_mask = [i < len(all_embeddings) for i in pool_indices]
        valid_indices = [i for i, ok in zip(pool_indices, valid_mask) if ok]

        if len(valid_indices) < self.gap_cfg.min_cluster_size * 2:
            logger.warning(
                f'GapAnalyzer: only {len(valid_indices)} pool samples have embeddings '
                f'(need ≥ {self.gap_cfg.min_cluster_size * 2}) -- skipping clustering'
            )
            pool_df = pool_df.copy()
            pool_df['cluster'] = -1
            return pool_df

        pool_embeddings = all_embeddings[valid_indices]

        # UMAP → low-dim for density-based clustering
        umap_dim = min(self.config.fragmentation_umap_dim, pool_embeddings.shape[1], len(valid_indices) - 2)
        umap_dim = max(2, umap_dim)

        logger.info(
            f'GapAnalyzer: UMAP-reducing {pool_embeddings.shape[1]}-D → {umap_dim}-D '
            f'for {len(valid_indices)} error-pool samples'
        )
        reducer = umap.UMAP(
            n_components=umap_dim,
            n_neighbors=min(15, len(valid_indices) - 1),
            min_dist=0.0,
            metric='cosine',
            random_state=42,
        )
        reduced = reducer.fit_transform(pool_embeddings)

        min_cluster = min(
            self.gap_cfg.min_cluster_size,
            max(2, len(valid_indices) // 20),
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            metric='euclidean',
        )
        cluster_labels = clusterer.fit_predict(reduced)

        # Prune smallest clusters if over cap
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
        if len(unique_clusters) > self.gap_cfg.max_clusters:
            cluster_sizes = {c: int((cluster_labels == c).sum()) for c in unique_clusters}
            keep = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[: self.gap_cfg.max_clusters]
            keep_set = set(keep)
            cluster_labels = np.where(np.isin(cluster_labels, list(keep_set)), cluster_labels, -1)
            logger.info(
                f'GapAnalyzer: pruned to top {self.gap_cfg.max_clusters} clusters '
                f'(removed {len(unique_clusters) - self.gap_cfg.max_clusters} smallest)'
            )

        # Write back -- samples whose index was out-of-range get noise label
        pool_df = pool_df.copy()
        pool_df['cluster'] = -1
        idx_to_cluster: dict[int, int] = {
            pool_idx: int(cluster_labels[local_i])
            for local_i, pool_idx in enumerate(valid_indices)
        }
        pool_df['cluster'] = pool_df['index'].map(idx_to_cluster).fillna(-1).astype(int)

        n_clusters = len([c for c in np.unique(cluster_labels) if c != -1])
        n_noise = int((pool_df['cluster'] == -1).sum())
        logger.info(f'GapAnalyzer: {n_clusters} clusters, {n_noise} noise samples')
        return pool_df

    # ------------------------------------------------------------------
    # Stage 3: Cluster summaries (TF-IDF topic labels)
    # ------------------------------------------------------------------

    def summarise_clusters(self, clustered_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Build a structured summary for each cluster (without LLM).

        Computes TF-IDF top terms as a lightweight topic label, label
        distribution, mean suspicion score, and representative texts.

        Args:
            clustered_df: Output of cluster_errors() with 'cluster' column.

        Returns:
            List of cluster summary dicts, sorted by severity descending.

        Example:
            >>> summaries = analyzer.summarise_clusters(clustered_df)
            >>> summaries[0]["topic_terms"]
            ['rate', 'hike', 'inflation', 'fed', 'hikes']
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        cluster_ids = sorted(set(clustered_df['cluster'].tolist()) - {-1})
        summaries: list[dict[str, Any]] = []

        all_texts = clustered_df['text'].astype(str).tolist()
        try:
            tfidf = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2),
            )
            tfidf.fit(all_texts)
            vocab = tfidf.get_feature_names_out()
        except Exception:
            tfidf = None
            vocab = np.array([])

        for cid in cluster_ids:
            mask = clustered_df['cluster'] == cid
            cluster_rows = clustered_df[mask]
            texts = cluster_rows['text'].astype(str).tolist()
            labels = cluster_rows['label'].astype(str).tolist()

            mean_suspicion = float(cluster_rows['suspicion_score'].mean())
            n_violations = int(cluster_rows.get('is_violation', pd.Series([False] * len(cluster_rows))).sum())
            label_dist = pd.Series(labels).value_counts(normalize=True).round(3).to_dict()

            # Representative samples: closest to centroid in TF-IDF space
            representatives = texts[: self.gap_cfg.representative_samples]
            if tfidf is not None and len(texts) > 1:
                try:
                    vecs = tfidf.transform(texts).toarray()
                    centroid = vecs.mean(axis=0, keepdims=True)
                    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
                    sims = _cos_sim(vecs, centroid).ravel()
                    top_idx = np.argsort(sims)[::-1][: self.gap_cfg.representative_samples]
                    representatives = [texts[i] for i in top_idx]
                except Exception:
                    pass

            # Top TF-IDF terms for this cluster
            topic_terms: list[str] = []
            if tfidf is not None and len(vocab) > 0:
                try:
                    cluster_vecs = tfidf.transform(texts).toarray()
                    mean_vec = cluster_vecs.mean(axis=0)
                    top_term_idx = np.argsort(mean_vec)[::-1][:10]
                    topic_terms = [str(vocab[i]) for i in top_term_idx]
                except Exception:
                    pass

            severity = len(cluster_rows) * mean_suspicion
            summaries.append({
                'cluster_id': int(cid),
                'n_samples': len(cluster_rows),
                'n_violations': n_violations,
                'mean_suspicion': round(mean_suspicion, 4),
                'severity': round(severity, 4),
                'label_distribution': label_dist,
                'topic_terms': topic_terms,
                'representative_texts': representatives,
                'llm_diagnosis': None,  # filled in stage 4
                'synthetic_examples': [],  # filled in stage 4
            })

        summaries.sort(key=lambda x: x['severity'], reverse=True)
        return summaries

    # ------------------------------------------------------------------
    # Stage 4: LLM diagnosis
    # ------------------------------------------------------------------

    async def _diagnose_cluster_async(
        self,
        summary: dict[str, Any],
        label_definitions: dict[str, str] | None,
        provider: Any,
    ) -> dict[str, Any]:
        """Call the LLM to diagnose one gap cluster.

        Args:
            summary: Cluster summary from summarise_clusters().
            label_definitions: Optional mapping of label -> description.
            provider: LLM provider instance.

        Returns:
            Updated summary dict with 'llm_diagnosis' and 'synthetic_examples' filled.
        """
        reps = summary['representative_texts']
        numbered = '\n'.join(f'{i + 1}. {t}' for i, t in enumerate(reps))
        label_dist_str = json.dumps(summary['label_distribution'])
        topic_hint = ', '.join(summary['topic_terms'][:5]) if summary['topic_terms'] else 'unknown'

        label_defs_block = ''
        if label_definitions:
            label_defs_block = '\n\nLabel definitions:\n' + '\n'.join(
                f'  {lbl}: {desc}' for lbl, desc in label_definitions.items()
            )

        synth_instruction = ''
        if self.gap_cfg.generate_synthetic:
            synth_instruction = (
                f'\n\n5. Generate {self.gap_cfg.synthetic_per_cluster} synthetic training examples '
                f'that would help a model distinguish this cluster correctly. '
                f'For each example provide: the text and the correct label.'
            )

        system_prompt = (
            'You are an expert data scientist analysing classifier errors to improve training data quality. '
            'Be concise, specific, and actionable. Respond in JSON only.'
        )

        user_prompt = f"""Analyse this cluster of texts that a classifier is struggling with.

Top keywords: {topic_hint}
Label distribution (assigned labels): {label_dist_str}
Number of samples: {summary['n_samples']}
Number of centroid violations: {summary['n_violations']}

Representative texts:
{numbered}
{label_defs_block}

Answer the following as a JSON object with these exact keys:
1. "topic": One concise phrase (≤8 words) naming the topic/theme of this cluster.
2. "ambiguity": What makes these texts hard to classify correctly? (1-2 sentences)
3. "likely_correct_labels": A JSON object mapping label → estimated_probability for what the correct labels should be.
4. "augmentation_strategy": What kinds of additional training examples would help? (1-2 sentences){synth_instruction}

Return only valid JSON with keys: topic, ambiguity, likely_correct_labels, augmentation_strategy{', synthetic_examples' if self.gap_cfg.generate_synthetic else ''}.
If generating synthetic examples, use key "synthetic_examples" as a list of objects with "text" and "label"."""

        try:
            from sibyls.core.llm_providers.providers import LLMResponse
            response: LLMResponse = await provider.call(
                system=system_prompt,
                user=user_prompt,
                temperature=0.3,
            )

            if response.parsed_json:
                diagnosis = response.parsed_json
            else:
                # Attempt to extract JSON from raw text
                raw = response.text or ''
                start = raw.find('{')
                end = raw.rfind('}') + 1
                diagnosis = json.loads(raw[start:end]) if start != -1 and end > start else {}

            summary['llm_diagnosis'] = {
                'topic': str(diagnosis.get('topic', topic_hint)),
                'ambiguity': str(diagnosis.get('ambiguity', '')),
                'likely_correct_labels': diagnosis.get('likely_correct_labels', {}),
                'augmentation_strategy': str(diagnosis.get('augmentation_strategy', '')),
            }

            if self.gap_cfg.generate_synthetic:
                raw_synth = diagnosis.get('synthetic_examples', [])
                summary['synthetic_examples'] = [
                    {'text': str(ex.get('text', '')), 'label': str(ex.get('label', ''))}
                    for ex in raw_synth
                    if isinstance(ex, dict)
                ]

        except Exception as exc:
            logger.warning(f'GapAnalyzer: LLM diagnosis failed for cluster {summary["cluster_id"]}: {exc}')
            summary['llm_diagnosis'] = {
                'topic': topic_hint,
                'ambiguity': f'LLM diagnosis failed: {exc}',
                'likely_correct_labels': {},
                'augmentation_strategy': '',
            }

        return summary

    def diagnose_clusters(
        self,
        summaries: list[dict[str, Any]],
        label_definitions: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run LLM diagnosis on all cluster summaries.

        Args:
            summaries: Output of summarise_clusters().
            label_definitions: Optional label -> natural-language description.

        Returns:
            Summaries with 'llm_diagnosis' and 'synthetic_examples' populated.

        Example:
            >>> diagnosed = analyzer.diagnose_clusters(summaries, label_defs)
            >>> diagnosed[0]['llm_diagnosis']['topic']
            'Conditional rate hike projections'
        """
        try:
            from sibyls.core.llm_providers.providers import get_provider
        except ImportError as exc:
            logger.error(f'GapAnalyzer: cannot import LLM providers: {exc}')
            return summaries

        try:
            provider = get_provider(self.gap_cfg.analysis_provider, self.gap_cfg.analysis_model)
        except Exception as exc:
            logger.error(f'GapAnalyzer: failed to init analysis provider: {exc}')
            return summaries

        logger.info(
            f'GapAnalyzer: diagnosing {len(summaries)} clusters with '
            f'{self.gap_cfg.analysis_provider}/{self.gap_cfg.analysis_model}'
        )

        async def _run_all() -> list[dict[str, Any]]:
            tasks = [
                self._diagnose_cluster_async(s, label_definitions, provider)
                for s in summaries
            ]
            return list(await asyncio.gather(*tasks))

        try:
            diagnosed = asyncio.run(_run_all())
        except RuntimeError:
            # Already inside an event loop (e.g. Jupyter)
            loop = asyncio.get_event_loop()
            diagnosed = loop.run_until_complete(_run_all())

        return diagnosed

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _write_gap_report_markdown(
        self,
        summaries: list[dict[str, Any]],
        output_path: Path,
        dataset_name: str,
    ) -> None:
        """Write gap report as Markdown.

        Args:
            summaries: Diagnosed cluster summaries.
            output_path: Target Markdown file path.
            dataset_name: Dataset name for the report header.
        """
        lines = [
            f'# Gap Analysis Report: {dataset_name}',
            '',
            f'Total error clusters identified: **{len(summaries)}**',
            '',
            '---',
            '',
        ]

        total_synthetic = sum(len(s['synthetic_examples']) for s in summaries)
        if total_synthetic:
            lines += [f'Synthetic examples generated: **{total_synthetic}**', '']

        for i, s in enumerate(summaries, 1):
            diag = s.get('llm_diagnosis') or {}
            topic = diag.get('topic') or ', '.join(s['topic_terms'][:3]) or f'Cluster {s["cluster_id"]}'
            lines += [
                f'## {i}. {topic}',
                '',
                f'**Severity score:** {s["severity"]:.2f} | '
                f'**Samples:** {s["n_samples"]} | '
                f'**Centroid violations:** {s["n_violations"]} | '
                f'**Mean suspicion:** {s["mean_suspicion"]:.3f}',
                '',
                f'**Assigned label distribution:** `{json.dumps(s["label_distribution"])}`',
                '',
            ]

            if diag.get('ambiguity'):
                lines += [f'**Why it\'s hard:** {diag["ambiguity"]}', '']

            if diag.get('likely_correct_labels'):
                lines += [f'**Likely correct labels:** `{json.dumps(diag["likely_correct_labels"])}`', '']

            if diag.get('augmentation_strategy'):
                lines += [f'**Augmentation strategy:** {diag["augmentation_strategy"]}', '']

            lines += ['**Top keywords:** ' + ', '.join(f'`{t}`' for t in s['topic_terms'][:8]), '']

            lines += [f'**Representative texts** (top {min(5, len(s["representative_texts"]))}):']
            for j, text in enumerate(s['representative_texts'][:5], 1):
                lines.append(f'{j}. {text[:120]}')
            lines.append('')

            if s['synthetic_examples']:
                lines += [f'**Synthetic examples** ({len(s["synthetic_examples"])}):']
                for j, ex in enumerate(s['synthetic_examples'], 1):
                    lines.append(f'{j}. `[{ex["label"]}]` {ex["text"]}')
                lines.append('')

            lines.append('---')
            lines.append('')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_all(
        self,
        labeled_df: pd.DataFrame,
        diagnostic_results: dict[str, Any],
        output_dir: Path,
        label_definitions: dict[str, str] | None = None,
        dataset_name: str = 'dataset',
    ) -> dict[str, Any]:
        """Run the full gap analysis pipeline and save outputs.

        Args:
            labeled_df: Labeled DataFrame (text + label columns required).
            diagnostic_results: Combined output from run_diagnostics().
            output_dir: Directory to save gap_report.json and gap_report.md.
            label_definitions: Optional label -> description for LLM context.
            dataset_name: Dataset name for report headers.

        Returns:
            Dict with:
                - pool_df: Error-pool DataFrame.
                - clustered_df: Pool DataFrame with cluster assignments.
                - summaries: List of diagnosed cluster dicts.
                - gap_report: High-level summary dict.

        Example:
            >>> results = analyzer.run_all(df, diag_results, Path("outputs/gaps"))
            >>> results["gap_report"]["n_clusters"]
            12
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info('GapAnalyzer: building error pool...')
        pool_df = self.build_error_pool(labeled_df, diagnostic_results)

        if pool_df.empty:
            logger.warning('GapAnalyzer: no errors to analyse')
            return {'pool_df': pool_df, 'clustered_df': pool_df, 'summaries': [], 'gap_report': {}}

        logger.info('GapAnalyzer: clustering errors...')
        clustered_df = self.cluster_errors(pool_df, diagnostic_results)

        logger.info('GapAnalyzer: summarising clusters...')
        summaries = self.summarise_clusters(clustered_df)

        if not summaries:
            logger.warning('GapAnalyzer: no clusters found (all noise)')
            return {
                'pool_df': pool_df,
                'clustered_df': clustered_df,
                'summaries': [],
                'gap_report': {'n_clusters': 0},
            }

        logger.info(f'GapAnalyzer: running LLM diagnosis on {len(summaries)} clusters...')
        summaries = self.diagnose_clusters(summaries, label_definitions)

        # Save outputs
        report_path = output_dir / 'gap_report.json'
        all_synthetic: list[dict[str, Any]] = []
        for s in summaries:
            all_synthetic.extend(s.get('synthetic_examples', []))

        gap_report = {
            'dataset_name': dataset_name,
            'n_error_pool': len(pool_df),
            'n_clusters': len(summaries),
            'n_synthetic_examples': len(all_synthetic),
            'clusters': summaries,
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(gap_report, f, indent=2, default=str)
        logger.info(f'GapAnalyzer: gap report (JSON) saved to {report_path}')

        md_path = output_dir / 'gap_report.md'
        self._write_gap_report_markdown(summaries, md_path, dataset_name)
        logger.info(f'GapAnalyzer: gap report (Markdown) saved to {md_path}')

        # Save synthetic examples as CSV for easy import
        if all_synthetic:
            synth_path = output_dir / 'synthetic_examples.csv'
            pd.DataFrame(all_synthetic).to_csv(synth_path, index=False)
            logger.info(f'GapAnalyzer: {len(all_synthetic)} synthetic examples saved to {synth_path}')

        logger.info(
            f'GapAnalyzer: complete -- '
            f'{len(summaries)} clusters, {len(all_synthetic)} synthetic examples'
        )

        return {
            'pool_df': pool_df,
            'clustered_df': clustered_df,
            'summaries': summaries,
            'gap_report': gap_report,
        }
