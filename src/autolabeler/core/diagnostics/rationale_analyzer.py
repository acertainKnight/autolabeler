"""Rationale consistency analysis for detecting when LLM reasoning contradicts labels.

Three detectors:
1. Keyword-label alignment -- rationale contains keywords from the wrong class.
2. Rationale embedding clustering -- rationale embeds closer to a different class.
3. Cross-juror rationale disagreement -- unanimous label but divergent reasoning.

These catch subtle errors where the final label looks plausible but the chain
of reasoning reveals the model was confused. Unanimous agreement with divergent
reasoning is especially fragile -- models may have reached the right answer
for different or wrong reasons.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .config import DiagnosticsConfig


class RationaleAnalyzer:
    """Detect inconsistencies between LLM rationales and their assigned labels.

    Args:
        config: DiagnosticsConfig with embedding model settings.

    Example:
        >>> analyzer = RationaleAnalyzer(config)
        >>> results = analyzer.run_all(df, embedding_analyzer)
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize with diagnostics configuration.

        Args:
            config: DiagnosticsConfig instance.
        """
        self.config = config
        self._embedding_analyzer = None

    def _get_embedding_analyzer(self) -> Any:
        """Lazily create an EmbeddingAnalyzer for rationale embeddings."""
        if self._embedding_analyzer is None:
            from .embedding_analyzer import EmbeddingAnalyzer
            self._embedding_analyzer = EmbeddingAnalyzer(self.config)
        return self._embedding_analyzer

    def keyword_label_alignment(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        reasoning_col: str = 'reasoning',
        keyword_map: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        """Flag rationales containing keywords strongly associated with a different class.

        A rationale that says "rate cuts" while the label is hawkish, or says
        "inflation surge" while the label is dovish, indicates a mismatch between
        the reasoning and the decision.

        Args:
            df: Labeled DataFrame with a reasoning/rationale column.
            label_col: Column with assigned label.
            reasoning_col: Column with rationale text. Supports JSON-encoded lists
                (multiple juror rationales).
            keyword_map: Mapping from label string to list of indicator keywords.
                If None, uses a built-in default keyword map for hawkish/dovish tasks.
                Custom maps should use lowercase keywords.

        Returns:
            DataFrame with columns:
                - index: sample position
                - label: assigned label
                - keyword_found: keyword that triggered the flag
                - keyword_class: the class that keyword is associated with
                - is_misaligned: True when keyword belongs to a different class

        Example:
            >>> df_flags = analyzer.keyword_label_alignment(df)
            >>> df_flags[df_flags["is_misaligned"]].head()
        """
        if keyword_map is None:
            # Default map for hawk/dove monetary policy tasks
            keyword_map = {
                '-2': ['rate cut', 'rate cuts', 'easing', 'stimulus', 'dovish', 'lower rates', 'accommodative', 'recession', 'slowdown'],
                '-1': ['cautious', 'gradual', 'patient', 'moderate easing', 'slight cut', 'slow pace'],
                '0': ['neutral', 'balanced', 'unchanged', 'hold', 'steady', 'mixed signals', 'wait and see'],
                '1': ['hawkish', 'tightening', 'rate hike', 'rate increase', 'inflation concern', 'restrictive'],
                '2': ['aggressive tightening', 'significant hike', 'emergency hike', 'inflation surge', 'runaway inflation', 'strongly hawkish'],
            }

        # Build inverted index: keyword -> class
        keyword_to_class: dict[str, str] = {}
        for lbl, keywords in keyword_map.items():
            for kw in keywords:
                keyword_to_class[kw.lower()] = lbl

        def extract_reasoning(val: Any) -> str:
            """Parse reasoning field which may be a JSON list or plain string."""
            if pd.isna(val) or val is None:
                return ''
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return ' '.join(str(x) for x in parsed if x)
                except json.JSONDecodeError:
                    pass
                return val
            if isinstance(val, list):
                return ' '.join(str(x) for x in val if x)
            return str(val)

        records = []
        for i, row in df.iterrows():
            lbl = str(row.get(label_col, ''))
            reasoning_text = extract_reasoning(row.get(reasoning_col)).lower()

            for keyword, keyword_class in keyword_to_class.items():
                if keyword in reasoning_text and keyword_class != lbl:
                    records.append({
                        'index': i,
                        'label': lbl,
                        'keyword_found': keyword,
                        'keyword_class': keyword_class,
                        'is_misaligned': True,
                    })
                    break  # One flag per sample is enough

        result_df = pd.DataFrame(records)
        if result_df.empty:
            result_df = pd.DataFrame(columns=['index', 'label', 'keyword_found', 'keyword_class', 'is_misaligned'])

        n_flags = len(result_df)
        if n_flags > 0:
            logger.warning(f'Keyword-label misalignment: {n_flags} samples flagged')
        else:
            logger.info('No keyword-label misalignments detected')

        return result_df

    def rationale_embedding_clustering(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        reasoning_col: str = 'reasoning',
    ) -> pd.DataFrame:
        """Flag samples whose rationale embeds closer to a different class's rationale cluster.

        A sophisticated indicator: even if keywords match, the overall semantic
        content of the reasoning may be inconsistent with the assigned label class.

        Args:
            df: Labeled DataFrame.
            label_col: Column with assigned label.
            reasoning_col: Column with rationale text.

        Returns:
            DataFrame with columns:
                - index: sample position
                - label: assigned label
                - assigned_centroid_sim: similarity to assigned class rationale centroid
                - nearest_other_label: label of nearest other rationale cluster
                - nearest_other_sim: similarity to that cluster
                - margin: assigned_sim - nearest_other_sim (negative = cluster mismatch)
                - is_cluster_mismatch: True when margin < 0

        Example:
            >>> df_mismatches = analyzer.rationale_embedding_clustering(df)
            >>> df_mismatches[df_mismatches["is_cluster_mismatch"]].head()
        """
        def extract_reasoning(val: Any) -> str:
            if pd.isna(val) or val is None:
                return ''
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return ' '.join(str(x) for x in parsed if x)
                except json.JSONDecodeError:
                    pass
                return val
            if isinstance(val, list):
                return ' '.join(str(x) for x in val if x)
            return str(val)

        rationales = [extract_reasoning(row) for _, row in df.iterrows()]
        labels = [str(row[label_col]) for _, row in df.iterrows()]

        # Skip samples with empty rationales
        valid_mask = [bool(r.strip()) for r in rationales]
        if sum(valid_mask) < 4:
            logger.warning('Insufficient rationale data for embedding clustering -- skipping')
            return pd.DataFrame(columns=['index', 'label', 'margin', 'is_cluster_mismatch'])

        valid_indices = [i for i, m in enumerate(valid_mask) if m]
        valid_rationales = [rationales[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]

        analyzer = self._get_embedding_analyzer()
        embeddings = analyzer.embed_texts(valid_rationales)

        label_array = np.array(valid_labels)
        unique_labels = np.unique(label_array)

        # Compute per-class rationale centroids
        centroids: dict[str, np.ndarray] = {}
        for lbl in unique_labels:
            mask = label_array == lbl
            if mask.sum() < 2:
                continue
            centroid = normalize(embeddings[mask].mean(axis=0, keepdims=True)).ravel()
            centroids[lbl] = centroid

        if len(centroids) < 2:
            logger.warning('Not enough classes with multiple rationales -- skipping rationale clustering')
            return pd.DataFrame(columns=['index', 'label', 'margin', 'is_cluster_mismatch'])

        centroid_matrix = np.stack(list(centroids.values()))
        centroid_labels = list(centroids.keys())

        sims = cosine_similarity(embeddings, centroid_matrix)

        records = []
        for local_i, global_i in enumerate(valid_indices):
            lbl = valid_labels[local_i]
            if lbl not in centroids:
                continue

            assigned_idx = centroid_labels.index(lbl)
            assigned_sim = float(sims[local_i, assigned_idx])

            other_sims = {
                centroid_labels[j]: float(sims[local_i, j])
                for j in range(len(centroid_labels))
                if j != assigned_idx
            }
            if not other_sims:
                continue

            nearest_other = max(other_sims, key=other_sims.get)
            nearest_other_sim = other_sims[nearest_other]
            margin = assigned_sim - nearest_other_sim

            records.append({
                'index': global_i,
                'label': lbl,
                'assigned_centroid_sim': assigned_sim,
                'nearest_other_label': nearest_other,
                'nearest_other_sim': nearest_other_sim,
                'margin': margin,
                'is_cluster_mismatch': margin < 0,
            })

        result_df = pd.DataFrame(records).sort_values('margin')
        n_mismatches = int(result_df['is_cluster_mismatch'].sum()) if not result_df.empty else 0

        if n_mismatches > 0:
            logger.warning(f'Rationale cluster mismatches: {n_mismatches} samples')
        else:
            logger.info('No rationale cluster mismatches detected')

        return result_df

    def cross_juror_rationale_disagreement(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        jury_labels_col: str = 'jury_labels',
        reasoning_col: str = 'reasoning',
    ) -> pd.DataFrame:
        """Flag unanimous-label samples where jurors gave divergent reasoning.

        Unanimous agreement on label with divergent reasoning means models reached
        the same answer via different logic paths -- this is fragile and may not
        generalise. Low cosine similarity between juror rationale embeddings
        indicates reasoning divergence.

        Args:
            df: Labeled DataFrame.
            label_col: Column with final consensus label.
            jury_labels_col: Column with JSON list of per-juror labels.
            reasoning_col: Column with JSON list of per-juror reasoning strings.

        Returns:
            DataFrame with columns:
                - index: sample position
                - label: unanimous label
                - n_jurors: number of jurors who agreed
                - rationale_sim: mean pairwise cosine similarity of rationale embeddings
                - is_divergent: True when mean similarity < 0.5

        Example:
            >>> df_div = analyzer.cross_juror_rationale_disagreement(df)
            >>> df_div[df_div["is_divergent"]].head()
        """
        def parse_json_list(val: Any) -> list:
            if pd.isna(val) or val is None:
                return []
            if isinstance(val, list):
                return val
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return []

        records = []
        all_rationale_texts = []
        sample_indices_for_embedding = []

        for i, row in df.iterrows():
            jury_labels = parse_json_list(row.get(jury_labels_col))
            if not jury_labels:
                continue

            # Only consider unanimous samples
            unique_votes = set(str(v) for v in jury_labels if v is not None)
            if len(unique_votes) != 1:
                continue

            n_jurors = len(jury_labels)
            reasoning_list = parse_json_list(row.get(reasoning_col))
            if len(reasoning_list) < 2:
                continue

            rationale_texts = [str(r) for r in reasoning_list if r]
            if len(rationale_texts) < 2:
                continue

            records.append({
                'index': i,
                'label': str(row[label_col]),
                'n_jurors': n_jurors,
                '_rationale_texts': rationale_texts,
            })
            all_rationale_texts.extend(rationale_texts)
            sample_indices_for_embedding.append((len(all_rationale_texts) - len(rationale_texts), len(all_rationale_texts)))

        if not records:
            return pd.DataFrame(columns=['index', 'label', 'n_jurors', 'rationale_sim', 'is_divergent'])

        analyzer = self._get_embedding_analyzer()
        all_embeddings = analyzer.embed_texts(all_rationale_texts)

        output_records = []
        for record, (start, end) in zip(records, sample_indices_for_embedding):
            rationale_embeddings = all_embeddings[start:end]
            if len(rationale_embeddings) < 2:
                continue
            sim_matrix = cosine_similarity(rationale_embeddings)
            # Mean of upper triangle (pairwise sims)
            n = len(rationale_embeddings)
            upper_indices = np.triu_indices(n, k=1)
            mean_sim = float(sim_matrix[upper_indices].mean()) if len(upper_indices[0]) > 0 else 1.0

            output_records.append({
                'index': record['index'],
                'label': record['label'],
                'n_jurors': record['n_jurors'],
                'rationale_sim': mean_sim,
                'is_divergent': mean_sim < 0.5,
            })

        result_df = pd.DataFrame(output_records).sort_values('rationale_sim')
        n_divergent = int(result_df['is_divergent'].sum()) if not result_df.empty else 0

        if n_divergent > 0:
            logger.warning(f'Cross-juror rationale divergence: {n_divergent} unanimous samples with inconsistent reasoning')

        return result_df

    def run_all(
        self,
        df: pd.DataFrame,
        keyword_map: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Run all rationale consistency analyses and return combined results.

        Args:
            df: Labeled DataFrame with label, reasoning, and jury_labels columns.
            keyword_map: Optional custom keyword-to-class mapping.

        Returns:
            Dictionary with keys for each analysis's results plus a summary.

        Example:
            >>> results = analyzer.run_all(df)
            >>> results["summary"]
        """
        logger.info(f'RationaleAnalyzer: analysing {len(df)} samples')

        keyword_df = self.keyword_label_alignment(df, keyword_map=keyword_map)
        cluster_df = self.rationale_embedding_clustering(df)
        divergence_df = self.cross_juror_rationale_disagreement(df)

        n_keyword = len(keyword_df) if not keyword_df.empty else 0
        n_cluster = int(cluster_df['is_cluster_mismatch'].sum()) if not cluster_df.empty and 'is_cluster_mismatch' in cluster_df else 0
        n_divergent = int(divergence_df['is_divergent'].sum()) if not divergence_df.empty and 'is_divergent' in divergence_df else 0

        logger.info(
            f'Rationale analysis complete: '
            f'{n_keyword} keyword misalignments, '
            f'{n_cluster} cluster mismatches, '
            f'{n_divergent} reasoning divergences'
        )

        return {
            'keyword_misalignments': keyword_df,
            'cluster_mismatches': cluster_df,
            'reasoning_divergence': divergence_df,
            'summary': {
                'n_keyword_misalignments': n_keyword,
                'n_cluster_mismatches': n_cluster,
                'n_reasoning_divergences': n_divergent,
            },
        }
