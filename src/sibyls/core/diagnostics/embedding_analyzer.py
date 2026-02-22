"""Embedding-space analysis for detecting geometrically inconsistent label assignments.

Four complementary detectors:
1. Intra-class outlier detection -- samples far from their class centroid.
2. Cross-class nearest-centroid violation -- samples closer to a different class centroid.
3. Cluster fragmentation -- classes that split into multiple HDBSCAN clusters.
4. Near-duplicate detection -- high-similarity pairs with conflicting labels.

Embedding backends are pluggable via ``embedding_provider`` in DiagnosticsConfig:
    - "local": sentence-transformers on-device (default, no API key needed).
    - "openai": OpenAI Embeddings API (text-embedding-3-small / large).
    - "openrouter": OpenRouter Embeddings API (any available model).

All embeddings are cached to disk so repeated diagnostic runs are cheap.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize

from .config import DiagnosticsConfig
from .embedding_providers import BaseEmbeddingProvider, get_embedding_provider


class EmbeddingAnalyzer:
    """Detect label errors via embedding-space geometry.

    Encodes texts using a pluggable embedding provider (local, OpenAI, or
    OpenRouter), then applies geometric tests to surface samples that are
    inconsistent with their assigned label.

    The cross-class nearest-centroid violation is the single highest-signal
    detector: a negative margin (sample closer to another class centroid than
    its own) is a near-certain labeling error.

    Args:
        config: DiagnosticsConfig controlling provider, model, thresholds, and cache.

    Example:
        >>> analyzer = EmbeddingAnalyzer(config)
        >>> results = analyzer.run_all(texts, labels)
        >>> suspects = results["centroid_violations"].query("margin < 0")
    """

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize the analyzer. Provider instantiation is deferred to first call.

        Args:
            config: Diagnostics configuration.
        """
        self.config = config
        self._provider: BaseEmbeddingProvider | None = None
        self._cache_dir = Path(config.embedding_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_provider(self) -> BaseEmbeddingProvider:
        """Lazily create the embedding provider from config."""
        if self._provider is None:
            self._provider = get_embedding_provider(self.config)
        return self._provider

    def _cache_key(self, texts: list[str]) -> str:
        """Build a deterministic cache key from texts + provider + model name.

        Args:
            texts: List of input texts.

        Returns:
            Hex digest string usable as a filename.
        """
        provider_name = getattr(self.config, 'embedding_provider', 'local')
        content = json.dumps({
            'provider': provider_name,
            'model': self.config.embedding_model,
            'texts': texts,
        })
        return hashlib.sha256(content.encode()).hexdigest()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts as L2-normalized embedding vectors with disk caching.

        Delegates to the configured embedding provider (local, OpenAI, or
        OpenRouter). Results are cached to disk so repeated runs skip encoding.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of shape (n_texts, embedding_dim), L2-normalized.

        Example:
            >>> embeddings = analyzer.embed_texts(["Fed raises rates", "Dovish pivot"])
            >>> embeddings.shape
            (2, 384)
        """
        cache_path = self._cache_dir / f'{self._cache_key(texts)}.pkl'

        if cache_path.exists():
            logger.info(f'Loading embeddings from cache: {cache_path}')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        provider = self._get_provider()
        embeddings = provider.embed(texts, batch_size=self.config.embedding_batch_size)

        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.debug(f'Cached embeddings to {cache_path}')

        return embeddings

    def detect_intraclass_outliers(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> pd.DataFrame:
        """Detect samples that are geometrically distant from their class centroid.

        For each label class, computes the centroid and measures each sample's
        cosine distance to it. Samples with z-score above threshold are flagged.
        A secondary LOF score provides robustness for non-spherical clusters.

        Args:
            embeddings: L2-normalized embedding matrix, shape (n, d).
            labels: Label for each sample, length n.

        Returns:
            DataFrame with columns:
                - index: original sample index
                - label: assigned label
                - centroid_distance: cosine distance to class centroid
                - centroid_z_score: z-score within class
                - lof_score: LOF outlier score (higher = more anomalous)
                - is_outlier: True if z_score > threshold

        Example:
            >>> df = analyzer.detect_intraclass_outliers(embeddings, labels)
            >>> df[df['is_outlier']].head()
        """
        label_array = np.array(labels)
        unique_labels = np.unique(label_array)
        records = []

        for lbl in unique_labels:
            mask = label_array == lbl
            idx = np.where(mask)[0]
            class_embs = embeddings[mask]

            if len(class_embs) < 2:
                # Cannot compute statistics with a single sample
                for i in idx:
                    records.append({
                        'index': i,
                        'label': lbl,
                        'centroid_distance': 0.0,
                        'centroid_z_score': 0.0,
                        'lof_score': 1.0,
                        'is_outlier': False,
                    })
                continue

            centroid = class_embs.mean(axis=0, keepdims=True)
            centroid = normalize(centroid)
            distances = cosine_distances(class_embs, centroid).ravel()

            mean_dist = distances.mean()
            std_dist = distances.std() or 1e-9
            z_scores = (distances - mean_dist) / std_dist

            # LOF within the class (catches non-spherical cluster outliers)
            n_neighbors = min(self.config.lof_neighbors, len(class_embs) - 1)
            lof = LocalOutlierFactor(n_neighbors=max(1, n_neighbors))
            lof_scores = -lof.fit_predict(class_embs)  # LOF: -1 = outlier, 1 = inlier

            for local_i, global_i in enumerate(idx):
                records.append({
                    'index': global_i,
                    'label': lbl,
                    'centroid_distance': float(distances[local_i]),
                    'centroid_z_score': float(z_scores[local_i]),
                    'lof_score': float(lof_scores[local_i]),
                    'is_outlier': bool(z_scores[local_i] > self.config.outlier_z_threshold),
                })

        return pd.DataFrame(records).sort_values('centroid_z_score', ascending=False)

    def detect_centroid_violations(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> pd.DataFrame:
        """Flag samples closer to a different class centroid than their own.

        This is the highest-signal geometric detector. A negative margin means
        the sample is geometrically in the wrong class -- a near-certain error.

        Margin = dist(sample, assigned_centroid) - dist(sample, nearest_other_centroid).
        Negative margin → likely labeling error.

        Args:
            embeddings: L2-normalized embedding matrix, shape (n, d).
            labels: Label for each sample, length n.

        Returns:
            DataFrame with columns:
                - index: original sample index
                - label: assigned label
                - assigned_centroid_dist: distance to assigned class centroid
                - nearest_other_label: label of nearest alternative centroid
                - nearest_other_dist: distance to nearest alternative centroid
                - margin: assigned_dist - nearest_other_dist (negative = error)
                - is_violation: True when margin < 0

        Example:
            >>> df = analyzer.detect_centroid_violations(embeddings, labels)
            >>> certain_errors = df[df['margin'] < 0].sort_values('margin')
        """
        label_array = np.array(labels)
        unique_labels = np.unique(label_array)

        # Compute per-class centroids
        centroids: dict[str, np.ndarray] = {}
        for lbl in unique_labels:
            mask = label_array == lbl
            centroid = normalize(embeddings[mask].mean(axis=0, keepdims=True)).ravel()
            centroids[lbl] = centroid

        centroid_matrix = np.stack(list(centroids.values()))  # (n_classes, d)
        centroid_labels = list(centroids.keys())

        # Distance from every sample to every centroid
        all_distances = cosine_distances(embeddings, centroid_matrix)  # (n, n_classes)

        records = []
        for i, (lbl, sample_dists) in enumerate(zip(labels, all_distances)):
            assigned_idx = centroid_labels.index(lbl)
            assigned_dist = sample_dists[assigned_idx]

            # Nearest OTHER centroid
            other_dists = {
                centroid_labels[j]: sample_dists[j]
                for j in range(len(centroid_labels))
                if j != assigned_idx
            }
            nearest_other_lbl = min(other_dists, key=other_dists.get)
            nearest_other_dist = other_dists[nearest_other_lbl]

            margin = float(assigned_dist - nearest_other_dist)
            records.append({
                'index': i,
                'label': lbl,
                'assigned_centroid_dist': float(assigned_dist),
                'nearest_other_label': nearest_other_lbl,
                'nearest_other_dist': float(nearest_other_dist),
                'margin': margin,
                'is_violation': margin < 0,
            })

        return pd.DataFrame(records).sort_values('margin')

    def detect_cluster_fragmentation(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> dict[str, Any]:
        """Run HDBSCAN within each label class to detect fragmentation.

        Follows the BERTopic pattern: UMAP-reduce the full embedding space to
        a low-dimensional representation before clustering. This avoids the
        curse of dimensionality (density estimates are unreliable in 1000+ dims)
        and makes the KD-tree in HDBSCAN orders of magnitude faster.

        The UMAP reduction is performed once on all samples (preserving global
        structure), then sliced per-class for HDBSCAN.

        A class that fragments into k > 2 disconnected clusters suggests
        either underspecified labeling criteria or a genuine multi-modal
        distribution in the data.

        Args:
            embeddings: L2-normalized embedding matrix, shape (n, d).
            labels: Label for each sample, length n.

        Returns:
            Dictionary mapping each label to:
                - n_clusters: number of HDBSCAN clusters found (excluding noise)
                - n_noise: samples assigned to noise cluster (-1)
                - fragmented: True if n_clusters > 2
                - cluster_sizes: list of cluster sizes

        Example:
            >>> report = analyzer.detect_cluster_fragmentation(embeddings, labels)
            >>> for lbl, info in report.items():
            ...     if info['fragmented']:
            ...         print(f"Class {lbl} has {info['n_clusters']} sub-clusters")
        """
        try:
            import hdbscan
        except ImportError:
            logger.warning('hdbscan not installed -- skipping fragmentation analysis. pip install hdbscan')
            return {}

        import umap

        umap_dim = self.config.fragmentation_umap_dim
        # Only reduce if the embedding dimensionality exceeds the target
        if embeddings.shape[1] > umap_dim:
            logger.info(
                f'UMAP-reducing {embeddings.shape[1]}-D → {umap_dim}-D '
                f'for HDBSCAN fragmentation analysis'
            )
            reducer = umap.UMAP(
                n_components=umap_dim,
                n_neighbors=15,
                min_dist=0.0,
                metric='cosine',
                random_state=42,
            )
            reduced = reducer.fit_transform(embeddings)
        else:
            reduced = embeddings

        label_array = np.array(labels)
        report: dict[str, Any] = {}

        for lbl in np.unique(label_array):
            mask = label_array == lbl
            class_embs = reduced[mask]
            n_samples = len(class_embs)

            if n_samples < self.config.fragmentation_min_cluster_size * 2:
                report[lbl] = {
                    'n_clusters': 1,
                    'n_noise': 0,
                    'fragmented': False,
                    'cluster_sizes': [n_samples],
                    'skipped': True,
                    'reason': 'too_few_samples',
                }
                continue

            min_cluster = min(
                self.config.fragmentation_min_cluster_size, max(2, n_samples // 10)
            )
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster,
                metric='euclidean',
            )
            cluster_labels = clusterer.fit_predict(class_embs)

            unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
            n_noise = int((cluster_labels == -1).sum())
            cluster_sizes = [int((cluster_labels == c).sum()) for c in unique_clusters]

            report[lbl] = {
                'n_clusters': len(unique_clusters),
                'n_noise': n_noise,
                'fragmented': len(unique_clusters) > 2,
                'cluster_sizes': cluster_sizes,
                'skipped': False,
            }

        return report

    def detect_near_duplicates(
        self,
        embeddings: np.ndarray,
        labels: list[str],
        texts: list[str] | None = None,
    ) -> pd.DataFrame:
        """Find high-similarity text pairs that have conflicting labels.

        Near-duplicates (cosine similarity > threshold) with different labels are
        almost certainly labeling errors -- the same or very similar text should
        receive the same label.

        Args:
            embeddings: L2-normalized embedding matrix, shape (n, d).
            labels: Label for each sample, length n.
            texts: Optional original texts for display in output.

        Returns:
            DataFrame with one row per conflicting near-duplicate pair:
                - idx_a, idx_b: indices of the two samples
                - text_a, text_b: original texts (if provided)
                - label_a, label_b: their conflicting labels
                - similarity: cosine similarity score
                - label_conflict: True (always True in this output)

        Example:
            >>> df = analyzer.detect_near_duplicates(embeddings, labels, texts)
            >>> print(f"Found {len(df)} conflicting near-duplicates")
        """
        label_array = np.array(labels)
        sim_matrix = cosine_similarity(embeddings)
        # Zero out lower triangle and diagonal to avoid duplicate pairs
        sim_matrix = np.triu(sim_matrix, k=1)

        pairs = np.argwhere(sim_matrix >= self.config.duplicate_similarity_threshold)

        records = []
        for idx_a, idx_b in pairs:
            if label_array[idx_a] != label_array[idx_b]:
                record: dict[str, Any] = {
                    'idx_a': int(idx_a),
                    'idx_b': int(idx_b),
                    'label_a': label_array[idx_a],
                    'label_b': label_array[idx_b],
                    'similarity': float(sim_matrix[idx_a, idx_b]),
                    'label_conflict': True,
                }
                if texts:
                    record['text_a'] = texts[idx_a]
                    record['text_b'] = texts[idx_b]
                records.append(record)

        return pd.DataFrame(records).sort_values('similarity', ascending=False)

    def run_all(
        self,
        texts: list[str],
        labels: list[str],
    ) -> dict[str, Any]:
        """Run all embedding-space analyses and return combined results.

        Args:
            texts: Input texts.
            labels: Corresponding label for each text.

        Returns:
            Dictionary with keys:
                - embeddings: np.ndarray of encoded texts
                - intraclass_outliers: DataFrame from detect_intraclass_outliers
                - centroid_violations: DataFrame from detect_centroid_violations
                - cluster_fragmentation: dict from detect_cluster_fragmentation
                - near_duplicates: DataFrame from detect_near_duplicates
                - summary: high-level counts and flags

        Example:
            >>> results = analyzer.run_all(texts, labels)
            >>> violations = results["centroid_violations"]
        """
        logger.info(f'EmbeddingAnalyzer: analysing {len(texts)} samples')
        embeddings = self.embed_texts(texts)

        intraclass_df = self.detect_intraclass_outliers(embeddings, labels)
        centroid_df = self.detect_centroid_violations(embeddings, labels)
        fragmentation = self.detect_cluster_fragmentation(embeddings, labels)
        near_dup_df = self.detect_near_duplicates(embeddings, labels, texts)

        n_outliers = int(intraclass_df['is_outlier'].sum())
        n_violations = int(centroid_df['is_violation'].sum())
        n_fragmented = sum(1 for v in fragmentation.values() if v.get('fragmented'))
        n_near_dups = len(near_dup_df)

        logger.info(
            f'Embedding analysis complete: '
            f'{n_outliers} intra-class outliers, '
            f'{n_violations} centroid violations, '
            f'{n_fragmented} fragmented classes, '
            f'{n_near_dups} conflicting near-duplicates'
        )

        return {
            'embeddings': embeddings,
            'intraclass_outliers': intraclass_df,
            'centroid_violations': centroid_df,
            'cluster_fragmentation': fragmentation,
            'near_duplicates': near_dup_df,
            'summary': {
                'n_intraclass_outliers': n_outliers,
                'n_centroid_violations': n_violations,
                'n_fragmented_classes': n_fragmented,
                'n_conflicting_near_duplicates': n_near_dups,
            },
        }
