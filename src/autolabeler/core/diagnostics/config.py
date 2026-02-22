"""Configuration for the diagnostics module.

Each field maps directly to the diagnostics: block in dataset YAML configs.
All thresholds have sensible defaults and can be overridden per-dataset.
"""

from dataclasses import dataclass, field


_DEFAULT_SUSPICION_WEIGHTS: dict[str, float] = {
    'embedding_outlier': 0.25,
    'nli_mismatch': 0.25,
    'low_confidence': 0.20,
    'jury_disagreement': 0.15,
    'rationale_inconsistency': 0.15,
}


@dataclass
class DiagnosticsConfig:
    """Configuration for post-labeling diagnostic analysis.

    Controls which diagnostic modules run, their thresholds, and where
    results are written. Wire this into DatasetConfig to enable per-dataset
    diagnostics without changing pipeline code.

    Args:
        enabled: Master switch. If False, no diagnostics run.
        run_post_labeling: Auto-run diagnostics after label_dataframe() completes.
        embedding_provider: Backend for embedding computation. One of:
            - "local" (default): Sentence-transformers on-device inference.
            - "openai": OpenAI Embeddings API.
            - "openrouter": OpenRouter Embeddings API.
        embedding_model: Model identifier for the chosen provider.
            Local: "all-MiniLM-L6-v2" (default), any sentence-transformers model.
            OpenAI: "text-embedding-3-small", "text-embedding-3-large".
            OpenRouter: "openai/text-embedding-3-small", or any available model.
        embedding_api_key: Optional API key for openai/openrouter providers.
            Falls back to OPENAI_API_KEY or OPENROUTER_API_KEY env vars.
        nli_model: CrossEncoder model for NLI coherence scoring.
        outlier_z_threshold: Z-score above which a sample is flagged as an
            intra-class embedding outlier (default 2.0).
        lof_neighbors: Number of neighbours for Local Outlier Factor (default 20).
        fragmentation_min_cluster_size: Minimum cluster size for HDBSCAN
            fragmentation analysis (default 5).
        nli_entailment_threshold: Entailment score below which a (text, label)
            pair is flagged as potentially mismatched (default 0.5).
        duplicate_similarity_threshold: Cosine similarity above which two texts
            are treated as near-duplicates (default 0.95).
        batch_drift_kl_threshold: KL divergence above which a label-distribution
            shift between batches is flagged (default 0.1).
        suspicion_weights: Per-signal weights for composite suspicion scoring.
            Must sum to approximately 1.0.
        hypotheses_path: Path to YAML file mapping labels to NLI hypothesis
            strings. Required for NLI scoring. If None, NLI scoring is skipped.
        embedding_cache_dir: Directory for caching computed embeddings to disk.
        top_k_suspects: Maximum number of high-suspicion samples to surface
            in the audit recommendations.

    Example:
        >>> cfg = DiagnosticsConfig(enabled=True, run_post_labeling=True)
        >>> cfg.outlier_z_threshold
        2.0
    """

    enabled: bool = False
    run_post_labeling: bool = False

    # Embedding provider and model selection
    embedding_provider: str = 'local'
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_api_key: str | None = None
    nli_model: str = 'cross-encoder/nli-deberta-v3-large'

    # Embedding analysis thresholds
    outlier_z_threshold: float = 2.0
    lof_neighbors: int = 20
    fragmentation_min_cluster_size: int = 5

    # NLI thresholds
    nli_entailment_threshold: float = 0.5

    # Duplicate detection
    duplicate_similarity_threshold: float = 0.95

    # Batch / distribution shift
    batch_drift_kl_threshold: float = 0.1

    # Composite suspicion scoring
    suspicion_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_SUSPICION_WEIGHTS)
    )

    # NLI hypothesis definitions
    hypotheses_path: str | None = None

    # Caching
    embedding_cache_dir: str = '.cache/embeddings'

    # Audit output
    top_k_suspects: int = 100

    @classmethod
    def from_dict(cls, data: dict) -> 'DiagnosticsConfig':
        """Build a DiagnosticsConfig from a plain dictionary (e.g., parsed YAML).

        Args:
            data: Dictionary with diagnostics configuration keys.

        Returns:
            DiagnosticsConfig instance.

        Example:
            >>> cfg = DiagnosticsConfig.from_dict({"enabled": True, "outlier_z_threshold": 3.0})
        """
        weights = data.get('suspicion_weights', _DEFAULT_SUSPICION_WEIGHTS)
        return cls(
            enabled=data.get('enabled', False),
            run_post_labeling=data.get('run_post_labeling', False),
            embedding_provider=data.get('embedding_provider', 'local'),
            embedding_model=data.get(
                'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'
            ),
            embedding_api_key=data.get('embedding_api_key'),
            nli_model=data.get('nli_model', 'cross-encoder/nli-deberta-v3-large'),
            outlier_z_threshold=data.get('outlier_z_threshold', 2.0),
            lof_neighbors=data.get('lof_neighbors', 20),
            fragmentation_min_cluster_size=data.get('fragmentation_min_cluster_size', 5),
            nli_entailment_threshold=data.get('nli_entailment_threshold', 0.5),
            duplicate_similarity_threshold=data.get('duplicate_similarity_threshold', 0.95),
            batch_drift_kl_threshold=data.get('batch_drift_kl_threshold', 0.1),
            suspicion_weights=weights,
            hypotheses_path=data.get('hypotheses_path'),
            embedding_cache_dir=data.get('embedding_cache_dir', '.cache/embeddings'),
            top_k_suspects=data.get('top_k_suspects', 100),
        )
