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
class GapAnalysisConfig:
    """Configuration for LLM-powered gap analysis.

    Controls error clustering, LLM diagnosis, and synthetic data generation
    for identifying systematic weaknesses in the training data. Runs as step
    10 in the diagnostics pipeline and requires embedding results (step 1) to
    be available for UMAP+HDBSCAN clustering.

    Args:
        enabled: Master switch for gap analysis. Must be True in the config
            for the pipeline to run it. Can be overridden per-run via the
            ``--force-gap-analysis`` CLI flag in run_diagnostics.py.
        analysis_provider: LLM provider for gap diagnosis. Supported values
            match the jury providers: "google", "openai", "anthropic",
            "openrouter". Flash-tier models (gemini-2.5-flash, gpt-4o-mini)
            are recommended for cost efficiency.
        analysis_model: Model identifier for the gap analysis LLM.
        min_cluster_size: Minimum samples to form an error cluster. Lower
            values produce more granular clusters but risk noise. Must be >= 2.
        max_clusters: Cap on HDBSCAN clusters returned; smallest clusters
            beyond this limit are merged into the noise label (-1).
        representative_samples: Number of texts shown to the LLM per cluster.
            More context costs more tokens but gives the LLM better signal.
        generate_synthetic: If True, the LLM generates synthetic training
            examples for each gap cluster. These are saved to
            synthetic_examples.csv in the output directory.
        synthetic_per_cluster: Number of synthetic examples to generate per
            gap cluster.
        top_n_suspicious: How many top suspects (by suspicion score) to pull
            into the error pool before clustering. Larger pools give more
            coverage but make clustering slower.

    Example:
        >>> cfg = GapAnalysisConfig(enabled=True, generate_synthetic=False)
        >>> cfg.min_cluster_size
        5

        >>> # Minimal config for a quick diagnostic pass
        >>> cfg = GapAnalysisConfig(enabled=True, max_clusters=10, top_n_suspicious=200)
    """

    enabled: bool = False
    analysis_provider: str = 'google'
    analysis_model: str = 'gemini-2.5-flash'
    min_cluster_size: int = 5
    max_clusters: int = 20
    representative_samples: int = 10
    generate_synthetic: bool = True
    synthetic_per_cluster: int = 5
    top_n_suspicious: int = 500

    @classmethod
    def from_dict(cls, data: dict) -> 'GapAnalysisConfig':
        """Build a GapAnalysisConfig from a plain dictionary.

        Args:
            data: Dictionary with gap_analysis configuration keys.

        Returns:
            GapAnalysisConfig instance.

        Example:
            >>> cfg = GapAnalysisConfig.from_dict({"enabled": True, "max_clusters": 10})
        """
        return cls(
            enabled=data.get('enabled', False),
            analysis_provider=data.get('analysis_provider', 'google'),
            analysis_model=data.get('analysis_model', 'gemini-2.5-flash'),
            min_cluster_size=data.get('min_cluster_size', 5),
            max_clusters=data.get('max_clusters', 20),
            representative_samples=data.get('representative_samples', 10),
            generate_synthetic=data.get('generate_synthetic', True),
            synthetic_per_cluster=data.get('synthetic_per_cluster', 5),
            top_n_suspicious=data.get('top_n_suspicious', 500),
        )


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
        fragmentation_umap_dim: Dimensionality to UMAP-reduce embeddings to
            before running HDBSCAN. High-dim embeddings suffer from the curse
            of dimensionality, making density estimates unreliable and KD-trees
            slow. 5-10 dims is standard (cf. BERTopic). Default 10.
        nli_entailment_threshold: Entailment score below which a (text, label)
            pair is flagged as potentially mismatched (default 0.5).
        nli_batch_size: Batch size for CrossEncoder.predict(). Higher values
            use more GPU memory but are faster. 256 is a good GPU default;
            use 32-64 on CPU.
        nli_max_length: Max token length for CrossEncoder tokenizer. Sequences
            are truncated/padded to this length. Lower values speed up inference
            on short texts. None uses the model default (512 for DeBERTa).
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
        gap_analysis: Optional sub-config for LLM-powered gap analysis. When
            present and enabled, the gap analyzer clusters high-suspicion samples
            by topic, sends each cluster to an LLM for diagnosis, and optionally
            generates synthetic training examples. See GapAnalysisConfig.

    Example:
        >>> cfg = DiagnosticsConfig(enabled=True, run_post_labeling=True)
        >>> cfg.outlier_z_threshold
        2.0

        >>> # With gap analysis enabled
        >>> cfg = DiagnosticsConfig.from_dict({
        ...     "enabled": True,
        ...     "gap_analysis": {"enabled": True, "max_clusters": 15},
        ... })
        >>> cfg.gap_analysis.max_clusters
        15
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

    # UMAP dimensionality for HDBSCAN clustering (à la BERTopic).
    # Full embedding dim (e.g. 1536) is too high for density-based clustering.
    # 5-10 preserves local structure while making HDBSCAN fast and reliable.
    fragmentation_umap_dim: int = 10

    # NLI thresholds
    nli_entailment_threshold: float = 0.5

    # NLI inference batch size for CrossEncoder.predict().
    # GPU: 256 is a safe default; can go to 512+ with >=8 GB VRAM.
    # CPU: 32-64 recommended.
    nli_batch_size: int = 256

    # Max token length for CrossEncoder tokenizer. Shorter = faster for short texts.
    # None = model default (512 for DeBERTa). Set to 128 for texts under ~40 tokens.
    nli_max_length: int | None = None

    # Embedding batch size: texts per encode call / API request.
    # Local: 64 is safe for most hardware; increase if you have a GPU.
    # OpenAI/OpenRouter: 256-2048 is fine; OpenRouter's limit depends on the upstream model.
    embedding_batch_size: int = 64

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

    # Gap analysis (optional)
    gap_analysis: 'GapAnalysisConfig | None' = None

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
            fragmentation_umap_dim=data.get('fragmentation_umap_dim', 10),
            nli_entailment_threshold=data.get('nli_entailment_threshold', 0.5),
            nli_batch_size=data.get('nli_batch_size', 256),
            nli_max_length=data.get('nli_max_length'),
            embedding_batch_size=data.get('embedding_batch_size', 64),
            duplicate_similarity_threshold=data.get('duplicate_similarity_threshold', 0.95),
            batch_drift_kl_threshold=data.get('batch_drift_kl_threshold', 0.1),
            suspicion_weights=weights,
            hypotheses_path=data.get('hypotheses_path'),
            embedding_cache_dir=data.get('embedding_cache_dir', '.cache/embeddings'),
            top_k_suspects=data.get('top_k_suspects', 100),
            gap_analysis=(
                GapAnalysisConfig.from_dict(data['gap_analysis'])
                if data.get('gap_analysis')
                else None
            ),
        )
