"""Post-labeling diagnostics: error detection and anomaly analysis.

This module provides a suite of complementary detectors that run on labeled
output data to surface potential labeling errors without requiring ground truth.

Quick start:
    >>> from autolabeler.core.diagnostics import run_diagnostics
    >>> from autolabeler.core.dataset_config import DatasetConfig
    >>> import pandas as pd
    >>> config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
    >>> labeled_df = pd.read_csv("outputs/fed_headlines/labeled.csv")
    >>> run_diagnostics(labeled_df, config, Path("outputs/fed_headlines/diagnostics"))

Module structure:
    config.py                -- DiagnosticsConfig dataclass
    embedding_providers.py   -- Pluggable embedding backends (local/OpenAI/OpenRouter)
    embedding_analyzer.py    -- Embedding-space outlier & violation detection
    nli_scorer.py            -- NLI-based coherence scoring
    distribution_diagnostics.py -- Distribution shift & bias detection
    rationale_analyzer.py    -- Reasoning consistency analysis
    batch_diagnostics.py     -- Temporal drift & cascade bias detection
    suspicion_scorer.py      -- Composite per-sample suspicion scoring
    active_discovery.py      -- Prioritised audit sample selection
    quality_report.py        -- Aggregate Markdown + JSON report generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .active_discovery import ActiveDiscovery
from .batch_diagnostics import BatchDiagnostics
from .config import DiagnosticsConfig
from .distribution_diagnostics import DistributionDiagnostics
from .embedding_analyzer import EmbeddingAnalyzer
from .embedding_providers import (
    BaseEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    get_embedding_provider,
)
from .nli_scorer import NLIScorer
from .quality_report import QualityReportGenerator
from .rationale_analyzer import RationaleAnalyzer
from .suspicion_scorer import SuspicionScorer

__all__ = [
    'DiagnosticsConfig',
    'BaseEmbeddingProvider',
    'LocalEmbeddingProvider',
    'OpenAIEmbeddingProvider',
    'OpenRouterEmbeddingProvider',
    'get_embedding_provider',
    'EmbeddingAnalyzer',
    'NLIScorer',
    'DistributionDiagnostics',
    'RationaleAnalyzer',
    'BatchDiagnostics',
    'SuspicionScorer',
    'ActiveDiscovery',
    'QualityReportGenerator',
    'run_diagnostics',
]


def run_diagnostics(
    labeled_df: pd.DataFrame,
    config: Any,  # DatasetConfig -- avoid circular import
    output_dir: Path,
    enabled_modules: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full diagnostics pipeline on a labeled DataFrame.

    Orchestrates all diagnostic modules, combines their results, computes
    composite suspicion scores, and generates a quality report.

    This is the main entry point for both the post-labeling pipeline hook and
    the standalone scripts/run_diagnostics.py CLI.

    Args:
        labeled_df: Output DataFrame from LabelingPipeline.label_dataframe().
            Expected columns: text (or configured text_column), label, tier,
            training_weight, agreement, jury_labels, jury_confidences.
        config: DatasetConfig instance. Uses config.diagnostics for settings.
            If config.diagnostics is None, uses DiagnosticsConfig() defaults.
        output_dir: Directory to write diagnostic output files.
        enabled_modules: Subset of modules to run, e.g. ['embedding', 'distribution'].
            Defaults to all modules. Valid values:
            'embedding', 'distribution', 'nli', 'batch', 'rationale', 'report'.

    Returns:
        Dictionary with keys per module name plus:
            - labeled_df: the original labeled DataFrame
            - scored_df: DataFrame with suspicion scores per sample
            - report: the full quality report dict

    Example:
        >>> results = run_diagnostics(
        ...     labeled_df, config, Path("outputs/diagnostics"),
        ...     enabled_modules=["embedding", "distribution"]
        ... )
        >>> top_suspects = results["scored_df"].head(20)
    """
    diag_config: DiagnosticsConfig = getattr(config, 'diagnostics', None) or DiagnosticsConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if enabled_modules is None:
        enabled_modules = ['embedding', 'distribution', 'nli', 'batch', 'rationale', 'report']

    logger.info(
        f'Starting diagnostics pipeline on {len(labeled_df)} samples '
        f'(modules: {enabled_modules})'
    )

    all_results: dict[str, Any] = {'labeled_df': labeled_df}

    # Determine text column
    dataset_name = getattr(config, 'name', 'dataset')
    text_col = getattr(config, 'text_column', 'text')
    if text_col not in labeled_df.columns:
        text_col = labeled_df.columns[0]
        logger.warning(f'text_column not found -- using first column: {text_col}')

    texts = labeled_df[text_col].astype(str).tolist()
    labels = labeled_df['label'].astype(str).tolist() if 'label' in labeled_df.columns else []

    # 1. Embedding analysis
    if 'embedding' in enabled_modules:
        try:
            emb_analyzer = EmbeddingAnalyzer(diag_config)
            all_results['embedding'] = emb_analyzer.run_all(texts, labels)
        except Exception as e:
            logger.error(f'Embedding analysis failed: {e}')
            all_results['embedding'] = {}

    # 2. Distribution diagnostics
    if 'distribution' in enabled_modules:
        try:
            dist_diag = DistributionDiagnostics(diag_config)
            all_results['distribution'] = dist_diag.run_all(labeled_df, text_col, 'label')
        except Exception as e:
            logger.error(f'Distribution diagnostics failed: {e}')
            all_results['distribution'] = {}

    # 3. NLI scoring (requires hypotheses file)
    if 'nli' in enabled_modules:
        hypotheses_path = diag_config.hypotheses_path
        if hypotheses_path is None:
            # Try default location
            hypotheses_path = f'prompts/{dataset_name}/hypotheses.yaml'

        hyp_file = Path(hypotheses_path)
        if hyp_file.exists():
            try:
                nli_scorer = NLIScorer(diag_config)
                hypotheses = nli_scorer.load_hypotheses(hyp_file)
                all_results['nli'] = nli_scorer.run_all(texts, labels, hypotheses)
            except Exception as e:
                logger.error(f'NLI scoring failed: {e}')
                all_results['nli'] = {}
        else:
            logger.warning(
                f'Hypotheses file not found at {hyp_file} -- skipping NLI scoring. '
                f'Create prompts/{dataset_name}/hypotheses.yaml to enable NLI.'
            )
            all_results['nli'] = {}

    # 4. Batch diagnostics
    if 'batch' in enabled_modules:
        try:
            batch_diag = BatchDiagnostics(diag_config)
            all_results['batch'] = batch_diag.run_all(labeled_df)
        except Exception as e:
            logger.error(f'Batch diagnostics failed: {e}')
            all_results['batch'] = {}

    # 5. Rationale analysis
    if 'rationale' in enabled_modules:
        try:
            rationale_analyzer = RationaleAnalyzer(diag_config)
            all_results['rationale'] = rationale_analyzer.run_all(labeled_df)
        except Exception as e:
            logger.error(f'Rationale analysis failed: {e}')
            all_results['rationale'] = {}

    # 6. Composite suspicion scoring
    scorer = SuspicionScorer(diag_config)
    try:
        scored_df = scorer.score_dataset(all_results)
        all_results['scored_df'] = scored_df

        # Save scored DataFrame
        scored_path = output_dir / 'suspicion_scores.csv'
        scored_df.drop(columns=['all_scores'], errors='ignore').to_csv(scored_path, index=False)
        logger.info(f'Suspicion scores saved to {scored_path}')

        # Save top suspects
        if not scored_df.empty and 'suspicion_score' in scored_df.columns:
            top_k = min(diag_config.top_k_suspects, len(scored_df))
            suspects_path = output_dir / 'top_suspects.csv'
            scored_df.nlargest(top_k, 'suspicion_score').to_csv(suspects_path, index=False)
            logger.info(f'Top {top_k} suspects saved to {suspects_path}')
    except Exception as e:
        logger.error(f'Suspicion scoring failed: {e}')
        all_results['scored_df'] = pd.DataFrame()

    # 7. Active discovery: quarantine reports
    discovery = ActiveDiscovery(diag_config)
    try:
        quarantine_reports = discovery.quarantine_report(
            labeled_df, all_results, all_results.get('scored_df')
        )
        all_results['quarantine_reports'] = quarantine_reports

        if quarantine_reports:
            import json
            qr_path = output_dir / 'quarantine_reports.json'
            with open(qr_path, 'w', encoding='utf-8') as f:
                json.dump(quarantine_reports, f, indent=2, default=str)
            logger.info(f'Quarantine reports saved to {qr_path}')
    except Exception as e:
        logger.error(f'Quarantine report generation failed: {e}')

    # 8. Quality report
    if 'report' in enabled_modules:
        try:
            reporter = QualityReportGenerator(diag_config)
            report = reporter.generate(
                all_results,
                output_dir=output_dir,
                formats=['markdown', 'json'],
                dataset_name=dataset_name,
            )
            all_results['report'] = report
        except Exception as e:
            logger.error(f'Quality report generation failed: {e}')
            all_results['report'] = {}

    logger.info(f'Diagnostics complete. All outputs saved to {output_dir}')
    return all_results
