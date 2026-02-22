"""Post-labeling diagnostics: error detection and anomaly analysis.

This module provides a suite of complementary detectors that run on labeled
output data to surface potential labeling errors without requiring ground truth.

Quick start:
    >>> from sibyls.core.diagnostics import run_diagnostics
    >>> from sibyls.core.dataset_config import DatasetConfig
    >>> import pandas as pd
    >>> config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
    >>> labeled_df = pd.read_csv("outputs/fed_headlines/labeled.csv")
    >>> run_diagnostics(labeled_df, config, Path("outputs/fed_headlines/diagnostics"))

Module structure:
    config.py                -- DiagnosticsConfig + GapAnalysisConfig dataclasses
    embedding_providers.py   -- Pluggable embedding backends (local/OpenAI/OpenRouter)
    embedding_analyzer.py    -- Embedding-space outlier & violation detection
    nli_scorer.py            -- NLI-based coherence scoring
    distribution_diagnostics.py -- Distribution shift & bias detection
    rationale_analyzer.py    -- Reasoning consistency analysis
    batch_diagnostics.py     -- Temporal drift & cascade bias detection
    suspicion_scorer.py      -- Composite per-sample suspicion scoring
    active_discovery.py      -- Prioritised audit sample selection
    quality_report.py        -- Aggregate Markdown + JSON report generation
    gap_analyzer.py          -- LLM-powered error clustering & gap diagnosis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .active_discovery import ActiveDiscovery
from .batch_diagnostics import BatchDiagnostics
from .config import DiagnosticsConfig, GapAnalysisConfig
from .distribution_diagnostics import DistributionDiagnostics
from .embedding_analyzer import EmbeddingAnalyzer
from .embedding_providers import (
    BaseEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    get_embedding_provider,
)
from .embedding_viz import generate_umap_html
from .gap_analyzer import GapAnalyzer
from .nli_scorer import NLIScorer
from .quality_report import QualityReportGenerator
from .rationale_analyzer import RationaleAnalyzer
from .suspicion_scorer import SuspicionScorer

__all__ = [
    'DiagnosticsConfig',
    'GapAnalysisConfig',
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
    'GapAnalyzer',
    'generate_umap_html',
    'run_diagnostics',
]


def run_diagnostics(
    labeled_df: pd.DataFrame,
    config: Any,  # DatasetConfig -- avoid circular import
    output_dir: Path,
    enabled_modules: list[str] | None = None,
    text_col: str | None = None,
    label_col: str | None = None,
) -> dict[str, Any]:
    """Run the full diagnostics pipeline on a labeled DataFrame.

    Orchestrates all diagnostic modules, combines their results, computes
    composite suspicion scores, and generates a quality report.

    This is the main entry point for both the post-labeling pipeline hook and
    the standalone scripts/run_diagnostics.py CLI.

    Works with both LLM-labeled pipeline output (with jury_labels, jury_confidences,
    tier, etc.) and human-labeled data (text + label columns only). Modules that
    require pipeline-specific columns are skipped gracefully when those columns
    are absent.

    Args:
        labeled_df: Labeled DataFrame. For pipeline output, expected columns:
            text, label, tier, training_weight, agreement, jury_labels,
            jury_confidences. For human-labeled data, only text and label
            columns are required (use text_col/label_col to specify names).
        config: DatasetConfig instance. Uses config.diagnostics for settings.
            If config.diagnostics is None, uses DiagnosticsConfig() defaults.
        output_dir: Directory to write diagnostic output files.
        enabled_modules: Subset of modules to run, e.g. ['embedding', 'distribution'].
            Defaults to all standard modules. Valid values:
            'embedding', 'distribution', 'nli', 'batch', 'rationale', 'report',
            'gap_analysis'. Note: gap_analysis is not included by default and
            must be opted in explicitly; it also requires the gap_analysis.enabled
            flag in config (or --force-gap-analysis from the CLI).
        text_col: Name of the text column. Defaults to config.text_column or 'text'.
        label_col: Name of the label column. Defaults to 'label'.

    Returns:
        Dictionary with keys per module name plus:
            - labeled_df: the working DataFrame (with standardised column names)
            - scored_df: DataFrame with suspicion scores per sample
            - report: the full quality report dict
            - gap_analysis: gap cluster summaries, synthetic examples (if enabled)

    Example:
        >>> results = run_diagnostics(
        ...     labeled_df, config, Path("outputs/diagnostics"),
        ...     enabled_modules=["embedding", "distribution"]
        ... )
        >>> top_suspects = results["scored_df"].head(20)

        >>> # Human-labeled data with custom column names
        >>> results = run_diagnostics(
        ...     human_df, config, Path("outputs/diagnostics"),
        ...     text_col="headline", label_col="hawk_dove",
        ... )

        >>> # Include gap analysis (requires embedding module for cached vectors)
        >>> results = run_diagnostics(
        ...     labeled_df, config, Path("outputs/diagnostics"),
        ...     enabled_modules=["embedding", "report", "gap_analysis"],
        ... )
    """
    diag_config: DiagnosticsConfig = getattr(config, 'diagnostics', None) or DiagnosticsConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if enabled_modules is None:
        enabled_modules = ['embedding', 'distribution', 'nli', 'batch', 'rationale', 'report']

    # Resolve text and label column names
    if text_col is None:
        text_col = getattr(config, 'text_column', 'text')
    if label_col is None:
        label_col = 'label'

    # Standardise column names so all downstream modules can use 'text' and 'label'
    working_df = labeled_df.copy()
    col_renames: dict[str, str] = {}
    if text_col != 'text' and text_col in working_df.columns:
        col_renames[text_col] = 'text'
    if label_col != 'label' and label_col in working_df.columns:
        col_renames[label_col] = 'label'
    if col_renames:
        working_df = working_df.rename(columns=col_renames)
        logger.info(f'Column remapping: {col_renames}')

    # Drop rows with missing labels (e.g. irrelevant headlines with no hawk_dove)
    if 'label' in working_df.columns:
        n_before = len(working_df)
        working_df = working_df.dropna(subset=['label']).reset_index(drop=True)
        n_dropped = n_before - len(working_df)
        if n_dropped > 0:
            logger.info(f'Dropped {n_dropped} rows with missing labels ({n_before} -> {len(working_df)})')

        # Normalize label strings: float-like values (e.g. "-2.0") become clean
        # ints ("-2") so they match hypotheses.yaml keys and config label lists.
        def _normalize_label(val: object) -> str:
            s = str(val)
            try:
                f = float(s)
                if f == int(f):
                    return str(int(f))
            except (ValueError, OverflowError):
                pass
            return s

        raw_labels = working_df['label'].astype(str)
        normalized = raw_labels.map(_normalize_label)
        if not raw_labels.equals(normalized):
            logger.info(f'Label normalization: {sorted(raw_labels.unique())} -> {sorted(normalized.unique())}')
        working_df['label'] = normalized

    # Detect data source: pipeline output vs human-labeled
    has_jury = 'jury_labels' in working_df.columns
    has_reasoning = 'reasoning' in working_df.columns
    has_tier = 'tier' in working_df.columns
    source_type = 'pipeline' if has_jury else 'human'
    logger.info(f'Data source detected: {source_type} (jury_labels={has_jury}, reasoning={has_reasoning}, tier={has_tier})')

    logger.info(
        f'Starting diagnostics pipeline on {len(working_df)} samples '
        f'(modules: {enabled_modules})'
    )

    all_results: dict[str, Any] = {'labeled_df': working_df}

    # Determine text column (should be 'text' after remapping)
    dataset_name = getattr(config, 'name', 'dataset')
    resolved_text_col = 'text' if 'text' in working_df.columns else working_df.columns[0]
    if resolved_text_col != 'text':
        logger.warning(f'text column not found after remapping -- using first column: {resolved_text_col}')

    texts = working_df[resolved_text_col].astype(str).tolist()
    labels = working_df['label'].astype(str).tolist() if 'label' in working_df.columns else []

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
            all_results['distribution'] = dist_diag.run_all(working_df, 'text', 'label')
        except Exception as e:
            logger.error(f'Distribution diagnostics failed: {e}')
            all_results['distribution'] = {}

    # 3. NLI scoring (requires hypotheses file)
    if 'nli' in enabled_modules:
        hypotheses_path = diag_config.hypotheses_path
        if hypotheses_path is None:
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

    # 4. Batch diagnostics (requires jury_labels for model drift)
    if 'batch' in enabled_modules:
        if not has_jury:
            logger.info('Skipping batch diagnostics: jury_labels column not present (human-labeled data)')
            all_results['batch'] = {}
        else:
            try:
                batch_diag = BatchDiagnostics(diag_config)
                all_results['batch'] = batch_diag.run_all(working_df)
            except Exception as e:
                logger.error(f'Batch diagnostics failed: {e}')
                all_results['batch'] = {}

    # 5. Rationale analysis (requires reasoning column)
    if 'rationale' in enabled_modules:
        if not has_reasoning:
            logger.info('Skipping rationale analysis: reasoning column not present (human-labeled data)')
            all_results['rationale'] = {}
        else:
            try:
                rationale_analyzer = RationaleAnalyzer(diag_config)
                all_results['rationale'] = rationale_analyzer.run_all(working_df)
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

    # 7. UMAP visualisation (requires embedding results)
    emb_results = all_results.get('embedding', {})
    embeddings = emb_results.get('embeddings')
    if embeddings is not None and 'embedding' in enabled_modules:
        try:
            suspicion_col = None
            scored_df = all_results.get('scored_df')
            if scored_df is not None and 'suspicion_score' in scored_df.columns:
                suspicion_col = scored_df['suspicion_score']

            generate_umap_html(
                embeddings=embeddings,
                labels=labels,
                texts=texts,
                output_dir=output_dir,
                config=diag_config,
                suspicion_scores=suspicion_col,
                title=f'{dataset_name} — Embedding Clusters',
            )
        except Exception as e:
            logger.error(f'UMAP visualisation failed: {e}')

    # 8. Active discovery: quarantine reports  
    discovery = ActiveDiscovery(diag_config)
    try:
        quarantine_reports = discovery.quarantine_report(
            working_df, all_results, all_results.get('scored_df')
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

    # 9. Quality report
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

    # 10. Gap analysis (LLM-powered; runs after suspicion scoring)
    if 'gap_analysis' in enabled_modules:
        gap_cfg = getattr(diag_config, 'gap_analysis', None)
        if gap_cfg is None or not gap_cfg.enabled:
            logger.info(
                'Skipping gap analysis: gap_analysis.enabled is False. '
                'Set diagnostics.gap_analysis.enabled: true in your config or pass '
                '--enable gap_analysis via CLI to force-enable.'
            )
            all_results['gap_analysis'] = {}
        else:
            try:
                from .gap_analyzer import GapAnalyzer

                # Load hypotheses as label definitions if available
                label_defs: dict[str, str] | None = None
                hyp_file = Path(diag_config.hypotheses_path or f'prompts/{dataset_name}/hypotheses.yaml')
                if hyp_file.exists():
                    import yaml
                    with open(hyp_file, encoding='utf-8') as _f:
                        hyp_data = yaml.safe_load(_f)
                    label_defs = {str(k): str(v) for k, v in (hyp_data.get('labels') or {}).items()}

                gap_analyzer = GapAnalyzer(diag_config)
                all_results['gap_analysis'] = gap_analyzer.run_all(
                    labeled_df=working_df,
                    diagnostic_results=all_results,
                    output_dir=output_dir,
                    label_definitions=label_defs,
                    dataset_name=dataset_name,
                )
            except Exception as e:
                logger.error(f'Gap analysis failed: {e}')
                all_results['gap_analysis'] = {}

    logger.info(f'Diagnostics complete. All outputs saved to {output_dir}')
    return all_results
