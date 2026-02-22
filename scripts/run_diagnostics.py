#!/usr/bin/env python3
"""Standalone diagnostics runner for post-hoc analysis of labeled datasets.

Runs the full diagnostics pipeline on an already-labeled CSV without
re-running the labeling pipeline. Useful for auditing existing outputs
or for running specific diagnostic modules on demand.

Works with both pipeline-generated output (jury_labels, tier, confidence
columns) and plain human-labeled CSVs (just text + label). Modules that
need pipeline-specific columns are skipped automatically.

Usage:
    # Full diagnostics with all standard modules
    python scripts/run_diagnostics.py \\
        --dataset fed_headlines \\
        --labeled-data outputs/fed_headlines/labeled.csv \\
        --output outputs/fed_headlines/diagnostics/

    # Only embedding and distribution modules (fastest)
    python scripts/run_diagnostics.py \\
        --dataset fed_headlines \\
        --labeled-data outputs/fed_headlines/labeled.csv \\
        --output outputs/fed_headlines/diagnostics/ \\
        --enable embedding,distribution

    # Skip NLI (avoids large model download) for faster runs
    python scripts/run_diagnostics.py \\
        --dataset fed_headlines \\
        --labeled-data outputs/fed_headlines/labeled.csv \\
        --output outputs/fed_headlines/diagnostics/ \\
        --enable embedding,distribution,batch,rationale,report \\
        --top-k-suspects 200

    # Human-labeled data with non-standard column names
    python scripts/run_diagnostics.py \\
        --dataset fed_headlines \\
        --labeled-data datasets/fedspeak/human_labeled.csv \\
        --output outputs/fed_headlines/diagnostics_human/ \\
        --text-column headline \\
        --label-column hawk_dove \\
        --enable embedding,distribution,nli,report

    # Include LLM-powered gap analysis (requires gap_analysis.enabled in config)
    python scripts/run_diagnostics.py \\
        --dataset fed_headlines \\
        --labeled-data outputs/fed_headlines/labeled.csv \\
        --output outputs/fed_headlines/diagnostics/ \\
        --enable embedding,distribution,nli,report,gap_analysis

    # Force gap analysis for a one-off run without editing YAML
    python scripts/run_diagnostics.py \\
        --dataset fed_headlines \\
        --labeled-data outputs/fed_headlines/labeled.csv \\
        --output outputs/fed_headlines/diagnostics/ \\
        --enable embedding,report,gap_analysis \\
        --force-gap-analysis

    # Quick test on first 500 rows
    python scripts/run_diagnostics.py \\
        --dataset fed_headlines \\
        --labeled-data outputs/fed_headlines/labeled.csv \\
        --output outputs/fed_headlines/diagnostics_test/ \\
        --limit 500 \\
        --enable embedding,report
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Add parent to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.autolabeler.core.dataset_config import DatasetConfig
from src.autolabeler.core.diagnostics import run_diagnostics


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging format.

    Args:
        verbose: Enable DEBUG level logging.
    """
    logger.remove()
    log_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | '
        '<level>{message}</level>'
    )
    level = 'DEBUG' if verbose else 'INFO'
    logger.add(sys.stderr, format=log_format, level=level)


def main() -> int:
    """Parse CLI arguments and run diagnostics.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(
        description='Post-hoc diagnostics for labeled datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (must match configs/{dataset}.yaml)',
    )
    parser.add_argument(
        '--labeled-data',
        required=True,
        help='Path to labeled CSV file (output from run_labeling.py)',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for diagnostic results',
    )
    parser.add_argument(
        '--enable',
        default='embedding,distribution,nli,batch,rationale,report',
        help=(
            'Comma-separated list of modules to run. '
            'Options: embedding, distribution, nli, batch, rationale, report, gap_analysis. '
            'Default: all standard modules (gap_analysis must be opted in explicitly '
            'and also requires diagnostics.gap_analysis.enabled: true in config).'
        ),
    )
    parser.add_argument(
        '--top-k-suspects',
        type=int,
        default=None,
        help='Override top_k_suspects from config (default: use config value)',
    )
    parser.add_argument(
        '--hypotheses',
        default=None,
        help='Path to hypotheses.yaml for NLI scoring (overrides config)',
    )
    parser.add_argument(
        '--text-column',
        default=None,
        help=(
            'Name of the column containing text to analyse. '
            'Defaults to the text_column value in the dataset config (usually "text"). '
            'Use this for human-labeled data where the text column has a different name '
            '(e.g. "headline").'
        ),
    )
    parser.add_argument(
        '--label-column',
        default=None,
        help=(
            'Name of the column containing labels. Defaults to "label". '
            'Use this for human-labeled data where the label column has a different name '
            '(e.g. "hawk_dove").'
        ),
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit to first N rows (useful for testing)',
    )
    parser.add_argument(
        '--force-gap-analysis',
        action='store_true',
        help=(
            'Force-enable gap_analysis even if diagnostics.gap_analysis.enabled '
            'is False in the config. Useful for one-off CLI runs without editing YAML.'
        ),
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable DEBUG logging',
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load config
    config_path = Path(f'configs/{args.dataset}.yaml')
    if not config_path.exists():
        logger.error(f'Config not found: {config_path}')
        available = [f.stem for f in Path('configs').glob('*.yaml')]
        logger.error(f'Available configs: {available}')
        return 1

    config = DatasetConfig.from_yaml(config_path)
    logger.info(f'Loaded config for {config.name}')

    # Apply CLI overrides to diagnostics config
    if config.diagnostics is None:
        from src.autolabeler.core.diagnostics.config import DiagnosticsConfig
        config.diagnostics = DiagnosticsConfig(enabled=True)

    config.diagnostics.enabled = True

    if args.top_k_suspects is not None:
        config.diagnostics.top_k_suspects = args.top_k_suspects

    if args.hypotheses is not None:
        config.diagnostics.hypotheses_path = args.hypotheses

    if args.force_gap_analysis:
        from src.autolabeler.core.diagnostics.config import GapAnalysisConfig
        if config.diagnostics.gap_analysis is None:
            config.diagnostics.gap_analysis = GapAnalysisConfig(enabled=True)
        else:
            config.diagnostics.gap_analysis.enabled = True

    # Load labeled data
    labeled_path = Path(args.labeled_data)
    if not labeled_path.exists():
        logger.error(f'Labeled data not found: {labeled_path}')
        return 1

    labeled_df = pd.read_csv(labeled_path)
    logger.info(f'Loaded {len(labeled_df)} labeled samples from {labeled_path}')

    if args.limit:
        labeled_df = labeled_df.head(args.limit)
        logger.info(f'Limited to first {args.limit} rows')

    # Parse enabled modules
    enabled_modules = [m.strip() for m in args.enable.split(',') if m.strip()]
    valid_modules = {'embedding', 'distribution', 'nli', 'batch', 'rationale', 'report', 'gap_analysis'}
    invalid = set(enabled_modules) - valid_modules
    if invalid:
        logger.error(f'Unknown modules: {invalid}. Valid: {valid_modules}')
        return 1

    logger.info(f'Enabled modules: {enabled_modules}')
    logger.info(f'Output directory: {args.output}')

    # Resolve column names (CLI overrides > config > defaults)
    text_col = args.text_column or getattr(config, 'text_column', 'text')
    label_col = args.label_column or 'label'

    # Run diagnostics
    output_dir = Path(args.output)
    results = run_diagnostics(
        labeled_df=labeled_df,
        config=config,
        output_dir=output_dir,
        enabled_modules=enabled_modules,
        text_col=text_col,
        label_col=label_col,
    )

    # Print summary to console
    report = results.get('report', {})
    summary = report.get('summary', {})
    recommendations = report.get('recommendations', [])

    if summary:
        logger.info('--- Diagnostics Summary ---')
        logger.info(f'  Samples analyzed: {summary.get("n_samples", 0):,}')
        logger.info(f'  Recommendations: {summary.get("n_recommendations", 0)} ({summary.get("n_high_priority_recommendations", 0)} high priority)')
        logger.info(f'  Top suspects identified: {summary.get("total_suspects", 0)}')

        if recommendations:
            logger.info('--- Top Recommendations ---')
            for rec in recommendations[:5]:
                logger.info(f'  [{rec["priority"].upper()}] {rec["finding"]}')

    logger.info(f'Full report saved to {output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
