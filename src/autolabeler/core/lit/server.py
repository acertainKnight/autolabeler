"""LIT server factory for autolabeler distilled models.

Provides a single create_lit_server() entry point that wires together a
LabeledDataset, a DistilledModelWrapper, and (optionally) a
JuryReferenceModel into a ready-to-serve LIT Server.

Usage (programmatic):
    >>> from autolabeler.core.dataset_config import DatasetConfig
    >>> from autolabeler.core.lit.server import create_lit_server
    >>> from autolabeler.core.lit.model import DistilledModelWrapper
    >>>
    >>> config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")
    >>> wrapper = DistilledModelWrapper(model, tokenizer, config)
    >>> server = create_lit_server(config, wrapper, "outputs/labeled.csv")
    >>> server.serve()

Usage (via CLI):
    python scripts/run_lit.py --config ... --loader ... --checkpoint ... --data ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

try:
    import lit_nlp
    from lit_nlp import dev_server
except ImportError as exc:
    raise ImportError(
        "lit-nlp is not installed. Install it with: pip install 'autolabeler[lit]'"
    ) from exc

from autolabeler.core.lit.dataset import LabeledDataset
from autolabeler.core.lit.model import DistilledModelWrapper, JuryReferenceModel


# Columns whose presence in the CSV indicates pre-computed jury predictions
_JURY_INDICATOR_COLUMNS: frozenset[str] = frozenset(
    {'soft_label', 'jury_labels', 'jury_confidences', 'agreement', 'tier'}
)


def _has_jury_columns(data_path: Path) -> bool:
    """Peek at the CSV header to check if jury prediction columns exist.

    Args:
        data_path: Path to the labeled CSV.

    Returns:
        True if at least one jury-indicator column is present.
    """
    import pandas as pd
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    return bool(_JURY_INDICATOR_COLUMNS & set(header))


def create_lit_server(
    config: Any,  # DatasetConfig
    model_wrapper: DistilledModelWrapper,
    data_path: str | Path,
    *,
    include_jury: bool | None = None,
    max_rows: int | None = None,
    port: int = 4321,
    warm_start: float = 0.0,
    **server_kwargs: Any,
) -> dev_server.Server:
    """Build and return a LIT Server wired to the autolabeler pipeline output.

    The server is returned but NOT started -- call ``.serve()`` on the returned
    object to open the browser UI, or use the server programmatically without
    starting HTTP.

    Args:
        config: DatasetConfig for the current labeling task.
        model_wrapper: Fully-constructed DistilledModelWrapper.
        data_path: Path to the labeled CSV produced by the labeling pipeline.
        include_jury: Whether to include a JuryReferenceModel for comparison.
            If None (default), auto-detected from the CSV columns.
        max_rows: Row limit passed to LabeledDataset. None = all rows.
        port: Port for the LIT development server.
        warm_start: Fraction of dataset to pre-cache predictions on startup
            (passed to LIT Server as ``warm_start``). 0.0 = lazy caching.
        **server_kwargs: Additional keyword arguments forwarded to
            ``lit_nlp.dev_server.Server``.

    Returns:
        Configured ``lit_nlp.dev_server.Server`` instance (not yet serving).

    Example:
        >>> server = create_lit_server(cfg, wrapper, "outputs/labeled.csv")
        >>> server.serve()
    """
    data_path = Path(data_path)

    # --- Dataset ---
    logger.info(f"Building LabeledDataset from {data_path}")
    dataset = LabeledDataset(data_path, config, max_rows=max_rows)

    datasets: dict[str, LabeledDataset] = {config.name: dataset}

    # --- Models ---
    models: dict[str, Any] = {'distilled': model_wrapper}

    # Auto-detect jury columns unless caller was explicit
    should_include_jury = (
        include_jury
        if include_jury is not None
        else _has_jury_columns(data_path)
    )

    if should_include_jury:
        logger.info("Jury columns detected -- adding JuryReferenceModel")
        jury_model = JuryReferenceModel(str(data_path), config)
        models['llm_jury'] = jury_model
    else:
        logger.info(
            "No jury columns found (or include_jury=False) -- "
            "loading distilled model only"
        )

    # --- Server ---
    logger.info(
        f"Creating LIT server | port={port} | "
        f"models={list(models.keys())} | datasets={list(datasets.keys())}"
    )

    server = dev_server.Server(
        models=models,
        datasets=datasets,
        port=port,
        warm_start=warm_start,
        **server_kwargs,
    )

    logger.info(
        f"LIT server ready. Call server.serve() to open the UI at "
        f"http://localhost:{port}"
    )
    return server
