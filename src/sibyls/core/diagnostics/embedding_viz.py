"""Interactive UMAP embedding visualization with Plotly (2-D and 3-D).

Projects high-dimensional embeddings to 2-D and 3-D via UMAP and renders
interactive Plotly scatter plots coloured by label. Hover text shows the
original headline and suspicion score (when available), making it easy to
visually inspect cluster separation and spot mislabeled outliers.

Both charts are saved as self-contained HTML files that can be opened in
any browser -- no notebook required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .config import DiagnosticsConfig

# Label colour palette: ordered from most dovish to most hawkish so
# adjacent colours on the ordinal scale are visually distinct.
_DEFAULT_COLOUR_MAP: dict[str, str] = {
    '-99': '#999999',  # irrelevant / unknown — grey
    '-2': '#1b4f72',   # strongly dovish — dark blue
    '-1': '#5dade2',   # dovish — light blue
    '0': '#aab7b8',    # neutral — grey-blue
    '1': '#e67e22',    # hawkish — orange
    '2': '#c0392b',    # strongly hawkish — red
}


def compute_umap(
    embeddings: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42,
) -> np.ndarray:
    """Project embeddings to lower dimensions using UMAP.

    Args:
        embeddings: L2-normalised embedding matrix, shape (n, d).
        n_components: Target dimensionality (2 or 3).
        n_neighbors: UMAP locality parameter. Higher = more global structure.
        min_dist: Minimum distance between points in the projection.
        metric: Distance metric for UMAP.
        random_state: Seed for reproducibility.

    Returns:
        Array of shape (n, n_components) with UMAP coordinates.

    Example:
        >>> coords_2d = compute_umap(embeddings, n_components=2)
        >>> coords_2d.shape
        (5000, 2)
        >>> coords_3d = compute_umap(embeddings, n_components=3)
        >>> coords_3d.shape
        (5000, 3)
    """
    import umap

    logger.info(
        f'Computing UMAP projection: {embeddings.shape[0]} samples, '
        f'd={embeddings.shape[1]} → {n_components}-D (n_neighbors={n_neighbors})'
    )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    logger.info(f'UMAP {n_components}-D projection complete')
    return coords


def _build_hover_text(
    texts: list[str],
    labels: list[str],
    suspicion_scores: pd.Series | None = None,
) -> list[str]:
    """Build per-sample hover HTML strings.

    Args:
        texts: Original texts, length n.
        labels: Labels, length n.
        suspicion_scores: Optional suspicion scores, length n.

    Returns:
        List of HTML hover strings, length n.
    """
    hover_lines: list[str] = []
    for i, (text, lbl) in enumerate(zip(texts, labels)):
        truncated = text[:120] + '...' if len(text) > 120 else text
        parts = [f'<b>Label:</b> {lbl}', f'<b>Text:</b> {truncated}']
        if suspicion_scores is not None:
            parts.append(f'<b>Suspicion:</b> {suspicion_scores.iloc[i]:.3f}')
        parts.append(f'<b>Index:</b> {i}')
        hover_lines.append('<br>'.join(parts))
    return hover_lines


def build_umap_figure_2d(
    coords: np.ndarray,
    labels: list[str],
    texts: list[str],
    *,
    suspicion_scores: pd.Series | None = None,
    colour_map: dict[str, str] | None = None,
    title: str = 'Embedding Clusters — 2-D UMAP',
) -> Any:
    """Build an interactive 2-D Plotly scatter of UMAP-projected embeddings.

    Args:
        coords: 2-D UMAP coordinates, shape (n, 2).
        labels: Label for each sample, length n.
        texts: Original text for each sample (shown on hover), length n.
        suspicion_scores: Optional per-sample suspicion scores for hover info.
        colour_map: Mapping from label string to hex colour. Defaults to a
            dovish-blue / hawkish-red palette for Fed sentiment labels.
        title: Chart title.

    Returns:
        plotly.graph_objects.Figure instance.

    Example:
        >>> fig = build_umap_figure_2d(coords, labels, texts)
        >>> fig.write_html("umap_2d.html")
    """
    import plotly.graph_objects as go

    if colour_map is None:
        colour_map = _DEFAULT_COLOUR_MAP

    hover_lines = _build_hover_text(texts, labels, suspicion_scores)
    unique_labels = sorted(set(labels), key=lambda x: (float(x) if _is_numeric(x) else 0, x))

    fig = go.Figure()
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        idx = np.where(mask)[0]
        fig.add_trace(go.Scattergl(
            x=coords[idx, 0],
            y=coords[idx, 1],
            mode='markers',
            name=f'Label {lbl}',
            marker={
                'color': colour_map.get(lbl, '#636efa'),
                'size': 5,
                'opacity': 0.7,
            },
            text=[hover_lines[i] for i in idx],
            hoverinfo='text',
        ))

    fig.update_layout(
        title=title,
        xaxis_title='UMAP-1',
        yaxis_title='UMAP-2',
        template='plotly_white',
        legend_title_text='Label',
        width=1100,
        height=750,
        hoverlabel={'bgcolor': 'white', 'font_size': 12},
    )
    return fig


def build_umap_figure_3d(
    coords: np.ndarray,
    labels: list[str],
    texts: list[str],
    *,
    suspicion_scores: pd.Series | None = None,
    colour_map: dict[str, str] | None = None,
    title: str = 'Embedding Clusters — 3-D UMAP',
) -> Any:
    """Build an interactive 3-D Plotly scatter of UMAP-projected embeddings.

    Supports orbit rotation, zoom, and pan. The third UMAP dimension often
    reveals cluster structure hidden in the 2-D projection (e.g. a neutral
    cluster sandwiched between hawkish and dovish groups).

    Args:
        coords: 3-D UMAP coordinates, shape (n, 3).
        labels: Label for each sample, length n.
        texts: Original text for each sample (shown on hover), length n.
        suspicion_scores: Optional per-sample suspicion scores for hover info.
        colour_map: Mapping from label string to hex colour.
        title: Chart title.

    Returns:
        plotly.graph_objects.Figure instance.

    Example:
        >>> fig = build_umap_figure_3d(coords_3d, labels, texts)
        >>> fig.write_html("umap_3d.html")
    """
    import plotly.graph_objects as go

    if colour_map is None:
        colour_map = _DEFAULT_COLOUR_MAP

    hover_lines = _build_hover_text(texts, labels, suspicion_scores)
    unique_labels = sorted(set(labels), key=lambda x: (float(x) if _is_numeric(x) else 0, x))

    fig = go.Figure()
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        idx = np.where(mask)[0]
        fig.add_trace(go.Scatter3d(
            x=coords[idx, 0],
            y=coords[idx, 1],
            z=coords[idx, 2],
            mode='markers',
            name=f'Label {lbl}',
            marker={
                'color': colour_map.get(lbl, '#636efa'),
                'size': 3,
                'opacity': 0.7,
            },
            text=[hover_lines[i] for i in idx],
            hoverinfo='text',
        ))

    fig.update_layout(
        title=title,
        scene={
            'xaxis_title': 'UMAP-1',
            'yaxis_title': 'UMAP-2',
            'zaxis_title': 'UMAP-3',
        },
        template='plotly_white',
        legend_title_text='Label',
        width=1100,
        height=800,
        hoverlabel={'bgcolor': 'white', 'font_size': 12},
    )
    return fig


def generate_umap_html(
    embeddings: np.ndarray,
    labels: list[str],
    texts: list[str],
    output_dir: Path | str,
    *,
    config: DiagnosticsConfig | None = None,
    suspicion_scores: pd.Series | None = None,
    colour_map: dict[str, str] | None = None,
    title: str = 'Embedding Clusters',
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> dict[str, Path]:
    """Generate both 2-D and 3-D interactive UMAP scatter plots.

    Runs UMAP twice (2-D and 3-D) and writes each as a standalone HTML file
    into ``output_dir``.

    Args:
        embeddings: L2-normalised embedding matrix, shape (n, d).
        labels: Per-sample label strings, length n.
        texts: Per-sample original texts, length n.
        output_dir: Directory to write HTML files into.
        config: Optional DiagnosticsConfig (reserved for future per-dataset
            UMAP parameters).
        suspicion_scores: Optional suspicion scores shown on hover.
        colour_map: Label-to-colour mapping.
        title: Base chart title (suffixed with "— 2-D" / "— 3-D").
        umap_n_neighbors: UMAP n_neighbors parameter.
        umap_min_dist: UMAP min_dist parameter.

    Returns:
        Dictionary with keys ``'2d'`` and ``'3d'`` mapping to the output Paths.

    Example:
        >>> paths = generate_umap_html(embeddings, labels, texts, "outputs/diag/")
        >>> print(f"2-D: {paths['2d']}, 3-D: {paths['3d']}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    umap_kwargs: dict[str, Any] = {
        'n_neighbors': umap_n_neighbors,
        'min_dist': umap_min_dist,
    }
    fig_kwargs: dict[str, Any] = {
        'labels': labels,
        'texts': texts,
        'suspicion_scores': suspicion_scores,
        'colour_map': colour_map,
    }

    # 2-D
    coords_2d = compute_umap(embeddings, n_components=2, **umap_kwargs)
    fig_2d = build_umap_figure_2d(coords_2d, **fig_kwargs, title=f'{title} — 2-D UMAP')
    path_2d = output_dir / 'umap_clusters_2d.html'
    fig_2d.write_html(str(path_2d), include_plotlyjs='cdn')
    logger.info(f'2-D UMAP saved to {path_2d}')

    # 3-D
    coords_3d = compute_umap(embeddings, n_components=3, **umap_kwargs)
    fig_3d = build_umap_figure_3d(coords_3d, **fig_kwargs, title=f'{title} — 3-D UMAP')
    path_3d = output_dir / 'umap_clusters_3d.html'
    fig_3d.write_html(str(path_3d), include_plotlyjs='cdn')
    logger.info(f'3-D UMAP saved to {path_3d}')

    return {'2d': path_2d, '3d': path_3d}


def _is_numeric(s: str) -> bool:
    """Check if a string can be parsed as a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False
