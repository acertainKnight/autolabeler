#!/usr/bin/env python3
"""Launch the LIT (Language Interpretability Tool) server for a distilled model.

The script dynamically imports a user-supplied ``load_model()`` factory
function, wraps the result in DistilledModelWrapper, and launches the LIT
browser UI pre-loaded with the labeled pipeline output.

Model loader contract
---------------------
Create a small Python file that exposes a ``load_model`` function:

    # my_loader.py
    import torch
    from my_project.model import MyClassifier
    from transformers import AutoTokenizer

    def load_model(checkpoint_path: str, device: str = "cpu"):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = MyClassifier(num_labels=6)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model, tokenizer

Usage
-----
# Minimal (auto-detects jury columns and uses defaults from YAML):
python scripts/run_lit.py \\
    --config configs/fed_headlines.yaml \\
    --loader my_loader.py \\
    --checkpoint outputs/distilled/best.pt \\
    --data outputs/fed_headlines/labeled.csv

# Full options:
python scripts/run_lit.py \\
    --config configs/fed_headlines.yaml \\
    --loader my_loader.py \\
    --checkpoint outputs/distilled/best.pt \\
    --data outputs/fed_headlines/labeled.csv \\
    --embedding-layer encoder.pooler \\
    --enable-gradients \\
    --enable-attention \\
    --include-jury \\
    --max-rows 5000 \\
    --port 4321 \\
    --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

from loguru import logger


def _load_factory(loader_path: str) -> types.ModuleType:
    """Dynamically import the user's loader module from a file path.

    Args:
        loader_path: Path to a Python file containing a ``load_model`` function.

    Returns:
        The imported module object.

    Raises:
        FileNotFoundError: If the loader file does not exist.
        AttributeError: If the module does not expose a ``load_model`` function.
    """
    path = Path(loader_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Loader file not found: {path}")

    spec = importlib.util.spec_from_file_location('_lit_loader', path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, 'load_model'):
        raise AttributeError(
            f"Loader file '{path}' must define a "
            f"'load_model(checkpoint_path, device) -> (model, tokenizer)' function."
        )

    return module


def _resolve_lit_config(
    config: 'DatasetConfig',  # noqa: F821
    args: argparse.Namespace,
) -> dict:
    """Merge YAML ``lit:`` defaults with CLI flags (CLI wins on conflict).

    Args:
        config: Loaded DatasetConfig, possibly with a ``lit_config`` attribute.
        args: Parsed CLI arguments.

    Returns:
        Dict of resolved settings for DistilledModelWrapper and the server.
    """
    # Pull YAML defaults (if the config extension was applied)
    yaml_lit = getattr(config, 'lit_config', {}) or {}

    return {
        'embedding_layer': args.embedding_layer or yaml_lit.get('embedding_layer'),
        'enable_gradients': args.enable_gradients or yaml_lit.get('enable_gradients', False),
        'enable_attention': args.enable_attention or yaml_lit.get('enable_attention', False),
        'port': args.port or yaml_lit.get('port', 4321),
    }


def setup_logging(verbose: bool) -> None:
    """Configure loguru for the script.

    Args:
        verbose: If True, sets DEBUG level.
    """
    logger.remove()
    fmt = (
        '<green>{time:HH:mm:ss}</green> | '
        '<level>{level: <8}</level> | '
        '<level>{message}</level>'
    )
    logger.add(sys.stderr, format=fmt, level='DEBUG' if verbose else 'INFO')


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments.

    Returns:
        Parsed argparse Namespace.
    """
    parser = argparse.ArgumentParser(
        description='Launch LIT for a distilled autolabeler model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        '--config',
        required=True,
        help='Path to DatasetConfig YAML (e.g. configs/fed_headlines.yaml)',
    )
    parser.add_argument(
        '--loader',
        required=True,
        help=(
            'Path to a Python file with a '
            'load_model(checkpoint_path, device) -> (model, tokenizer) function'
        ),
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to the PyTorch checkpoint file passed to load_model()',
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Path to labeled CSV (output from run_labeling.py)',
    )

    # Optional interpretability flags
    parser.add_argument(
        '--embedding-layer',
        default=None,
        help=(
            'Dot-path to the layer used for embedding extraction '
            '(e.g. "encoder.pooler"). Enables UMAP projector in LIT.'
        ),
    )
    parser.add_argument(
        '--enable-gradients',
        action='store_true',
        default=False,
        help='Compute token-level gradient saliency (requires autograd)',
    )
    parser.add_argument(
        '--enable-attention',
        action='store_true',
        default=False,
        help='Expose attention heads in LIT (model must support output_attentions)',
    )

    # Jury comparison
    jury_group = parser.add_mutually_exclusive_group()
    jury_group.add_argument(
        '--include-jury',
        action='store_true',
        default=None,
        dest='include_jury',
        help='Always include the JuryReferenceModel for comparison',
    )
    jury_group.add_argument(
        '--no-jury',
        action='store_false',
        dest='include_jury',
        help='Never include the JuryReferenceModel',
    )

    # Server settings
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='LIT server port (default: 4321, or value from YAML lit: section)',
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Limit number of rows loaded into LIT (useful for large datasets)',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='PyTorch device string (default: cpu)',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose DEBUG logging',
    )

    return parser.parse_args()


def main() -> int:
    """Entry point: load model, build LIT server, serve.

    Returns:
        Exit code (0 = success).
    """
    args = parse_args()
    setup_logging(args.verbose)

    # ------------------------------------------------------------------ #
    # 1. Load DatasetConfig
    # ------------------------------------------------------------------ #
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from autolabeler.core.dataset_config import DatasetConfig

    logger.info(f'Loading DatasetConfig from {args.config}')
    config = DatasetConfig.from_yaml(args.config)
    logger.info(f'Dataset: {config.name} | labels: {config.labels}')

    # ------------------------------------------------------------------ #
    # 2. Resolve final settings (YAML defaults + CLI overrides)
    # ------------------------------------------------------------------ #
    settings = _resolve_lit_config(config, args)
    port: int = settings['port']

    # ------------------------------------------------------------------ #
    # 3. Load user model via factory function
    # ------------------------------------------------------------------ #
    logger.info(f'Importing loader from {args.loader}')
    loader_module = _load_factory(args.loader)

    logger.info(f'Loading model from checkpoint: {args.checkpoint}')
    model, tokenizer = loader_module.load_model(args.checkpoint, device=args.device)
    logger.info('Model loaded successfully')

    # ------------------------------------------------------------------ #
    # 4. Build DistilledModelWrapper
    # ------------------------------------------------------------------ #
    from autolabeler.core.lit.model import DistilledModelWrapper

    wrapper = DistilledModelWrapper(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=args.device,
        embedding_layer=settings['embedding_layer'],
        enable_gradients=settings['enable_gradients'],
        enable_attention=settings['enable_attention'],
    )

    # ------------------------------------------------------------------ #
    # 5. Build and serve LIT server
    # ------------------------------------------------------------------ #
    from autolabeler.core.lit.server import create_lit_server

    server = create_lit_server(
        config=config,
        model_wrapper=wrapper,
        data_path=args.data,
        include_jury=args.include_jury,
        max_rows=args.max_rows,
        port=port,
    )

    logger.info(f'Starting LIT UI at http://localhost:{port}')
    server.serve()
    return 0


if __name__ == '__main__':
    sys.exit(main())
