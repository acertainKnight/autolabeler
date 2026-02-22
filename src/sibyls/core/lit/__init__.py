"""LIT (Language Interpretability Tool) integration for sibyls.

Provides reusable wrappers to explore any distilled model trained from
the sibyls pipeline output interactively in the LIT browser UI.

Requires the optional ``lit`` dependency group::

    pip install 'sibyls[lit]'

Quick start::

    from sibyls.core.dataset_config import DatasetConfig
    from sibyls.core.lit import DistilledModelWrapper, create_lit_server

    config = DatasetConfig.from_yaml("configs/fed_headlines.yaml")

    # Load your model via your own factory function
    model, tokenizer = load_model("outputs/distilled/best.pt")

    wrapper = DistilledModelWrapper(
        model=model,
        tokenizer=tokenizer,
        config=config,
        embedding_layer="encoder.pooler",
        enable_gradients=True,
    )

    server = create_lit_server(config, wrapper, "outputs/labeled.csv")
    server.serve()  # opens http://localhost:4321

Public API:
    LabeledDataset         -- LIT Dataset built from pipeline output CSV
    DistilledModelWrapper  -- LIT Model wrapper for any PyTorch nn.Module
    JuryReferenceModel     -- Pseudo-model serving pre-computed jury labels
    create_lit_server      -- Factory that wires everything into a LIT Server
"""

from sibyls.core.lit.dataset import LabeledDataset
from sibyls.core.lit.model import DistilledModelWrapper, JuryReferenceModel
from sibyls.core.lit.server import create_lit_server

__all__ = [
    'LabeledDataset',
    'DistilledModelWrapper',
    'JuryReferenceModel',
    'create_lit_server',
]
