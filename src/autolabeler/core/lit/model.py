"""LIT Model wrappers for distilled and jury reference models.

Two wrappers are provided:

DistilledModelWrapper
    Wraps any user-supplied PyTorch nn.Module + tokenizer.  Uses forward hooks
    to extract embeddings from a named layer without modifying the model code.
    Optionally computes token-level gradients (for gradient-based saliency) and
    exposes attention heads -- enabling the full suite of LIT interpretability
    features out of the box.

JuryReferenceModel
    A lightweight "pseudo-model" that serves pre-computed LLM jury predictions
    straight from the labeled CSV rather than running any live inference.  Load
    it alongside DistilledModelWrapper to get side-by-side comparison in LIT's
    Compare mode (distilled model vs. LLM jury vs. human labels).
"""

from __future__ import annotations

import json
from operator import attrgetter
from typing import Any, Iterable

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install it with: pip install torch"
    ) from exc

try:
    from lit_nlp.api import model as lit_model
    from lit_nlp.api import types as lit_types
except ImportError as exc:
    raise ImportError(
        "lit-nlp is not installed. Install it with: pip install 'autolabeler[lit]'"
    ) from exc


def _resolve_layer(model: nn.Module, dotpath: str) -> nn.Module:
    """Resolve a dot-separated layer path to the actual nn.Module.

    Args:
        model: Root PyTorch module.
        dotpath: Dot-separated path, e.g. "encoder.layers.11" or "pooler".

    Returns:
        The nn.Module at that path.

    Raises:
        AttributeError: If any segment in the path does not exist.

    Example:
        >>> layer = _resolve_layer(model, "encoder.layer.11")
    """
    return attrgetter(dotpath)(model)


class DistilledModelWrapper(lit_model.BatchedModel):
    """LIT wrapper for any distilled PyTorch classification model.

    Handles tokenization, batched forward passes, and optional embedding /
    gradient / attention extraction via registered forward hooks.  All
    interpretability outputs are declared in output_spec() so LIT automatically
    enables the relevant visualisation panels.

    The model and tokenizer are supplied by the user's factory function rather
    than being constructed here, keeping this wrapper architecture-agnostic.

    Args:
        model: Trained PyTorch nn.Module in eval mode.
        tokenizer: Tokenizer whose __call__ returns input_ids, attention_mask,
            etc. (HuggingFace-compatible interface).
        config: DatasetConfig for the current dataset.
        device: PyTorch device string, e.g. "cpu" or "cuda".
        embedding_layer: Dot-path to the layer whose output is used as the
            sentence embedding (e.g. "encoder.pooler" or "encoder.layer.11").
            If None, embedding outputs are not exposed in LIT.
        enable_gradients: If True, compute token-level gradient norms and
            token embeddings for integrated-gradients saliency.
        enable_attention: If True, collect all AttentionHead outputs and expose
            them in the spec.  Requires the model to return
            ``attentions`` in its forward output when
            ``output_attentions=True`` is passed.
        max_minibatch_size: Maximum batch size for a single forward pass.

    Example:
        >>> wrapper = DistilledModelWrapper(
        ...     model=my_model, tokenizer=my_tokenizer, config=cfg,
        ...     embedding_layer="encoder.pooler", enable_gradients=True,
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,  # DatasetConfig
        device: str = 'cpu',
        embedding_layer: str | None = None,
        enable_gradients: bool = False,
        enable_attention: bool = False,
        max_minibatch_size: int = 32,
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._tokenizer = tokenizer
        self._config = config
        self._device = torch.device(device)
        self._embedding_layer = embedding_layer
        self._enable_gradients = enable_gradients
        self._enable_attention = enable_attention
        self._max_minibatch_size = max_minibatch_size
        self._label_vocab: list[str] = [str(l) for l in config.labels]
        self._text_col: str = config.text_column

        # Register embedding hook if requested
        self._embedding_hook_handle = None
        self._last_embeddings: np.ndarray | None = None
        if embedding_layer:
            self._register_embedding_hook(embedding_layer)

        logger.info(
            f"DistilledModelWrapper ready | labels={self._label_vocab} | "
            f"embed_layer={embedding_layer} | grads={enable_gradients} | "
            f"attn={enable_attention}"
        )

    def _register_embedding_hook(self, dotpath: str) -> None:
        """Attach a forward hook to capture the named layer's output.

        Args:
            dotpath: Dot-separated path to the layer within self._model.
        """
        try:
            layer = _resolve_layer(self._model, dotpath)
        except AttributeError as exc:
            raise AttributeError(
                f"Could not find layer '{dotpath}' in model. "
                f"Check --embedding-layer matches an actual module path."
            ) from exc

        def _hook(module: nn.Module, input: Any, output: Any) -> None:  # noqa: ARG001
            # output may be a tensor or a tuple (take the first element)
            raw = output[0] if isinstance(output, tuple) else output
            # Flatten to (batch, hidden_dim) -- take [CLS] / mean-pool
            if raw.dim() == 3:
                raw = raw[:, 0, :]  # [CLS] token
            self._last_embeddings = raw.detach().cpu().numpy()

        self._embedding_hook_handle = layer.register_forward_hook(_hook)

    def max_minibatch_size(self) -> int:
        """Return the maximum batch size for a single forward pass.

        Returns:
            Maximum minibatch size.
        """
        return self._max_minibatch_size

    def predict_minibatch(
        self, inputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run forward pass on a minibatch and return LIT-formatted outputs.

        Args:
            inputs: List of LIT input dicts (must contain self._text_col).

        Returns:
            List of output dicts matching output_spec().
        """
        texts = [ex[self._text_col] for ex in inputs]
        tokens_batch = self._tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        tokens_batch = {k: v.to(self._device) for k, v in tokens_batch.items()}

        # Collect per-example tokenised strings for LIT Tokens field
        token_strings_batch = [
            self._tokenizer.convert_ids_to_tokens(ids.tolist())
            for ids in tokens_batch['input_ids']
        ]

        # Forward pass
        forward_kwargs: dict[str, Any] = {}
        if self._enable_attention:
            forward_kwargs['output_attentions'] = True

        if self._enable_gradients:
            # Need grad for token embeddings
            token_emb_layer = self._get_token_embedding_layer()
            token_emb_layer.requires_grad_(True)

        with torch.set_grad_enabled(self._enable_gradients):
            outputs = self._model(**tokens_batch, **forward_kwargs)

        # Extract logits -- handle plain tensor or object with .logits attribute
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probas_batch = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        # Attention heads -- shape (num_layers, batch, heads, seq, seq)
        attentions_batch: list[np.ndarray] | None = None
        if self._enable_attention and hasattr(outputs, 'attentions') and outputs.attentions:
            # Stack: (batch, num_layers, heads, seq, seq)
            stacked = torch.stack(
                [a.detach().cpu() for a in outputs.attentions], dim=1
            ).numpy()
            attentions_batch = list(stacked)  # one per example

        # Token gradients (grad norm w.r.t. input embeddings)
        token_grads_batch: list[np.ndarray] | None = None
        if self._enable_gradients:
            token_grads_batch = self._compute_token_gradients(
                tokens_batch, logits, token_strings_batch
            )

        # Assemble per-example outputs
        results = []
        for i, (probas, token_strings) in enumerate(
            zip(probas_batch, token_strings_batch)
        ):
            out: dict[str, Any] = {
                'probas': probas.tolist(),
                'tokens': token_strings,
            }

            if self._last_embeddings is not None:
                out['cls_emb'] = self._last_embeddings[i].tolist()

            if attentions_batch is not None:
                # attentions_batch[i]: (num_layers, heads, seq, seq)
                attn = attentions_batch[i]
                for layer_idx in range(attn.shape[0]):
                    # LIT expects (heads, seq_q, seq_k)
                    out[f'attention_layer{layer_idx}'] = attn[layer_idx].tolist()

            if token_grads_batch is not None:
                out['token_grads'] = token_grads_batch[i].tolist()

            results.append(out)

        return results

    def _get_token_embedding_layer(self) -> nn.Module:
        """Attempt to locate the token embedding layer for gradient computation.

        Returns:
            The embedding nn.Module (best guess).
        """
        # Common paths for HuggingFace-compatible models
        for path in ('embeddings.word_embeddings', 'embed_tokens', 'embedding'):
            try:
                return _resolve_layer(self._model, path)
            except AttributeError:
                continue
        # Fallback: first parameter
        return self._model

    def _compute_token_gradients(
        self,
        tokens_batch: dict[str, torch.Tensor],
        logits: torch.Tensor,
        token_strings_batch: list[list[str]],
    ) -> list[np.ndarray]:
        """Compute gradient norm over input token embeddings.

        Takes the gradient of the winning class logit w.r.t. the token
        embedding matrix and returns the L2 norm per token position.

        Args:
            tokens_batch: Tokenised input tensors (on device).
            logits: Model logit output tensor.
            token_strings_batch: Per-example list of token strings.

        Returns:
            List of 1-D numpy arrays (one per example) with per-token scores.
        """
        grad_norms = []
        for i in range(logits.shape[0]):
            winning_class = logits[i].argmax()
            scalar = logits[i, winning_class]

            # Compute gradient w.r.t. all parameters; sum embedding grads
            self._model.zero_grad()
            scalar.backward(retain_graph=True)

            grad_norm = np.zeros(len(token_strings_batch[i]))
            for name, param in self._model.named_parameters():
                if 'embed' in name and param.grad is not None:
                    # Use the rows that correspond to this example's token ids
                    ids = tokens_batch['input_ids'][i]
                    token_grads = param.grad[ids].detach().cpu().numpy()
                    norms = np.linalg.norm(token_grads, axis=-1)
                    length = min(len(norms), len(grad_norm))
                    grad_norm[:length] += norms[:length]
                    break

            grad_norms.append(grad_norm)

        return grad_norms

    def input_spec(self) -> dict[str, lit_types.LitType]:
        """Describe the inputs the model expects.

        Returns:
            Dict with a single TextSegment field for the text input.
        """
        return {self._text_col: lit_types.TextSegment()}

    def output_spec(self) -> dict[str, lit_types.LitType]:
        """Describe all outputs this wrapper can produce.

        Returns:
            Dict mapping output field names to LIT types.
        """
        spec: dict[str, lit_types.LitType] = {
            'probas': lit_types.MulticlassPreds(
                vocab=self._label_vocab, parent='label'
            ),
            'tokens': lit_types.Tokens(parent=self._text_col),
        }

        if self._embedding_layer:
            spec['cls_emb'] = lit_types.Embeddings()

        if self._enable_gradients:
            spec['token_grads'] = lit_types.TokenGradients(align='tokens')

        if self._enable_attention:
            # We don't know the number of layers until the first forward pass,
            # so we expose a placeholder spec entry with a wildcard.
            # Actual layers are added dynamically; here we add layer 0 as a
            # representative entry -- LIT will discover others at runtime.
            spec['attention_layer0'] = lit_types.AttentionHeads(
                align_in='tokens', align_out='tokens'
            )

        return spec

    def __del__(self) -> None:
        """Clean up the embedding hook handle when the wrapper is garbage collected."""
        if self._embedding_hook_handle is not None:
            self._embedding_hook_handle.remove()


class JuryReferenceModel(lit_model.Model):
    """Pseudo-model that serves pre-computed LLM jury predictions from the CSV.

    No inference is performed.  This model looks up the consensus label and
    soft label distribution from the loaded labeled DataFrame for each input
    text.  Loading it alongside DistilledModelWrapper in the same LIT server
    lets you compare the distilled model against the LLM jury and human labels
    side by side in LIT's Compare mode.

    Args:
        data_path: Path to the labeled CSV (the same file passed to LabeledDataset).
        config: DatasetConfig for the current dataset.

    Example:
        >>> jury = JuryReferenceModel("outputs/labeled.csv", config)
    """

    def __init__(self, data_path: str, config: Any) -> None:
        """Load the labeled CSV and index it by text for fast lookup.

        Args:
            data_path: Path to labeled CSV.
            config: DatasetConfig instance.
        """
        import pandas as pd  # local to keep top-level import fast

        df = pd.read_csv(data_path)
        self._text_col: str = config.text_column
        self._label_vocab: list[str] = [str(l) for l in config.labels]

        # Build a text -> (label, soft_label) lookup
        self._lookup: dict[str, tuple[str, list[float]]] = {}
        for _, row in df.iterrows():
            text = str(row.get(self._text_col, ''))
            label = str(row.get('label', ''))
            soft_label = self._parse_soft_label(row.get('soft_label'), label)
            self._lookup[text] = (label, soft_label)

        logger.info(
            f"JuryReferenceModel ready: {len(self._lookup)} entries indexed"
        )

    def _parse_soft_label(
        self,
        raw: Any,
        hard_label: str,
    ) -> list[float]:
        """Parse soft_label column into a probability list aligned to vocab.

        Args:
            raw: Raw value from the soft_label column (JSON string or None).
            hard_label: Fallback hard label if soft_label is missing.

        Returns:
            List of floats (one per label in vocab), summing to 1.
        """
        dist: dict[str, float] = {}
        if raw is not None:
            try:
                parsed = json.loads(str(raw)) if isinstance(raw, str) else raw
                if isinstance(parsed, dict):
                    dist = {str(k): float(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if not dist:
            dist = {hard_label: 1.0}

        # Align to vocab order
        total = sum(dist.values()) or 1.0
        return [dist.get(label, 0.0) / total for label in self._label_vocab]

    def predict(
        self, inputs: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        """Look up pre-computed jury predictions for each input.

        Args:
            inputs: Iterable of LIT input dicts containing the text field.

        Returns:
            Iterable of output dicts with 'probas' matching the label vocab.
        """
        results = []
        for ex in inputs:
            text = str(ex.get(self._text_col, ''))
            _label, probas = self._lookup.get(
                text, ('', [1.0 / len(self._label_vocab)] * len(self._label_vocab))
            )
            results.append({'probas': probas})
        return results

    def input_spec(self) -> dict[str, lit_types.LitType]:
        """Describe the text input field.

        Returns:
            Dict with a single TextSegment for the text column.
        """
        return {self._text_col: lit_types.TextSegment()}

    def output_spec(self) -> dict[str, lit_types.LitType]:
        """Describe the jury probability output.

        Returns:
            Dict with MulticlassPreds aligned to the label vocab.
        """
        return {
            'probas': lit_types.MulticlassPreds(
                vocab=self._label_vocab, parent='label'
            )
        }
