"""Embedding provider abstraction for API-based and local embedding models.

Supports three backends:
    - **local**: Sentence-transformers models running on-device (default).
    - **openai**: OpenAI Embeddings API (text-embedding-3-small, text-embedding-3-large).
    - **openrouter**: OpenRouter Embeddings API (OpenAI-compatible endpoint).

All providers return L2-normalized vectors suitable for cosine-distance analysis.
Disk caching is handled by the caller (EmbeddingAnalyzer), not the providers.

Example:
    >>> from sibyls.core.diagnostics.embedding_providers import get_embedding_provider
    >>> from sibyls.core.diagnostics.config import DiagnosticsConfig
    >>> config = DiagnosticsConfig(embedding_provider="openai", embedding_model="text-embedding-3-small")
    >>> provider = get_embedding_provider(config)
    >>> vectors = provider.embed(["Fed raises rates", "Dovish pivot"])
    >>> vectors.shape
    (2, 1536)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from sklearn.preprocessing import normalize


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Subclasses must implement ``embed()`` which takes a list of texts and
    returns an L2-normalized numpy array of shape (n_texts, embedding_dim).

    Args:
        model: Model identifier string.

    Example:
        >>> class MyProvider(BaseEmbeddingProvider):
        ...     def embed(self, texts):
        ...         return np.random.randn(len(texts), 128)
    """

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts into L2-normalized embedding vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per API call / batch.

        Returns:
            np.ndarray of shape (n_texts, embedding_dim), L2-normalized.
        """


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Sentence-transformers provider for on-device embedding inference.

    Wraps ``sentence_transformers.SentenceTransformer`` with lazy model loading.
    This is the default provider and requires no API keys.

    Args:
        model: Sentence-transformers model name (e.g., "all-MiniLM-L6-v2").

    Example:
        >>> provider = LocalEmbeddingProvider("all-MiniLM-L6-v2")
        >>> vecs = provider.embed(["hello world"])
        >>> vecs.shape
        (1, 384)
    """

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._st_model = None

    def _load_model(self) -> None:
        """Lazily load the sentence-transformer model on first call."""
        if self._st_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        model_name = self.model.replace('sentence-transformers/', '')
        logger.info(f'Loading local embedding model: {model_name}')
        self._st_model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts using sentence-transformers with L2 normalization.

        Args:
            texts: List of text strings to embed.
            batch_size: Encoding batch size (passed to SentenceTransformer.encode).

        Returns:
            np.ndarray of shape (n_texts, embedding_dim), L2-normalized.

        Example:
            >>> provider = LocalEmbeddingProvider("all-MiniLM-L6-v2")
            >>> vecs = provider.embed(["Fed raises rates", "Dovish pivot"])
            >>> vecs.shape
            (2, 384)
        """
        self._load_model()
        logger.info(f'Embedding {len(texts)} texts locally with {self.model}...')
        embeddings = self._st_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI Embeddings API provider.

    Uses the synchronous OpenAI client to call the embeddings endpoint.
    Supports text-embedding-3-small (1536d), text-embedding-3-large (3072d),
    and text-embedding-ada-002 (1536d, legacy).

    Args:
        model: OpenAI embedding model name (e.g., "text-embedding-3-small").
        api_key: Optional API key. Falls back to OPENAI_API_KEY env var.

    Example:
        >>> provider = OpenAIEmbeddingProvider("text-embedding-3-small")
        >>> vecs = provider.embed(["Fed raises rates"])
        >>> vecs.shape
        (1, 1536)
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model)
        import openai

        self._client = openai.OpenAI(api_key=api_key)

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Encode texts via the OpenAI Embeddings API.

        Batches requests to stay within API limits. Each batch sends up to
        ``batch_size`` texts in a single request.

        Args:
            texts: List of text strings to embed.
            batch_size: Max texts per API request (OpenAI supports up to 2048).

        Returns:
            np.ndarray of shape (n_texts, embedding_dim), L2-normalized.

        Example:
            >>> provider = OpenAIEmbeddingProvider("text-embedding-3-small")
            >>> vecs = provider.embed(["hello", "world"], batch_size=100)
            >>> vecs.shape
            (2, 1536)
        """
        logger.info(
            f'Embedding {len(texts)} texts via OpenAI ({self.model}) '
            f'in batches of {batch_size}...'
        )
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            if len(texts) > batch_size:
                logger.debug(
                    f'  Batch {i // batch_size + 1}/'
                    f'{(len(texts) - 1) // batch_size + 1} complete'
                )

        embeddings = np.array(all_embeddings, dtype=np.float32)
        return normalize(embeddings)


class OpenRouterEmbeddingProvider(BaseEmbeddingProvider):
    """OpenRouter Embeddings API provider.

    OpenRouter exposes an OpenAI-compatible ``/api/v1/embeddings`` endpoint
    that routes to upstream embedding models. This provider uses ``httpx``
    for synchronous HTTP calls with rate-limit awareness.

    Supported models include any embedding model available on OpenRouter,
    such as "openai/text-embedding-3-small" or other provider-prefixed models.

    Args:
        model: OpenRouter model identifier (e.g., "openai/text-embedding-3-small").
        api_key: Optional API key. Falls back to OPENROUTER_API_KEY or
            API_OPENROUTER_KEY env vars.

    Example:
        >>> provider = OpenRouterEmbeddingProvider("openai/text-embedding-3-small")
        >>> vecs = provider.embed(["Fed raises rates"])
        >>> vecs.shape
        (1, 1536)
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model)
        self._api_key = (
            api_key
            or os.getenv('OPENROUTER_API_KEY')
            or os.getenv('API_OPENROUTER_KEY')
        )
        if not self._api_key:
            raise ValueError(
                'OpenRouter API key required for embeddings. '
                'Set OPENROUTER_API_KEY environment variable or pass api_key.'
            )
        self._base_url = 'https://openrouter.ai/api/v1'

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Encode texts via the OpenRouter Embeddings API.

        Sends batched requests to the OpenAI-compatible embeddings endpoint.

        Args:
            texts: List of text strings to embed.
            batch_size: Max texts per API request.

        Returns:
            np.ndarray of shape (n_texts, embedding_dim), L2-normalized.

        Raises:
            httpx.HTTPStatusError: On HTTP errors from OpenRouter.

        Example:
            >>> provider = OpenRouterEmbeddingProvider("openai/text-embedding-3-small")
            >>> vecs = provider.embed(["hello", "world"], batch_size=100)
            >>> vecs.shape
            (2, 1536)
        """
        import httpx

        logger.info(
            f'Embedding {len(texts)} texts via OpenRouter ({self.model}) '
            f'in batches of {batch_size}...'
        )

        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/sibyls',
            'X-Title': 'AutoLabeler',
        }

        all_embeddings: list[list[float]] = []

        with httpx.Client(timeout=120.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {
                    'model': self.model,
                    'input': batch,
                }
                response = client.post(
                    f'{self._base_url}/embeddings',
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                sorted_data = sorted(data['data'], key=lambda x: x['index'])
                batch_embeddings = [item['embedding'] for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

                if len(texts) > batch_size:
                    logger.debug(
                        f'  Batch {i // batch_size + 1}/'
                        f'{(len(texts) - 1) // batch_size + 1} complete'
                    )

        embeddings = np.array(all_embeddings, dtype=np.float32)
        return normalize(embeddings)


_EMBEDDING_PROVIDERS: dict[str, type[BaseEmbeddingProvider]] = {
    'local': LocalEmbeddingProvider,
    'openai': OpenAIEmbeddingProvider,
    'openrouter': OpenRouterEmbeddingProvider,
}


def get_embedding_provider(config: 'DiagnosticsConfig') -> BaseEmbeddingProvider:
    """Factory function to create an embedding provider from diagnostics config.

    Dispatches based on ``config.embedding_provider`` to the appropriate
    provider class, passing ``config.embedding_model`` and optional API key.

    Args:
        config: DiagnosticsConfig with embedding_provider, embedding_model,
            and optionally embedding_api_key fields.

    Returns:
        BaseEmbeddingProvider instance ready to call ``.embed()``.

    Raises:
        ValueError: If the provider name is not recognized.

    Example:
        >>> from sibyls.core.diagnostics.config import DiagnosticsConfig
        >>> config = DiagnosticsConfig(embedding_provider="openai", embedding_model="text-embedding-3-small")
        >>> provider = get_embedding_provider(config)
        >>> type(provider).__name__
        'OpenAIEmbeddingProvider'
    """
    provider_name = getattr(config, 'embedding_provider', 'local')
    api_key = getattr(config, 'embedding_api_key', None)

    if provider_name not in _EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider: '{provider_name}'. "
            f"Available: {sorted(_EMBEDDING_PROVIDERS.keys())}"
        )

    provider_cls = _EMBEDDING_PROVIDERS[provider_name]

    if provider_name == 'local':
        return provider_cls(model=config.embedding_model)

    return provider_cls(model=config.embedding_model, api_key=api_key)
