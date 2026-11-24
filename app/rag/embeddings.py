"""Local Hugging Face sentence embeddings with batching and L2 normalization."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when a model cannot load or returns invalid vectors."""


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str, cache_dir: str) -> SentenceTransformer:
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
    except Exception as exc:
        logger.exception("Could not load local embedding model %s", model_name)
        raise EmbeddingError(
            f"Could not load embedding model {model_name!r}. Check model access and local cache: {exc}"
        ) from exc


class HuggingFaceEmbedder:
    """Thin wrapper around a cached SentenceTransformer model."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        cache_dir: str | Path | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size if batch_size is not None else settings.embedding_batch_size
        self.cache_dir = str(cache_dir or settings.model_cache_dir)
        self._provided_model = model
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    @property
    def model(self) -> SentenceTransformer:
        if self._provided_model is not None:
            return self._provided_model
        return _load_model(self.model_name, self.device, self.cache_dir)

    @property
    def dimension(self) -> int:
        model = self.model
        get_dimension = getattr(model, "get_embedding_dimension", None)
        dimension = get_dimension() if get_dimension else model.get_sentence_embedding_dimension()
        if not dimension or dimension < 1:
            raise EmbeddingError("Embedding model did not expose a positive vector dimension")
        return int(dimension)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        try:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            logger.exception("Embedding generation failed for %d input(s)", len(texts))
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
        return self._normalize(vectors, expected_rows=len(texts))

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return self.encode([query])[0]

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.model.tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:
            logger.warning("Tokenizer count failed; using approximate count: %s", exc)
            return max(1, len(text.split())) if text.strip() else 0

    def _normalize(self, vectors: object, *, expected_rows: int) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1 and expected_rows == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] != expected_rows:
            raise EmbeddingError(f"Expected {expected_rows} vectors in a matrix; got shape {array.shape}")
        dimension = self.dimension
        if array.shape[1] != dimension:
            raise EmbeddingError(f"Embedding dimension mismatch: expected {dimension}, received {array.shape[1]}")
        if not np.isfinite(array).all():
            raise EmbeddingError("Embeddings contain NaN or infinite values")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise EmbeddingError("Embedding model returned a zero vector")
        return np.ascontiguousarray(array / norms, dtype=np.float32)
