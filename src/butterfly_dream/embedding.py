"""📐 Semantic embedding service for Butterfly Dream v2.

Wraps FastEmbed (ONNX Runtime) to provide lightweight, CPU-efficient
text embeddings.

Uses BAAI/bge-small-zh-v1.5 — a 33 MB bilingual (Chinese + English) model
that produces 512-dimensional vectors.

Singleton pattern: the model is loaded once on first use.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
_EMBED_DIM = 512  # bge-small-zh outputs 512-dim vectors

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional["EmbeddingService"] = None
_lock = threading.Lock()


class EmbeddingService:
    """Thin wrapper around FastEmbed providing encode + serialize helpers."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None  # lazy init
        self._model_lock = threading.Lock()

    # -- Lazy-loading model ------------------------------------------------

    def _ensure_model(self) -> bool:
        """Load the embedding model on first call.  Thread-safe."""
        if self._model is not None:
            return True
        with self._model_lock:
            if self._model is not None:
                return True
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self._model_name,
                    providers=None,  # auto-detect (CPU OK)
                )
                logger.info(
                    "Embedding model '%s' loaded (dim=%d)", self._model_name, _EMBED_DIM
                )
                return True
            except Exception as exc:
                logger.warning(
                    "Failed to load embedding model '%s': %s — "
                    "falling back to HRR",
                    self._model_name, exc,
                )
                return False

    # -- Public API ---------------------------------------------------------

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        """Encode a list of texts into 512-dim vectors.

        Returns a list of numpy arrays (float32).  Falls back to empty list
        if the model could not be loaded — callers should fall back to HRR.
        """
        if not texts or not self._ensure_model():
            return []
        try:
            return [
                np.array(v, dtype=np.float32) for v in self._model.embed(texts)
            ]
        except Exception as exc:
            logger.debug("Embedding batch encode failed: %s", exc)
            return []

    def encode_one(self, text: str) -> Optional[np.ndarray]:
        """Encode a single text string."""
        results = self.encode([text])
        return results[0] if results else None

    # -- Serialization helpers for BLOB storage -----------------------------

    @staticmethod
    def serialize(vector: np.ndarray) -> bytes:
        """Convert a numpy vector to SQLite BLOB bytes."""
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def deserialize(data: bytes) -> Optional[np.ndarray]:
        """Reconstruct a numpy vector from SQLite BLOB bytes."""
        try:
            return np.frombuffer(data, dtype=np.float32).copy()
        except Exception:
            return None

    # -- Cosine similarity --------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        return dot / norm if norm > 0 else 0.0

    @staticmethod
    def cosine_similarity_batch(
        query: np.ndarray, candidates: list[np.ndarray],
    ) -> list[float]:
        """Cosine similarity from one query vector to many candidates."""
        if not candidates:
            return []
        q_norm = query / (np.linalg.norm(query) + 1e-12)
        matrix = np.stack(candidates)  # (N, D)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        scores = np.dot(matrix_norm, q_norm).tolist()
        return [float(s) for s in scores]


# ---------------------------------------------------------------------------
# Module-level helpers (convenience, route through singleton)
# ---------------------------------------------------------------------------


def get_embedding_service() -> EmbeddingService:
    """Return the singleton EmbeddingService (lazy init on first call)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EmbeddingService()
    return _instance


def encode(texts: list[str]) -> list[np.ndarray]:
    """Convenience: encode texts via the singleton service."""
    return get_embedding_service().encode(texts)


def encode_one(text: str) -> Optional[np.ndarray]:
    """Convenience: encode a single text via the singleton service."""
    return get_embedding_service().encode_one(text)
