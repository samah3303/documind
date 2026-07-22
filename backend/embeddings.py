"""
DocuMind Embeddings Service
Uses sentence-transformers (all-MiniLM-L6-v2) running locally for free embeddings.
"""

from __future__ import annotations

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Global singleton — loaded once at startup
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Return the global SentenceTransformer instance, loading it on first call."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(
            "Embedding model loaded. Dimension: %s",
            _embedding_model.get_sentence_embedding_dimension(),
        )
    return _embedding_model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text chunks to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def generate_single_embedding(text: str) -> list[float]:
    """Generate an embedding for a single text string."""
    result = generate_embeddings([text])
    return result[0] if result else []
