"""
Retrieval layer for Gakondo-Vision.

Wraps the low-level `db.similarity_search` in a higher-level API that
accepts raw text or image queries, embeds them, and returns ranked
results with optional metadata filtering and reranking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.config import settings
from src.db import similarity_search
from src.embeddings import embed_single_image, embed_single_text

logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────────

class RetrievalResult:
    """Thin wrapper around a ranked search result for ergonomic access."""

    __slots__ = ("id", "content", "image_path", "metadata", "source_type", "score")

    def __init__(self, raw: dict) -> None:
        self.id: str = raw["id"]
        self.content: Optional[str] = raw.get("content")
        self.image_path: Optional[str] = raw.get("image_path")
        self.metadata: dict = raw.get("metadata", {})
        self.source_type: str = raw.get("source_type", "text")
        self.score: float = raw.get("similarity", 0.0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "image_path": self.image_path,
            "metadata": self.metadata,
            "source_type": self.source_type,
            "score": self.score,
        }

    def __repr__(self) -> str:
        label = (self.content or "")[:60] if self.content else self.image_path
        return f"<Result id={self.id!r} score={self.score:.4f} '{label}…'>"


# ── Public retrieval API ─────────────────────────────────────────────────

def retrieve_by_text(
    query: str,
    top_k: int | None = None,
    metadata_filter: dict | None = None,
) -> list[RetrievalResult]:
    """
    Embed a text query with CLIP and return the top-k most similar
    heritage chunks from pgvector.

    Parameters
    ----------
    query : str
        Natural-language search query.
    top_k : int, optional
        Number of results (defaults to ``settings.top_k``).
    metadata_filter : dict, optional
        Key-value metadata constraints (e.g. ``{"region": "Southern"}``).
    """
    top_k = top_k or settings.top_k
    query_vec = embed_single_text(query)
    raw_results = similarity_search(
        query_embedding=query_vec,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
    results = [RetrievalResult(r) for r in raw_results]
    logger.info(
        "Text query '%s…' → %d results (top score %.4f)",
        query[:40],
        len(results),
        results[0].score if results else 0.0,
    )
    return results


def retrieve_by_image(
    image_path: Union[str, Path],
    top_k: int | None = None,
    metadata_filter: dict | None = None,
) -> list[RetrievalResult]:
    """
    Embed an image query with CLIP and return the most similar
    heritage chunks (text *or* image) from pgvector.

    Parameters
    ----------
    image_path : str | Path
        Path to the query image file.
    top_k : int, optional
        Number of results.
    metadata_filter : dict, optional
        Key-value metadata constraints.
    """
    top_k = top_k or settings.top_k
    query_vec = embed_single_image(image_path)
    raw_results = similarity_search(
        query_embedding=query_vec,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
    results = [RetrievalResult(r) for r in raw_results]
    logger.info(
        "Image query '%s' → %d results (top score %.4f)",
        Path(image_path).name,
        len(results),
        results[0].score if results else 0.0,
    )
    return results


# ── Reranking (lightweight) ──────────────────────────────────────────────

def rerank_by_metadata_boost(
    results: list[RetrievalResult],
    preferred_category: str | None = None,
    preferred_era: str | None = None,
    boost: float = 0.05,
) -> list[RetrievalResult]:
    """
    Apply a small score boost to results matching preferred metadata,
    then re-sort.

    This is a simple heuristic ranker; a production system might use
    a cross-encoder model instead.
    """
    for r in results:
        if preferred_category and r.metadata.get("cultural_category") == preferred_category:
            r.score += boost
        if preferred_era and r.metadata.get("historical_era") == preferred_era:
            r.score += boost

    results.sort(key=lambda r: r.score, reverse=True)
    return results
