"""
CLIP embedding pipeline for Gakondo-Vision.

Provides a unified interface to embed both text chunks and images
into the same 512-dimensional vector space using the HuggingFace
`sentence-transformers/clip-ViT-B-32` model.

Design notes
────────────
• The model is loaded lazily on first call and cached as a module-level
  singleton so that repeated calls don't re-download or re-initialise.
• Batch encoding is supported for throughput; callers can tune
  `batch_size` per their GPU/CPU budget.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from src.config import settings

logger = logging.getLogger(__name__)

# ── Lazy model singleton ─────────────────────────────────────────────────
_model = None
_processor = None
_tokenizer = None


def _load_model():
    """Load CLIP model + processor on first use."""
    global _model, _processor, _tokenizer

    if _model is not None:
        return

    logger.info("Loading CLIP model: %s …", settings.clip_model_name)

    try:
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

        _model = CLIPModel.from_pretrained(settings.clip_model_name)
        _processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
        _tokenizer = CLIPTokenizerFast.from_pretrained(settings.clip_model_name)
        _model.eval()
        logger.info("CLIP model loaded successfully (dim=%d).", settings.embedding_dim)

    except ImportError:
        raise ImportError(
            "transformers is required for CLIP embeddings. "
            "Install with: pip install transformers torch"
        )


# ── Text embedding ───────────────────────────────────────────────────────

def embed_text(
    texts: list[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> list[np.ndarray]:
    """
    Encode a list of text strings into CLIP text embeddings.

    Parameters
    ----------
    texts : list[str]
        The text chunks to embed.
    batch_size : int
        Number of texts to encode per forward pass.
    normalize : bool
        If True, L2-normalise each vector (recommended for cosine search).

    Returns
    -------
    list[np.ndarray]
        One 512-d vector per input string.
    """
    import torch

    _load_model()
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP's context window
            return_tensors="pt",
        )
        with torch.no_grad():
            text_features = _model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output

        vecs = text_features.cpu().numpy()
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms

        all_embeddings.extend(vecs)
        logger.debug("Embedded text batch %d–%d", start, start + len(batch))

    logger.info("Embedded %d text chunks.", len(all_embeddings))
    return all_embeddings


# ── Image embedding ──────────────────────────────────────────────────────

def embed_image(
    image_paths: list[Union[str, Path]],
    batch_size: int = 16,
    normalize: bool = True,
) -> list[np.ndarray]:
    """
    Encode a list of images into CLIP image embeddings.

    Parameters
    ----------
    image_paths : list[str | Path]
        File paths to the images to embed.
    batch_size : int
        Number of images to encode per forward pass.
    normalize : bool
        If True, L2-normalise each vector.

    Returns
    -------
    list[np.ndarray]
        One 512-d vector per image.
    """
    import torch

    _load_model()
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_results: list[Optional[np.ndarray]] = [None] * len(batch_paths)
        valid_images: list[Image.Image] = []
        valid_indices: list[int] = []

        for i, p in enumerate(batch_paths):
            try:
                img = Image.open(str(p)).convert("RGB")
                valid_images.append(img)
                valid_indices.append(i)
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", p, exc)
                batch_results[i] = np.zeros(settings.embedding_dim, dtype=np.float32)

        if valid_images:
            inputs = _processor(images=valid_images, return_tensors="pt")
            with torch.no_grad():
                image_features = _model.get_image_features(**inputs)
                if hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output

            vecs = image_features.cpu().numpy()
            if normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                vecs = vecs / norms

            for idx, vec in zip(valid_indices, vecs):
                batch_results[idx] = vec

        all_embeddings.extend([v for v in batch_results if v is not None])
        logger.debug("Embedded image batch %d–%d", start, start + len(batch_paths))

    logger.info("Embedded %d images.", len(all_embeddings))
    return all_embeddings


# ── Single-item convenience wrappers ─────────────────────────────────────

def embed_single_text(text: str, normalize: bool = True) -> np.ndarray:
    """Embed a single text string and return its 512-d vector."""
    return embed_text([text], normalize=normalize)[0]


def embed_single_image(
    image_path: Union[str, Path],
    normalize: bool = True,
) -> np.ndarray:
    """Embed a single image file and return its 512-d vector."""
    return embed_image([image_path], normalize=normalize)[0]
