"""
Semantic chunking engine for Gakondo-Vision.

Splits ingested text documents into overlapping chunks sized between
500–1000 tokens and attaches structured cultural-heritage metadata
to every chunk for downstream filtered retrieval.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion import TextDocument

logger = logging.getLogger(__name__)

# ── Approximate token-to-character ratio ─────────────────────────────────
# For English text the rough ratio is ~4 chars per token.
# We use this to map the user-facing *token* parameters into the
# character-based splitter while keeping the config intuitive.
CHARS_PER_TOKEN = 4

# ── Cultural metadata defaults ───────────────────────────────────────────
DEFAULT_METADATA: dict[str, str] = {
    "cultural_category": "general",
    "historical_era": "unspecified",
    "region": "Rwanda",
}

# ── Metadata inference heuristics ────────────────────────────────────────
# Simple keyword-based heuristics for auto-tagging chunks.
# A production system would use a classifier; this gives us useful
# structure from day one.

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "oral_tradition": [
        "umugani", "igitaramo", "folklore", "proverb", "tale", "story",
        "legend", "oral", "narrative", "fable",
    ],
    "art_craft": [
        "imigongo", "agaseke", "basket", "weaving", "pottery",
        "art", "pattern", "design", "craft", "motif",
    ],
    "dance_music": [
        "intore", "ingoma", "drum", "dance", "song", "music",
        "rhythm", "ceremony", "performance",
    ],
    "governance": [
        "umwami", "king", "monarchy", "court", "gacaca",
        "governance", "chief", "council",
    ],
    "spirituality": [
        "imana", "ryangombe", "kubandwa", "spirit", "ancestor",
        "ritual", "sacred", "shrine",
    ],
}

_ERA_KEYWORDS: dict[str, list[str]] = {
    "pre_colonial": [
        "kingdom", "dynasty", "ancient", "traditional", "pre-colonial",
        "precolonial", "umwami",
    ],
    "colonial": [
        "colonial", "german", "belgian", "mandate", "colonisation",
        "colonization",
    ],
    "post_independence": [
        "independence", "republic", "modern", "contemporary",
        "post-independence", "reconciliation",
    ],
}

_REGION_KEYWORDS: dict[str, list[str]] = {
    "Northern": ["musanze", "burera", "northern", "volcano", "virunga"],
    "Southern": ["huye", "butare", "nyanza", "southern"],
    "Eastern": ["kayonza", "kirehe", "eastern", "akagera"],
    "Western": ["karongi", "kibuye", "western", "kivu"],
    "Kigali": ["kigali", "capital"],
}


def _infer_field(
    text_lower: str,
    keyword_map: dict[str, list[str]],
    default: str,
) -> str:
    """Return the first matching key whose keywords appear in *text_lower*."""
    for label, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            return label
    return default


# ── Chunk data structure ─────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single text chunk with source provenance and cultural metadata."""
    chunk_id: str
    content: str
    source_path: str
    source_type: str
    char_count: int
    metadata: dict = field(default_factory=dict)


# ── Splitter factory ─────────────────────────────────────────────────────

def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter calibrated to the config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        is_separator_regex=False,
    )


# ── Metadata builder ────────────────────────────────────────────────────

def _build_chunk_metadata(
    text: str,
    doc: TextDocument,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build a metadata dictionary for a chunk.

    Priority order:
      1. Explicit *extra* overrides (caller-supplied).
      2. Metadata already attached to the source document.
      3. Keyword-inferred categories.
      4. Defaults.
    """
    text_lower = text.lower()

    meta: dict = {
        "source_type": doc.source_type,
        "source_path": doc.source_path,
        "source_filename": Path(doc.source_path).name,
        "cultural_category": _infer_field(
            text_lower,
            _CATEGORY_KEYWORDS,
            DEFAULT_METADATA["cultural_category"],
        ),
        "historical_era": _infer_field(
            text_lower,
            _ERA_KEYWORDS,
            DEFAULT_METADATA["historical_era"],
        ),
        "region": _infer_field(
            text_lower,
            _REGION_KEYWORDS,
            DEFAULT_METADATA["region"],
        ),
    }

    # Layer document-level metadata
    if doc.metadata:
        meta.update(doc.metadata)

    # Layer caller overrides
    if extra:
        meta.update(extra)

    return meta


def _make_chunk_id(source_path: str, index: int, content: str) -> str:
    """Deterministic chunk ID based on source + position + content hash."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{Path(source_path).stem}__chunk{index:04d}__{content_hash}"


# ── Public API ───────────────────────────────────────────────────────────

def chunk_document(
    doc: TextDocument,
    extra_metadata: Optional[dict] = None,
) -> list[TextChunk]:
    """
    Split a single TextDocument into overlapping TextChunks.

    Each chunk is enriched with:
      • Deterministic `chunk_id`
      • Structured metadata (source_type, cultural_category,
        historical_era, region, source_filename)
    """
    splitter = _build_splitter()
    raw_chunks: list[str] = splitter.split_text(doc.content)

    chunks: list[TextChunk] = []
    for idx, text in enumerate(raw_chunks):
        chunk_id = _make_chunk_id(doc.source_path, idx, text)
        metadata = _build_chunk_metadata(text, doc, extra_metadata)
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                content=text,
                source_path=doc.source_path,
                source_type=doc.source_type,
                char_count=len(text),
                metadata=metadata,
            )
        )

    logger.info(
        "Chunked %s → %d chunks (avg %d chars)",
        Path(doc.source_path).name,
        len(chunks),
        int(sum(c.char_count for c in chunks) / max(len(chunks), 1)),
    )
    return chunks


def chunk_documents(
    docs: list[TextDocument],
    extra_metadata: Optional[dict] = None,
) -> list[TextChunk]:
    """Chunk a batch of TextDocuments and return a flat list of TextChunks."""
    all_chunks: list[TextChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, extra_metadata))
    logger.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks
