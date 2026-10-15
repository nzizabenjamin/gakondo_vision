"""
FastAPI backend for Gakondo-Vision.

Endpoints
─────────
GET  /health       → service health check
POST /search       → multimodal search (text or image query)
POST /ingest       → trigger ingestion pipeline for new documents
GET  /stats        → database statistics
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.db import count_chunks, init_db, upsert_chunks
from src.chunker import chunk_documents
from src.embeddings import embed_single_image, embed_text
from src.ingestion import ImageRecord, ingest_texts, load_images
from src.retriever import retrieve_by_image, retrieve_by_text

logger = logging.getLogger(__name__)

# ── App factory ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Gakondo-Vision API",
    description=(
        "Multimodal RAG engine for Rwandan intangible cultural heritage. "
        "Supports cross-modal text ↔ image retrieval."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup event ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Bootstrap database schema and pre-warm CLIP model."""
    try:
        init_db()
        logger.info("Database ready.")
    except Exception as exc:
        logger.error("Database init failed: %s", exc)

    try:
        from src.embeddings import _load_model
        _load_model()
        logger.info("CLIP model pre-warmed and ready in memory.")
    except Exception as exc:
        logger.error("CLIP model pre-warm failed: %s", exc)


# ── Response models ──────────────────────────────────────────────────────

class SearchResult(BaseModel):
    id: str
    content: Optional[str] = None
    image_path: Optional[str] = None
    metadata: dict = {}
    source_type: str = "text"
    score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    images_catalogued: int
    chunks_stored: int


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str


class StatsResponse(BaseModel):
    total_chunks: int


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Service health check."""
    db_status = "unknown"
    try:
        count_chunks()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    return HealthResponse(
        status="ok",
        version="0.1.0",
        database=db_status,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return database statistics."""
    try:
        total = count_chunks()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")
    return StatsResponse(total_chunks=total)


@app.post("/search", response_model=SearchResponse)
async def search(
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(default=5),
    cultural_category: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
    historical_era: Optional[str] = Form(None),
):
    """
    Multimodal search endpoint.

    Supply either a `query` (text) or an `image` (file upload).
    Optionally filter by metadata fields.
    """
    if not query and not image:
        raise HTTPException(
            status_code=400,
            detail="Provide either a 'query' (text) or 'image' (file upload).",
        )

    # Build metadata filter
    meta_filter: dict = {}
    if cultural_category:
        meta_filter["cultural_category"] = cultural_category
    if region:
        meta_filter["region"] = region
    if historical_era:
        meta_filter["historical_era"] = historical_era
    meta_filter = meta_filter or None

    if image:
        # Save uploaded image to a temp file for CLIP encoding
        suffix = Path(image.filename or "query.png").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            results = retrieve_by_image(
                tmp_path, top_k=top_k, metadata_filter=meta_filter
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        query_label = f"[image:{image.filename}]"
    else:
        results = retrieve_by_text(
            query, top_k=top_k, metadata_filter=meta_filter
        )
        query_label = query

    return SearchResponse(
        query=query_label,
        results=[
            SearchResult(**r.to_dict()) for r in results
        ],
        total=len(results),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """
    Trigger the full ingestion pipeline:
      1. Load text documents from data/raw_texts/
      2. Chunk documents with metadata
      3. Embed chunks with CLIP
      4. Catalogue images from data/raw_images/
      5. Embed images with CLIP
      6. Upsert everything into pgvector
    """
    # 1. Load text documents
    docs = ingest_texts()

    # 2. Chunk
    chunks = chunk_documents(docs)

    # 3. Embed text chunks
    chunk_texts = [c.content for c in chunks]
    text_embeddings = embed_text(chunk_texts) if chunk_texts else []

    # 4. Prepare text chunk records
    text_records = []
    for chunk, emb in zip(chunks, text_embeddings):
        text_records.append({
            "id": chunk.chunk_id,
            "content": chunk.content,
            "image_path": None,
            "embedding": emb,
            "metadata": chunk.metadata,
            "source_type": chunk.source_type,
        })

    # 5. Load and embed images
    image_records_raw: list[ImageRecord] = load_images()
    image_db_records = []
    if image_records_raw:
        from src.embeddings import embed_image
        image_paths = [r.image_path for r in image_records_raw]
        image_embeddings = embed_image(image_paths)
        for img_rec, emb in zip(image_records_raw, image_embeddings):
            img_id = f"img__{Path(img_rec.image_path).stem}"
            image_db_records.append({
                "id": img_id,
                "content": None,
                "image_path": img_rec.image_path,
                "embedding": emb,
                "metadata": img_rec.metadata,
                "source_type": "image",
            })

    # 6. Upsert all records
    all_records = text_records + image_db_records
    stored = upsert_chunks(all_records) if all_records else 0

    return IngestResponse(
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        images_catalogued=len(image_records_raw),
        chunks_stored=stored,
    )


# ── Run directly ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
