"""
PostgreSQL + pgvector database layer for Gakondo-Vision.

Responsibilities
────────────────
• Enable the `vector` extension on first run.
• Create and manage the `heritage_chunks` table storing text/image
  embeddings, content, and JSONB metadata.
• Provide upsert and similarity-search helpers used by the rest of
  the pipeline.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from src.config import settings

logger = logging.getLogger(__name__)

# ── SQL statements ───────────────────────────────────────────────────────

_SQL_ENABLE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

_SQL_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS heritage_chunks (
    id              TEXT PRIMARY KEY,
    content         TEXT,
    image_path      TEXT,
    embedding       vector({settings.embedding_dim}),
    metadata        JSONB DEFAULT '{{}}'::jsonb,
    source_type     TEXT NOT NULL DEFAULT 'text',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_SQL_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_heritage_chunks_embedding
    ON heritage_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

_SQL_CREATE_METADATA_INDEX = """
CREATE INDEX IF NOT EXISTS idx_heritage_chunks_metadata
    ON heritage_chunks
    USING gin (metadata);
"""

_SQL_UPSERT = """
INSERT INTO heritage_chunks (id, content, image_path, embedding, metadata, source_type)
VALUES (%(id)s, %(content)s, %(image_path)s, %(embedding)s, %(metadata)s, %(source_type)s)
ON CONFLICT (id) DO UPDATE SET
    content     = EXCLUDED.content,
    image_path  = EXCLUDED.image_path,
    embedding   = EXCLUDED.embedding,
    metadata    = EXCLUDED.metadata,
    source_type = EXCLUDED.source_type,
    updated_at  = now();
"""

_SQL_SIMILARITY_SEARCH = """
SELECT
    id,
    content,
    image_path,
    metadata,
    source_type,
    1 - (embedding <=> %(query_embedding)s) AS similarity
FROM heritage_chunks
{where_clause}
ORDER BY embedding <=> %(query_embedding)s
LIMIT %(top_k)s;
"""

_SQL_COUNT = "SELECT count(*) FROM heritage_chunks;"

_SQL_DELETE_BY_ID = "DELETE FROM heritage_chunks WHERE id = %(id)s;"


# ── Connection helpers ───────────────────────────────────────────────────

@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Yield a psycopg3 connection with pgvector type registration."""
    conn = psycopg.connect(settings.database_url, autocommit=False)
    try:
        try:
            register_vector(conn)
        except Exception:
            pass
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema bootstrap ────────────────────────────────────────────────────

def init_db() -> None:
    """
    Ensure the pgvector extension is enabled and the heritage_chunks
    table + indices exist.  Safe to call multiple times.
    """
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(_SQL_ENABLE_EXTENSION)
            cur.execute(_SQL_CREATE_TABLE)
            try:
                cur.execute(_SQL_CREATE_INDEX)
            except Exception as exc:
                logger.warning(
                    "IVFFlat index creation skipped: %s. It can be created after data is populated.",
                    exc,
                )
            try:
                cur.execute(_SQL_CREATE_METADATA_INDEX)
            except Exception as exc:
                logger.warning("Metadata index notice: %s", exc)
    logger.info("Database schema initialised.")


# ── Upsert operations ───────────────────────────────────────────────────

def upsert_chunk(
    chunk_id: str,
    content: str,
    embedding: np.ndarray,
    metadata: dict,
    source_type: str = "text",
    image_path: Optional[str] = None,
) -> None:
    """Insert or update a single chunk."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                _SQL_UPSERT,
                {
                    "id": chunk_id,
                    "content": content,
                    "image_path": image_path,
                    "embedding": embedding,
                    "metadata": json.dumps(metadata),
                    "source_type": source_type,
                },
            )
    logger.debug("Upserted chunk: %s", chunk_id)


def upsert_chunks(
    records: list[dict[str, Any]],
) -> int:
    """
    Batch-upsert multiple chunk records.

    Each dict in *records* must contain keys:
        id, content, embedding, metadata, source_type
    and optionally: image_path
    """
    count = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for rec in records:
                cur.execute(
                    _SQL_UPSERT,
                    {
                        "id": rec["id"],
                        "content": rec.get("content"),
                        "image_path": rec.get("image_path"),
                        "embedding": rec["embedding"],
                        "metadata": json.dumps(rec.get("metadata", {})),
                        "source_type": rec.get("source_type", "text"),
                    },
                )
                count += 1
    logger.info("Batch upserted %d chunks.", count)
    return count


def upsert_images(
    records: list[dict[str, Any]],
) -> int:
    """
    Convenience wrapper for batch-upserting image records.

    Each dict must contain: id, image_path, embedding, metadata.
    """
    for rec in records:
        rec.setdefault("source_type", "image")
        rec.setdefault("content", None)
    return upsert_chunks(records)


# ── Query operations ────────────────────────────────────────────────────

def similarity_search(
    query_embedding: np.ndarray,
    top_k: int = 5,
    metadata_filter: Optional[dict] = None,
) -> list[dict]:
    """
    Find the top-k most similar chunks to *query_embedding*.

    Parameters
    ----------
    query_embedding : np.ndarray
        512-d query vector (text or image CLIP embedding).
    top_k : int
        Number of results to return.
    metadata_filter : dict, optional
        Key-value pairs to filter on the JSONB metadata column.
        Example: {"cultural_category": "art_craft", "region": "Southern"}

    Returns
    -------
    list[dict]
        Ranked results with keys: id, content, image_path, metadata,
        source_type, similarity.
    """
    where_parts: list[str] = []
    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "top_k": top_k,
    }

    if metadata_filter:
        for key, value in metadata_filter.items():
            param_name = f"meta_{key}"
            where_parts.append(f"metadata->>'{key}' = %({param_name})s")
            params[param_name] = str(value)

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts)

    sql = _SQL_SIMILARITY_SEARCH.format(where_clause=where_clause)

    results: list[dict] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                results.append(
                    {
                        "id": row[0],
                        "content": row[1],
                        "image_path": row[2],
                        "metadata": row[3],
                        "source_type": row[4],
                        "similarity": float(row[5]),
                    }
                )

    logger.info("Similarity search returned %d results.", len(results))
    return results


# ── Utility ──────────────────────────────────────────────────────────────

def count_chunks() -> int:
    """Return total number of stored chunks."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SQL_COUNT)
            return cur.fetchone()[0]


def delete_chunk(chunk_id: str) -> None:
    """Delete a chunk by its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SQL_DELETE_BY_ID, {"id": chunk_id})
    logger.debug("Deleted chunk: %s", chunk_id)
