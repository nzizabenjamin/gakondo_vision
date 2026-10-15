"""
Centralised configuration for the Gakondo-Vision pipeline.

All tuneable parameters (DB connection, model names, chunk sizes, etc.)
are defined here as Pydantic settings so they can be overridden via
environment variables or a `.env` file without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Resolve project root relative to this file ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_TEXTS_DIR = DATA_DIR / "raw_texts"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
METADATA_DIR = DATA_DIR / "metadata"


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ─────────────────────────────────────────────────────────
    database_url_override: Optional[str] = Field(default=None, validation_alias="database_url")
    postgres_user: str = "gakondo"
    postgres_password: str = "gakondo"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "gakondo_vision"
    postgres_sslmode: Optional[str] = None

    @property
    def database_url(self) -> str:
        """Build a psycopg3-compatible DSN or return direct DATABASE_URL."""
        if self.database_url_override:
            return self.database_url_override
        env_url = os.getenv("DATABASE_URL")
        if env_url:
            return env_url
        dsn = (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
        if self.postgres_sslmode:
            dsn += f"?sslmode={self.postgres_sslmode}"
        return dsn

    # ── CLIP model ───────────────────────────────────────────────────────
    clip_model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512

    # ── Chunking parameters ──────────────────────────────────────────────
    chunk_size: int = 800          # target chunk size in characters (~200 tokens)
    chunk_overlap: int = 150       # overlap in characters (~40 tokens)
    min_chunk_tokens: int = 500    # hard floor (tokens) – used for validation
    max_chunk_tokens: int = 1000   # hard ceiling (tokens) – used for validation

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k: int = 5                 # default number of results to return

    # ── FastAPI ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Streamlit ────────────────────────────────────────────────────────
    streamlit_port: int = 8501


# Singleton
settings = Settings()
