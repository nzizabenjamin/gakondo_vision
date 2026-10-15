"""
Document & image ingestion pipeline for Gakondo-Vision.

Responsibilities
────────────────
• Load PDF files using PyMuPDF (fitz) with fallback to pdfplumber.
• Load plain-text (.txt) files.
• Catalogue images from the raw_images directory with optional
  sidecar JSON metadata.
• Provide a stub for OCR-based ingestion of historical scanned documents.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import RAW_IMAGES_DIR, RAW_TEXTS_DIR, METADATA_DIR

logger = logging.getLogger(__name__)

# ── Data containers ──────────────────────────────────────────────────────

@dataclass
class TextDocument:
    """Represents a single loaded text document."""
    content: str
    source_path: str
    source_type: str  # "pdf" | "txt" | "scan"
    metadata: dict = field(default_factory=dict)


@dataclass
class ImageRecord:
    """Represents a catalogued image with its metadata."""
    image_path: str
    metadata: dict = field(default_factory=dict)


# ── PDF loading ──────────────────────────────────────────────────────────

def load_pdf_pymupdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF ingestion. "
            "Install it with: pip install PyMuPDF"
        )

    doc = fitz.open(str(path))
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def load_pdf_pdfplumber(path: Path) -> str:
    """Fallback PDF extraction using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required as a fallback PDF reader. "
            "Install it with: pip install pdfplumber"
        )

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)
    return "\n\n".join(pages)


def load_pdf(path: Path) -> str:
    """
    Load text from a PDF, attempting PyMuPDF first and falling back
    to pdfplumber if the primary extraction yields no content.
    """
    try:
        content = load_pdf_pymupdf(path)
        if content.strip():
            return content
        logger.info("PyMuPDF returned empty text for %s; trying pdfplumber.", path)
    except Exception as exc:
        logger.warning("PyMuPDF failed for %s: %s — falling back.", path, exc)

    return load_pdf_pdfplumber(path)


# ── Plain text loading ───────────────────────────────────────────────────

def load_txt(path: Path, encoding: str = "utf-8") -> str:
    """Load a plain-text file."""
    return path.read_text(encoding=encoding)


# ── OCR stub for historical scans ───────────────────────────────────────

def load_scan_ocr(path: Path) -> str:
    """
    Placeholder for OCR-based ingestion of historical document scans.

    In production this would call Tesseract or an equivalent OCR engine.
    For Phase 1 we return a stub string so the rest of the pipeline can
    be exercised end-to-end.
    """
    logger.warning(
        "OCR ingestion is stubbed — returning placeholder for %s", path
    )
    return (
        f"[OCR PLACEHOLDER] Scanned document: {path.name}. "
        "Full OCR extraction will be enabled in Phase 2."
    )


# ── Image cataloguing ───────────────────────────────────────────────────

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}


def _load_sidecar_metadata(image_path: Path) -> dict:
    """
    Look for a JSON sidecar file with the same stem as the image.
    Example: `imigongo_01.png` → `imigongo_01.json`
    """
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid JSON sidecar: %s", sidecar)
    return {}


def load_images(
    directory: Optional[Path] = None,
) -> list[ImageRecord]:
    """
    Walk the images directory and return catalogued ImageRecord objects.

    For each image, the function looks for a matching `.json` sidecar
    file to populate the metadata field.
    """
    directory = directory or RAW_IMAGES_DIR
    records: list[ImageRecord] = []

    if not directory.exists():
        logger.warning("Image directory does not exist: %s", directory)
        return records

    for img_path in sorted(directory.rglob("*")):
        if img_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            meta = _load_sidecar_metadata(img_path)
            meta.setdefault("source_type", "image")
            records.append(
                ImageRecord(
                    image_path=str(img_path),
                    metadata=meta,
                )
            )
            logger.debug("Catalogued image: %s", img_path.name)

    logger.info("Catalogued %d images from %s", len(records), directory)
    return records


# ── Unified ingestion entry-point ────────────────────────────────────────

def ingest_texts(
    directory: Optional[Path] = None,
) -> list[TextDocument]:
    """
    Scan the raw_texts directory and load every supported document.

    File dispatch:
      .pdf → load_pdf()
      .txt → load_txt()
      .tif / .tiff / .png / .jpg (inside raw_texts) → load_scan_ocr()
    """
    directory = directory or RAW_TEXTS_DIR
    documents: list[TextDocument] = []

    if not directory.exists():
        logger.warning("Text directory does not exist: %s", directory)
        return documents

    for file_path in sorted(directory.rglob("*")):
        if file_path.is_dir():
            continue

        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                content = load_pdf(file_path)
                source_type = "pdf"
            elif ext == ".txt":
                content = load_txt(file_path)
                source_type = "txt"
            elif ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                content = load_scan_ocr(file_path)
                source_type = "scan"
            else:
                logger.debug("Skipping unsupported file: %s", file_path)
                continue

            documents.append(
                TextDocument(
                    content=content,
                    source_path=str(file_path),
                    source_type=source_type,
                )
            )
            logger.info("Ingested [%s]: %s", source_type, file_path.name)

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", file_path, exc)

    logger.info("Total documents ingested: %d", len(documents))
    return documents
