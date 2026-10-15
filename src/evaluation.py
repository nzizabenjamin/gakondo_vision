"""
Evaluation module for Gakondo-Vision (Ragas-inspired).

Implements lightweight faithfulness and relevancy scoring to verify
that retrieved chunks genuinely support any generated answer and
are semantically aligned with the user query.

Design notes
────────────
• We follow the Ragas framework *principles* but implement them with
  local CLIP similarity rather than requiring an external LLM judge,
  keeping the system fully local-first.
• An optional LLM-judge path (using a local model) can be added in
  Phase 2 for deeper faithfulness analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import settings, PROJECT_ROOT
from src.embeddings import embed_text

logger = logging.getLogger(__name__)


# ── Evaluation result containers ─────────────────────────────────────────

@dataclass
class ChunkScore:
    """Evaluation score for a single retrieved chunk."""
    chunk_id: str
    relevancy: float          # cosine similarity between query and chunk
    faithfulness: float       # overlap / support score
    content_preview: str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation report for a single query."""
    query: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    chunk_scores: list[ChunkScore] = field(default_factory=list)
    mean_relevancy: float = 0.0
    mean_faithfulness: float = 0.0
    top_k: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "top_k": self.top_k,
            "mean_relevancy": round(self.mean_relevancy, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "chunks": [
                {
                    "chunk_id": cs.chunk_id,
                    "relevancy": round(cs.relevancy, 4),
                    "faithfulness": round(cs.faithfulness, 4),
                    "content_preview": cs.content_preview,
                }
                for cs in self.chunk_scores
            ],
        }


# ── Cosine similarity helper ────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ── Relevancy scoring ───────────────────────────────────────────────────

def score_relevancy(
    query: str,
    chunk_contents: list[str],
) -> list[float]:
    """
    Measure semantic relevancy of each retrieved chunk to the query
    using CLIP text embeddings + cosine similarity.

    A score of 1.0 means perfect alignment; 0.0 means no relation.
    """
    all_texts = [query] + chunk_contents
    embeddings = embed_text(all_texts)
    query_vec = embeddings[0]

    scores: list[float] = []
    for chunk_vec in embeddings[1:]:
        scores.append(_cosine_similarity(query_vec, chunk_vec))

    return scores


# ── Faithfulness scoring ─────────────────────────────────────────────────

def score_faithfulness(
    answer: str,
    chunk_contents: list[str],
) -> list[float]:
    """
    Estimate how well each retrieved chunk *supports* the generated
    answer.  A high score indicates the chunk's content is
    semantically close to the answer text, suggesting the answer is
    grounded in retrieval rather than hallucinated.

    Methodology (local CLIP):
      1. Embed the answer and each chunk with CLIP text encoder.
      2. Compute cosine similarity between answer ↔ chunk.
      3. Return per-chunk faithfulness scores.
    """
    if not answer.strip():
        return [0.0] * len(chunk_contents)

    all_texts = [answer] + chunk_contents
    embeddings = embed_text(all_texts)
    answer_vec = embeddings[0]

    scores: list[float] = []
    for chunk_vec in embeddings[1:]:
        scores.append(_cosine_similarity(answer_vec, chunk_vec))

    return scores


# ── Full evaluation pipeline ────────────────────────────────────────────

def evaluate_retrieval(
    query: str,
    retrieved_chunks: list[dict],
    answer: str = "",
) -> EvalReport:
    """
    Run the full evaluation pipeline on a set of retrieved chunks.

    Parameters
    ----------
    query : str
        The user's original query.
    retrieved_chunks : list[dict]
        Each dict should have at least 'id' and 'content' keys.
    answer : str, optional
        A generated answer to evaluate faithfulness against.
        If empty, faithfulness is scored as 0.0 for all chunks.

    Returns
    -------
    EvalReport
        Aggregated report with per-chunk and mean scores.
    """
    chunk_ids = [c["id"] for c in retrieved_chunks]
    chunk_contents = [c.get("content", "") or "" for c in retrieved_chunks]

    # Relevancy: query ↔ chunks
    relevancy_scores = score_relevancy(query, chunk_contents)

    # Faithfulness: answer ↔ chunks
    if answer.strip():
        faithfulness_scores = score_faithfulness(answer, chunk_contents)
    else:
        faithfulness_scores = [0.0] * len(chunk_contents)

    chunk_eval: list[ChunkScore] = []
    for cid, content, rel, faith in zip(
        chunk_ids, chunk_contents, relevancy_scores, faithfulness_scores
    ):
        chunk_eval.append(
            ChunkScore(
                chunk_id=cid,
                relevancy=rel,
                faithfulness=faith,
                content_preview=content[:120],
            )
        )

    report = EvalReport(
        query=query,
        chunk_scores=chunk_eval,
        mean_relevancy=(
            float(np.mean(relevancy_scores)) if relevancy_scores else 0.0
        ),
        mean_faithfulness=(
            float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0
        ),
        top_k=len(chunk_eval),
    )

    logger.info(
        "Eval for '%s…': relevancy=%.4f, faithfulness=%.4f (%d chunks)",
        query[:30],
        report.mean_relevancy,
        report.mean_faithfulness,
        report.top_k,
    )
    return report


# ── Report persistence ───────────────────────────────────────────────────

def save_report_json(report: EvalReport, output_dir: Optional[Path] = None) -> Path:
    """Save an evaluation report as a JSON file."""
    output_dir = output_dir or PROJECT_ROOT / "data" / "metadata" / "eval_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = report.timestamp.replace(":", "-").replace("+", "_")
    path = output_dir / f"eval_{ts}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved eval report: %s", path)
    return path


def save_report_markdown(report: EvalReport, output_dir: Optional[Path] = None) -> Path:
    """Save an evaluation report as a human-readable Markdown file."""
    output_dir = output_dir or PROJECT_ROOT / "data" / "metadata" / "eval_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = report.timestamp.replace(":", "-").replace("+", "_")
    path = output_dir / f"eval_{ts}.md"

    lines: list[str] = [
        f"# Evaluation Report",
        f"",
        f"- **Query:** {report.query}",
        f"- **Timestamp:** {report.timestamp}",
        f"- **Chunks evaluated:** {report.top_k}",
        f"- **Mean relevancy:** {report.mean_relevancy:.4f}",
        f"- **Mean faithfulness:** {report.mean_faithfulness:.4f}",
        f"",
        f"## Per-Chunk Scores",
        f"",
        f"| # | Chunk ID | Relevancy | Faithfulness | Preview |",
        f"|---|----------|-----------|--------------|---------|",
    ]
    for i, cs in enumerate(report.chunk_scores, 1):
        preview = cs.content_preview.replace("|", "\\|")[:80]
        lines.append(
            f"| {i} | `{cs.chunk_id[:30]}…` | {cs.relevancy:.4f} | "
            f"{cs.faithfulness:.4f} | {preview}… |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved eval markdown: %s", path)
    return path
