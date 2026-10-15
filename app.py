"""
Streamlit frontend for Gakondo-Vision.

A rich, interactive interface for exploring Rwandan cultural heritage
through multimodal search. Supports text queries, image uploads,
metadata filtering, and result cards with similarity scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

from src.config import settings

# ── Page configuration ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Gakondo-Vision — Cultural Heritage Explorer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark glassmorphic header ──────────────────────────── */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        background: linear-gradient(90deg, #e94560, #f5a623, #e94560);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer {
        to { background-position: 200% center; }
    }
    .hero-header p {
        color: rgba(255,255,255,0.7);
        font-size: 1.05rem;
        margin: 0;
    }

    /* ── Result card ───────────────────────────────────────── */
    .result-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233,69,96,0.15);
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #e94560, #f5a623);
        color: #fff;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .meta-tag {
        display: inline-block;
        background: rgba(233,69,96,0.12);
        color: #e94560;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1>🏛️ Gakondo-Vision</h1>
    <p>Explore Rwanda's intangible cultural heritage through multimodal AI-powered search</p>
</div>
""", unsafe_allow_html=True)

# ── API base URL ─────────────────────────────────────────────────────────
API_BASE = f"http://localhost:{settings.api_port}"

# ── Sidebar: filters & options ───────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔧 Search Settings")

    top_k = st.slider("Results to return", min_value=1, max_value=20, value=5)

    st.markdown("### 🏷️ Metadata Filters")

    cultural_category = st.selectbox(
        "Cultural category",
        ["Any", "oral_tradition", "art_craft", "dance_music",
         "governance", "spirituality", "general"],
    )
    region = st.selectbox(
        "Region",
        ["Any", "Rwanda", "Northern", "Southern", "Eastern", "Western", "Kigali"],
    )
    historical_era = st.selectbox(
        "Historical era",
        ["Any", "pre_colonial", "colonial", "post_independence", "unspecified"],
    )

    st.markdown("---")
    st.markdown("### ⚙️ Admin")
    if st.button("🔄 Run Ingestion Pipeline", use_container_width=True):
        with st.spinner("Ingesting documents & images…"):
            try:
                resp = requests.post(f"{API_BASE}/ingest", timeout=300)
                if resp.ok:
                    data = resp.json()
                    st.success(
                        f"✅ Loaded {data['documents_loaded']} docs → "
                        f"{data['chunks_created']} chunks, "
                        f"{data['images_catalogued']} images. "
                        f"Stored {data['chunks_stored']} records."
                    )
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Is the FastAPI server running?")

    # Health check
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        status_icon = "🟢" if health.get("database") == "connected" else "🟡"
        st.caption(f"{status_icon} API: {health.get('status')} | DB: {health.get('database')}")
    except Exception:
        st.caption("🔴 API offline")


# ── Main search area ─────────────────────────────────────────────────────

tab_text, tab_image = st.tabs(["📝 Text Search", "🖼️ Image Search"])

def _build_form_data() -> dict:
    """Build metadata filter form data."""
    data = {"top_k": top_k}
    if cultural_category != "Any":
        data["cultural_category"] = cultural_category
    if region != "Any":
        data["region"] = region
    if historical_era != "Any":
        data["historical_era"] = historical_era
    return data


def _render_results(results: list[dict]) -> None:
    """Render search result cards."""
    if not results:
        st.info("No results found. Try a different query or adjust filters.")
        return

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source_type = r.get("source_type", "text")
        metadata = r.get("metadata", {})

        st.markdown(f"""
        <div class="result-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <strong>#{i} — {r.get('id', 'unknown')[:40]}</strong>
                <span class="score-badge">{score:.4f}</span>
            </div>
        """, unsafe_allow_html=True)

        # Content or image
        if source_type == "image" and r.get("image_path"):
            img_path = Path(r["image_path"])
            if img_path.exists():
                st.image(str(img_path), width=300)
            else:
                st.caption(f"📎 Image: {img_path.name}")
        elif r.get("content"):
            st.markdown(r["content"][:500])

        # Metadata tags
        tags_html = ""
        for key in ["cultural_category", "historical_era", "region", "source_type"]:
            val = metadata.get(key, r.get(key))
            if val:
                tags_html += f'<span class="meta-tag">{key}: {val}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ── Text search tab ─────────────────────────────────────────────────────

with tab_text:
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g. What is the significance of Imigongo art patterns?",
    )
    if st.button("🔍 Search", key="text_search", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching…"):
                try:
                    form_data = _build_form_data()
                    form_data["query"] = query
                    resp = requests.post(
                        f"{API_BASE}/search", data=form_data, timeout=120
                    )
                    if resp.ok:
                        data = resp.json()
                        st.markdown(f"**{data['total']} results** for: _{data['query']}_")
                        _render_results(data["results"])
                    else:
                        st.error(f"Search failed: {resp.text}")
                except requests.exceptions.Timeout:
                    st.warning("⏱️ The search timed out while the model/database was waking up. Please click Search again.")
                except requests.ConnectionError:
                    st.error("Cannot connect to API. Is the FastAPI server running?")

# ── Image search tab ────────────────────────────────────────────────────

with tab_image:
    uploaded = st.file_uploader(
        "Upload an image to find similar cultural artifacts",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded:
        st.image(uploaded, caption="Query image", width=250)

    if st.button("🔍 Search by Image", key="img_search", use_container_width=True):
        if not uploaded:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Searching…"):
                try:
                    form_data = _build_form_data()
                    files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    resp = requests.post(
                        f"{API_BASE}/search",
                        data=form_data,
                        files=files,
                        timeout=120,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.markdown(f"**{data['total']} results** for image query")
                        _render_results(data["results"])
                    else:
                        st.error(f"Search failed: {resp.text}")
                except requests.exceptions.Timeout:
                    st.warning("⏱️ The search timed out while the model/database was waking up. Please click Search again.")
                except requests.ConnectionError:
                    st.error("Cannot connect to API. Is the FastAPI server running?")


# ── Footer ───────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Gakondo-Vision v0.1.0 · Preserving Rwanda's intangible cultural heritage "
    "through multimodal AI · Built with CLIP, pgvector, FastAPI & Streamlit"
)
