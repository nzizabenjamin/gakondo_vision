# Gakondo-Vision 🏛️

**Multimodal Cultural Heritage RAG Engine** — Preserving and exploring Rwanda's intangible cultural heritage through AI-powered search.

## What It Does

Gakondo-Vision ingests historical text documents, folklore archives, and visual artifacts (traditional basketry, Imigongo art patterns) into a unified vector embedding space. Users can search across modalities — ask a text question and retrieve relevant images, or upload an image to find related textual descriptions.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | CLIP ViT-B/32 (512-d joint text-image space) |
| Vector DB | PostgreSQL + pgvector |
| Backend | FastAPI |
| Frontend | Streamlit |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Evaluation | Ragas-inspired faithfulness & relevancy scoring |

## Quick Start

### 1. Prerequisites
- Python 3.10+
- PostgreSQL 15+ with `pgvector` extension

### 2. Setup

```bash
# Clone & install
cd gakondo_vision
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# Initialise the database
psql -U gakondo -d gakondo_vision -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Add Data
- Place PDF/TXT documents in `data/raw_texts/`
- Place images (JPG/PNG) in `data/raw_images/`
- Optionally add JSON sidecar metadata files alongside images

### 4. Run

```bash
# Start the API server
python -m src.api

# In a separate terminal, start the Streamlit UI
streamlit run app.py --server.port 8501
```

### 5. Ingest & Search
- Click **"Run Ingestion Pipeline"** in the Streamlit sidebar
- Type a query or upload an image to search

## Project Structure

```
src/
├── config.py       # Pydantic settings (DB, model, chunk params)
├── ingestion.py    # PDF/TXT/image loading pipeline
├── chunker.py      # Semantic chunking with cultural metadata
├── embeddings.py   # CLIP text + image encoding
├── db.py           # pgvector schema, upsert, similarity search
├── retriever.py    # High-level retrieval API
├── api.py          # FastAPI endpoints
└── evaluation.py   # Ragas-style evaluation scoring
app.py              # Streamlit frontend
```

## License

MIT
