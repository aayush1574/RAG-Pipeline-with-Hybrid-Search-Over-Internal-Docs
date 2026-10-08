# RAG Pipeline with Hybrid Search Over Internal Docs

Production-oriented reference implementation for the attached project brief.

## Included

- PDF, Markdown, text, HTML, and DOCX ingestion with source/page/section metadata.
- Fixed-size, recursive, and semantic chunking strategies.
- OpenAI-compatible embeddings, a portable persistent vector index, BM25 sparse search, weighted RRF fusion.
- Optional cross-encoder reranking.
- Grounded generation with numbered citations, citation verification, confidence scoring, and an explicit abstention path.
- Evaluation harness with a starter golden dataset and chunking-strategy comparison.
- FastAPI endpoints, a small browser dashboard, and Docker Compose.
- Atomic on-disk index persistence under `data/index.json`, so indexed documents survive API restarts.

## Quick start

```bash
copy .env.example .env
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000`. Set `OPENAI_API_KEY` in `.env` for embeddings and generation.

## API

- `POST /v1/ingest` — multipart upload; accepts one or more documents.
- `GET /v1/documents` — indexed source summary.
- `POST /v1/ask` — `{ "question": "..." }` with answer, citations, confidence, and retrieval details.
- `POST /v1/evaluate` — run the bundled or supplied JSONL golden dataset.
- `GET /healthz` and `GET /readyz` — container/orchestrator health checks.

The implementation is intentionally provider-configurable: any OpenAI-compatible endpoint can be selected with `OPENAI_BASE_URL`, `EMBEDDING_MODEL`, and `GENERATION_MODEL`.
