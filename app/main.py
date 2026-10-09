from pathlib import Path
import os
import tempfile
import shutil
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

try:
    from .config import settings
    from .evaluation import evaluate, load_dataset
    from .generation import answer_question
    from .ingestion import ingest_paths
    from .providers import embed
    from .retrieval import HybridRetriever
    from .schemas import AskRequest, EvaluationRequest
except ImportError:
    from config import settings
    from evaluation import evaluate, load_dataset
    from generation import answer_question
    from ingestion import ingest_paths
    from providers import embed
    from retrieval import HybridRetriever
    from schemas import AskRequest, EvaluationRequest

app = FastAPI(title="Internal Docs Hybrid RAG", version="1.0.0")
retriever = HybridRetriever()
documents: dict[str, int] = {}
index_path = settings.data_dir / "index.json"
retriever.load(index_path)
for chunk in retriever.chunks:
    documents[chunk.source] = documents.get(chunk.source, 0) + 1


@app.get("/healthz", tags=["operations"])
def healthz():
    return {"status": "ok"}


@app.get("/readyz", tags=["operations"])
def readyz():
    return {"status": "ready", "indexed_chunks": len(retriever.chunks), "provider_configured": bool(settings.openai_api_key),
            "features": {"query_rewrite": bool(settings.openai_api_key), "fallback_embeddings": not bool(settings.openai_api_key),
                         "reranking": True, "persistent_index": True}}


@app.get("/v1/documents")
def list_documents():
    return {"documents": [{"source": k, "chunks": v} for k, v in documents.items()], "total_chunks": len(retriever.chunks)}


@app.post("/v1/ingest")
async def ingest(files: list[UploadFile] = File(...), chunk_strategy: str | None = None):
    paths = []
    work_dir = Path(tempfile.mkdtemp(prefix="rag-upload-"))
    try:
        if not files or len(files) > 20:
            raise HTTPException(400, "Upload between 1 and 20 files")
        for upload in files:
            original_name = Path(upload.filename or "document").name
            suffix = Path(original_name).suffix.lower()
            if suffix not in {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}:
                raise HTTPException(400, f"Unsupported file type: {suffix}")
            content = await upload.read()
            if len(content) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, f"{original_name} exceeds the {settings.max_upload_mb} MB upload limit")
            safe_name = f"{uuid4().hex}_{original_name}"
            path = work_dir / safe_name
            path.write_bytes(content)
            paths.append(path)
        chunks = ingest_paths(paths, chunk_strategy)
        if not chunks:
            raise HTTPException(400, "No text could be extracted")
        vectors = embed([c.text for c in chunks])
        retriever.add(chunks, vectors)
        # Temporary filenames contain a random prefix; restore the original name in metadata.
        for c in chunks:
            c.source = c.source.split("_", 1)[-1]
            documents[c.source] = documents.get(c.source, 0) + 1
        retriever.save(index_path)
        return {"indexed_chunks": len(chunks), "documents": documents}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/v1/ask")
def ask(request: AskRequest):
    try:
        return answer_question(retriever, request.question, request.top_k, request.semantic_weight, request.sparse_weight)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/v1/evaluate")
def run_evaluation(request: EvaluationRequest):
    try:
        return evaluate(lambda q: answer_question(retriever, q), load_dataset(request.dataset_path))
    except Exception as exc:
        raise HTTPException(400, str(exc))


frontend = Path(__file__).parent.parent / "frontend"
if frontend.exists() and os.getenv("CLOUDFLARE_WORKER") != "1":
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
