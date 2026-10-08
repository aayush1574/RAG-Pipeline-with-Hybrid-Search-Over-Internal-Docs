from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from .config import settings
from .evaluation import evaluate, load_dataset
from .generation import answer_question
from .ingestion import ingest_paths
from .providers import embed
from .retrieval import HybridRetriever
from .schemas import AskRequest, EvaluationRequest

app = FastAPI(title="Internal Docs Hybrid RAG", version="1.0.0")
retriever = HybridRetriever()
documents: dict[str, int] = {}


@app.get("/v1/documents")
def list_documents():
    return {"documents": [{"source": k, "chunks": v} for k, v in documents.items()], "total_chunks": len(retriever.chunks)}


@app.post("/v1/ingest")
async def ingest(files: list[UploadFile] = File(...), chunk_strategy: str | None = None):
    paths = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}:
                raise HTTPException(400, f"Unsupported file type: {suffix}")
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp.write(await upload.read()); temp.close(); paths.append(Path(temp.name))
        chunks = ingest_paths(paths, chunk_strategy)
        if not chunks:
            raise HTTPException(400, "No text could be extracted")
        vectors = embed([c.text for c in chunks])
        retriever.add(chunks, vectors)
        for c in chunks: documents[c.source] = documents.get(c.source, 0) + 1
        return {"indexed_chunks": len(chunks), "documents": documents}
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


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
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
