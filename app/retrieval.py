from __future__ import annotations

import re
import json
from collections import defaultdict
from typing import Any

import numpy as np
try:
    from rank_bm25 import BM25Okapi
except ImportError:  # keeps the API importable before optional dependencies are installed
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
        def get_scores(self, query):
            query = set(query)
            return np.array([sum(token in query for token in doc) for doc in self.corpus], dtype=float)

try:
    from .config import settings
    from .ingestion import Chunk
except ImportError:
    from config import settings
    from ingestion import Chunk


class HybridRetriever:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        self.chunks.extend(chunks)
        self.bm25 = BM25Okapi([re.findall(r"\w+", c.text.lower()) for c in self.chunks]) if self.chunks else None
        if vectors is not None:
            new = np.asarray(vectors, dtype=np.float32)
            self.embeddings = new if self.embeddings is None else np.vstack([self.embeddings, new])

    def save(self, path) -> None:
        """Persist the index in a portable JSON file for single-process deployments."""
        payload = self.payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)

    def payload(self) -> dict:
        return {
            "chunks": [{"text": c.text, "source": c.source, "page": c.page,
                        "section": c.section, "chunk_index": c.chunk_index} for c in self.chunks],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
        }

    def load_payload(self, payload: dict) -> None:
        chunks = [Chunk(**item) for item in payload.get("chunks", [])]
        embeddings = payload.get("embeddings") or None
        self.chunks = []
        self.embeddings = None
        self.bm25 = None
        self.add(chunks, embeddings)

    def load(self, path) -> None:
        if not path.exists():
            return
        self.load_payload(json.loads(path.read_text(encoding="utf-8")))

    def search(self, question: str, query_vector: list[float] | None = None, top_k: int | None = None,
               semantic_weight: float | None = None, sparse_weight: float | None = None) -> list[dict[str, Any]]:
        top_k = top_k or settings.top_k
        sw = settings.semantic_weight if semantic_weight is None else semantic_weight
        kw = settings.sparse_weight if sparse_weight is None else sparse_weight
        dense: list[int] = []
        if query_vector is not None and self.embeddings is not None and len(self.chunks):
            q = np.asarray(query_vector, dtype=np.float32)
            denom = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q)
            scores = np.divide(self.embeddings @ q, denom, out=np.zeros(len(self.chunks)), where=denom != 0)
            dense = [int(value) for value in np.argsort(-scores)[:top_k]]
        sparse: list[int] = []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(re.findall(r"\w+", question.lower()))
            sparse = [int(value) for value in np.argsort(-scores)[:top_k]]
        fused: dict[int, float] = defaultdict(float)
        details: dict[int, dict[str, float]] = defaultdict(lambda: {"semantic_rank": 0, "sparse_rank": 0})
        for rank, idx in enumerate(dense, 1):
            fused[idx] += sw / (settings.rrf_k + rank)
            details[idx]["semantic_rank"] = rank
        for rank, idx in enumerate(sparse, 1):
            fused[idx] += kw / (settings.rrf_k + rank)
            details[idx]["sparse_rank"] = rank
        ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [{"id": i, "text": self.chunks[i].text, "source": self.chunks[i].source,
                 "page": self.chunks[i].page, "section": self.chunks[i].section,
                 "rrf_score": fused[i], **details[i]} for i in ordered]

    def rerank(self, question: str, results: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
        limit = limit or settings.rerank_top_k
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(settings.reranker_model)
            scores = model.predict([(question, item["text"]) for item in results])
            for item, score in zip(results, scores):
                item["rerank_score"] = float(score)
            return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)[:limit]
        except Exception:
            # Reranking is optional; fused ranking remains a valid fallback.
            return results[:limit]
