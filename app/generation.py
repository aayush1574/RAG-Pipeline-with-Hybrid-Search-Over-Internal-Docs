from __future__ import annotations

import re
from typing import Any

from .providers import complete, embed
from .retrieval import HybridRetriever
from .schemas import Citation


def answer_question(retriever: HybridRetriever, question: str, top_k: int | None = None,
                    semantic_weight: float | None = None, sparse_weight: float | None = None) -> dict[str, Any]:
    query_vector = embed([question])[0]
    results = retriever.search(question, query_vector, top_k, semantic_weight, sparse_weight)
    results = retriever.rerank(question, results)
    if not results or results[0]["rrf_score"] < 0.005:
        return {"answer": "I don't know based on the indexed documents.", "citations": [],
                "confidence": {"overall": 0.0, "retrieval": 0.0, "citation_coverage": 0.0, "completeness": 0.0, "label": "low"},
                "abstained": True, "retrieved_chunks": results}
    context = "\n\n".join(f"[{i}] {r['text']}" for i, r in enumerate(results, 1))
    prompt = ("Answer only from the numbered context. Cite every factual claim with [n]. "
              "If the context is insufficient, say you don't know.\n\nCONTEXT:\n" + context + "\n\nQUESTION:\n" + question)
    text = complete("You are a careful internal-document assistant. Never invent facts or citations.", prompt)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(results)})
    citations = [Citation(number=i, source=results[i-1]["source"], page=results[i-1]["page"], section=results[i-1]["section"], supported=True, evidence=results[i-1]["text"][:300]) for i in cited]
    retrieval = min(1.0, max(0.0, float(results[0]["rrf_score"]) * 100))
    coverage = len(citations) / max(1, len(re.findall(r"(?<=[.!?])\s+", text)) + 1)
    completeness = 0.8 if "don't know" not in text.lower() else 0.3
    overall = round((retrieval + min(1, coverage) + completeness) / 3, 3)
    return {"answer": text, "citations": [c.model_dump() for c in citations],
            "confidence": {"overall": overall, "retrieval": round(retrieval, 3), "citation_coverage": round(min(1, coverage), 3), "completeness": completeness, "label": "high" if overall >= .75 else "medium" if overall >= .45 else "low"},
            "abstained": False, "retrieved_chunks": results}
