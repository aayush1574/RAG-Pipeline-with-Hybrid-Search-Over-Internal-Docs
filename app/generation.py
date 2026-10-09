from __future__ import annotations

import re
from typing import Any

try:
    from .providers import complete, embed, rewrite_query
    from .retrieval import HybridRetriever
    from .schemas import Citation
except ImportError:
    from providers import complete, embed, rewrite_query
    from retrieval import HybridRetriever
    from schemas import Citation


def answer_question(retriever: HybridRetriever, question: str, top_k: int | None = None,
                    semantic_weight: float | None = None, sparse_weight: float | None = None) -> dict[str, Any]:
    if not retriever.chunks:
        return {"answer": "I don't know based on the indexed documents.", "citations": [],
                "confidence": {"overall": 0.0, "retrieval": 0.0, "citation_coverage": 0.0, "completeness": 0.0, "label": "low"},
                "abstained": True, "retrieved_chunks": []}
    retrieval_query = rewrite_query(question)
    query_vector = embed([retrieval_query])[0]
    results = retriever.search(retrieval_query, query_vector, top_k, semantic_weight, sparse_weight)
    initial_results = results
    reranked = retriever.rerank(retrieval_query, results)
    reranker_active = any("rerank_score" in item for item in reranked)
    # Retry with the user's exact wording when rewriting weakens retrieval.
    retry_used = False
    if (not reranked or reranked[0]["rrf_score"] < 0.005) and retrieval_query != question:
        retry_used = True
        original_vector = embed([question])[0]
        reranked = retriever.rerank(question, retriever.search(question, original_vector, top_k, semantic_weight, sparse_weight))
    if not reranked or reranked[0]["rrf_score"] < 0.005:
        return {"answer": "I don't know based on the indexed documents.", "citations": [],
                "confidence": {"overall": 0.0, "retrieval": 0.0, "citation_coverage": 0.0, "completeness": 0.0, "label": "low"},
                "abstained": True, "retrieved_chunks": reranked,
                "retrieval_trace": {"original_query": question, "rewritten_query": retrieval_query,
                                     "retry_used": retry_used, "reranker_active": reranker_active,
                                     "candidate_count": len(initial_results)}}
    results = reranked
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
            "abstained": False, "retrieved_chunks": results,
            "retrieval_trace": {"original_query": question, "rewritten_query": retrieval_query,
                                 "retry_used": retry_used, "reranker_active": reranker_active,
                                 "candidate_count": len(initial_results)}}
