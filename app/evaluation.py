from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .config import settings
    from .ingestion import ingest_paths
except ImportError:
    from config import settings
    from ingestion import ingest_paths


def load_dataset(path: str | None = None) -> list[dict[str, Any]]:
    source = Path(path or "data/eval/golden.jsonl")
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(answer_fn, dataset: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in dataset:
        response = answer_fn(item["question"])
        expected = set(item.get("expected_sources", []))
        actual = {c["source"] for c in response["citations"]}
        rows.append({"question": item["question"], "correct": bool(actual & expected) if expected else not response["abstained"], "response": response})
    return {"count": len(rows), "accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)), "rows": rows}


def compare_strategies(paths: list[Path], answer_factory) -> dict[str, Any]:
    report = {}
    for strategy in ("fixed", "recursive", "semantic"):
        chunks = ingest_paths(paths, strategy)
        report[strategy] = {"chunks": len(chunks), "evaluation": evaluate(answer_factory(strategy), load_dataset())}
    return report
