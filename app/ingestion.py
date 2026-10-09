from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

try:
    from .config import settings
except ImportError:
    from config import settings


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0

    @property
    def key(self) -> str:
        return hashlib.sha256(self.text.strip().lower().encode()).hexdigest()


def extract_document(path: Path) -> list[tuple[str, int | None, str | None]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None, None)]
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        return [(soup.get_text("\n", strip=True), None, None)]
    if suffix == ".pdf":
        from pypdf import PdfReader
        return [(page.extract_text() or "", i + 1, None) for i, page in enumerate(PdfReader(str(path)).pages)]
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return [("\n".join(p.text for p in doc.paragraphs), None, None)]
    raise ValueError(f"Unsupported file type: {suffix}")


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    step = size - overlap
    return [text[i:i + size].strip() for i in range(0, len(text), step) if text[i:i + size].strip()]


def _recursive(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.extend(_fixed(current, size, overlap) if len(current) > size else [current])
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{paragraph}".strip()
        else:
            current = f"{current}\n{paragraph}".strip()
    if current:
        chunks.extend(_fixed(current, size, overlap) if len(current) > size else [current])
    return chunks


def _semantic(text: str, size: int) -> list[str]:
    # Lightweight semantic boundary approximation: headings and paragraph/topic shifts.
    blocks = [b.strip() for b in re.split(r"(?=^#{1,6}\s)|\n\s*\n", text, flags=re.M) if b.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) > size:
            chunks.append(current.strip())
            current = ""
        current = f"{current}\n{block}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, strategy: str, size: int, overlap: int) -> list[str]:
    strategy = strategy.lower()
    if strategy == "fixed":
        return _fixed(text, size, overlap)
    if strategy == "semantic":
        return _semantic(text, size)
    if strategy == "recursive":
        return _recursive(text, size, overlap)
    raise ValueError("strategy must be fixed, recursive, or semantic")


def ingest_paths(paths: Iterable[Path], strategy: str | None = None) -> list[Chunk]:
    result: list[Chunk] = []
    strategy = strategy or settings.chunk_strategy
    for path in paths:
        for text, page, section in extract_document(path):
            parts = chunk_text(text, strategy, settings.chunk_size, settings.chunk_overlap)
            for i, part in enumerate(parts):
                result.append(Chunk(part, path.name, page, section, i))
    return deduplicate(result)


def deduplicate(chunks: list[Chunk]) -> list[Chunk]:
    unique: list[Chunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.key in seen:
            continue
        seen.add(chunk.key)
        unique.append(chunk)
    return unique
