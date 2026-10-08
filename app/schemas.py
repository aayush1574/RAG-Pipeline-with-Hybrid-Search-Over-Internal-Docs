from typing import Any
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    chunk_strategy: str | None = None
    semantic_weight: float | None = Field(default=None, ge=0, le=1)
    sparse_weight: float | None = Field(default=None, ge=0, le=1)


class Citation(BaseModel):
    number: int
    source: str
    page: int | None = None
    section: str | None = None
    supported: bool = False
    evidence: str = ""


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: dict[str, float | str]
    abstained: bool
    retrieved_chunks: list[dict[str, Any]]


class EvaluationRequest(BaseModel):
    dataset_path: str | None = None
    strategy: str | None = None
