from typing import Any
from pydantic import BaseModel, Field, model_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    chunk_strategy: str | None = None
    semantic_weight: float | None = Field(default=None, ge=0, le=1)
    sparse_weight: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_weights(self):
        if self.semantic_weight is not None and self.sparse_weight is not None:
            if abs(self.semantic_weight + self.sparse_weight - 1.0) > 0.001:
                raise ValueError("semantic_weight and sparse_weight must sum to 1.0")
        return self


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
