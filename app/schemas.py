from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["docs"])
    clear_existing: bool = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer using the indexed knowledge base.")
    top_k: int = Field(default=5, ge=1, le=10)


class SourceItem(BaseModel):
    file: str
    score: float
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    question: str
    top_k: int
    model: str | None = None
