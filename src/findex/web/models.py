"""Pydantic models at the HTTP boundary. Inside, findex keeps its dataclasses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

type Mode = Literal["keyword", "semantic", "hybrid"]


class SearchHit(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str = Field(description="Plain-text snippet.")
    snippet_html: str = Field(description="Snippet with matches wrapped in <mark>, HTML-escaped.")


class SearchResponse(BaseModel):
    query: str
    mode: Mode
    scorer: Literal["bm25", "tfidf"]
    total: int = Field(description="Documents matching the query (before paging).")
    page: int
    k: int
    pages: int
    took_ms: float
    results: list[SearchHit]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int = Field(description="Length in tokens.")
    text: str


class StatsResponse(BaseModel):
    version: str
    documents: int
    terms: int
    total_tokens: int
    avg_doc_length: float
    positions: bool
    index_bytes: int
    semantic: bool
    uptime_s: float


class Health(BaseModel):
    status: Literal["ok"]


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
