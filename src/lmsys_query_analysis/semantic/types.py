from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunSpace(BaseModel):
    """Resolved vector-space and run context.

    - embedding_provider/model/dimension define the vector space used for Chroma collections
    - run_id optionally anchors searches to a clustering run for provenance
    """

    embedding_provider: str
    embedding_model: str
    embedding_dimension: int | None = None
    run_id: str | None = None


class ClusterHit(BaseModel):
    """A ranked cluster result from summary search."""

    cluster_id: int
    distance: float
    title: str | None = None
    description: str | None = None
    num_queries: int | None = None


class QueryHit(BaseModel):
    """A ranked query result from query search."""

    query_id: int
    distance: float
    snippet: str
    model: str | None = None
    language: str | None = None
    cluster_id: int | None = None


class FacetBucket(BaseModel):
    """A single facet bucket with optional metadata."""

    key: str | int
    count: int
    meta: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Canonical JSON-friendly result shape for CLI/SDK."""

    text: str
    run_id: str | None = None
    applied_clusters: list[ClusterHit] = Field(default_factory=list)
    results: list[QueryHit] = Field(default_factory=list)
    facets: dict[str, list[FacetBucket]] = Field(default_factory=dict)
