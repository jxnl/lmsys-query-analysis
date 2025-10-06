from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class RunSpace(BaseModel):
    """Resolved vector-space and run context.

    - embedding_provider/model/dimension define the vector space used for Chroma collections
    - run_id optionally anchors searches to a clustering run for provenance
    """

    embedding_provider: str
    embedding_model: str
    embedding_dimension: Optional[int] = None
    run_id: Optional[str] = None


class ClusterHit(BaseModel):
    """A ranked cluster result from summary search."""

    cluster_id: int
    distance: float
    title: Optional[str] = None
    description: Optional[str] = None
    num_queries: Optional[int] = None


class QueryHit(BaseModel):
    """A ranked query result from query search."""

    query_id: int
    distance: float
    snippet: str
    model: Optional[str] = None
    language: Optional[str] = None
    cluster_id: Optional[int] = None


class FacetBucket(BaseModel):
    """A single facet bucket with optional metadata."""

    key: Union[str, int]
    count: int
    meta: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Canonical JSON-friendly result shape for CLI/SDK."""

    text: str
    run_id: Optional[str] = None
    applied_clusters: List[ClusterHit] = Field(default_factory=list)
    results: List[QueryHit] = Field(default_factory=list)
    facets: Dict[str, List[FacetBucket]] = Field(default_factory=dict)
