"""Pydantic schemas for FastAPI request/response models.

Based on FASTAPI_SPEC.md with field names using snake_case to match SQLModel layer.
"""

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


# ===== Base Classes & Common Patterns =====


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic pagination wrapper for list responses."""

    items: list[T]
    total: int
    page: int
    pages: int
    limit: int


class OperationResponse(BaseModel):
    """Generic operation response."""

    status: str
    message: str | None = None
    data: dict[str, Any] | None = None


# ===== Query Models =====


class QueryResponse(BaseModel):
    """Single query response."""

    id: int
    conversation_id: str
    model: str
    query_text: str
    language: str | None = None
    timestamp: datetime | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ClusterInfo(BaseModel):
    """Cluster info attached to a query."""

    run_id: str
    cluster_id: int
    title: str | None = None
    confidence_score: float | None = None


class QueryDetailResponse(BaseModel):
    """Query with its cluster assignments."""

    query: QueryResponse
    clusters: list[ClusterInfo]


class PaginatedQueriesResponse(PaginatedResponse[QueryResponse]):
    """Paginated list of queries."""

    pass


# ===== Clustering Run Models =====


class ClusteringRunSummary(BaseModel):
    """Summary of a clustering run."""

    run_id: str
    algorithm: str
    num_clusters: int | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None
    created_at: datetime
    status: Literal["pending", "running", "completed", "failed"] = "completed"

    class Config:
        from_attributes = True


class ClusteringRunDetail(ClusteringRunSummary):
    """Detailed clustering run with metrics."""

    metrics: dict[str, Any] | None = None
    latest_errors: list[str] | None = None


class ClusteringRunListResponse(PaginatedResponse[ClusteringRunSummary]):
    """Paginated list of clustering runs."""

    pass


class ClusteringRunStatusResponse(BaseModel):
    """Status of a clustering run (for polling)."""

    run_id: str
    status: str
    processed: int | None = None


# ===== Cluster Summary Models =====


class ClusterSummaryResponse(BaseModel):
    """Cluster summary with LLM-generated metadata."""

    run_id: str
    cluster_id: int
    title: str | None = None
    description: str | None = None
    summary: str | None = None
    num_queries: int | None = None
    representative_queries: list[str] | None = None
    summary_run_id: str | None = None
    alias: str | None = None
    # Enhanced aggregations
    query_count: int | None = None  # Alias for num_queries
    percentage: float | None = None  # Percentage of total queries

    class Config:
        from_attributes = True


class ClusterListResponse(PaginatedResponse[ClusterSummaryResponse]):
    """Paginated list of clusters with optional aggregations."""

    total_queries: int | None = None  # Total queries in run (for percentage calc)


class ClusterDetailResponse(BaseModel):
    """Detailed cluster view with paginated queries."""

    cluster: ClusterSummaryResponse
    queries: PaginatedQueriesResponse


# ===== Hierarchy Models =====


class HierarchyNode(BaseModel):
    """Single node in a cluster hierarchy."""

    hierarchy_run_id: str
    run_id: str
    cluster_id: int
    parent_cluster_id: int | None = None
    level: int
    children_ids: list[int]
    title: str | None = None
    description: str | None = None
    # Enhanced aggregations
    query_count: int | None = None
    percentage: float | None = None

    class Config:
        from_attributes = True


class HierarchyRunInfo(BaseModel):
    """Metadata about a hierarchy run."""

    hierarchy_run_id: str
    run_id: str
    created_at: datetime


class HierarchyTreeResponse(BaseModel):
    """Full hierarchy tree with all nodes."""

    nodes: list[HierarchyNode]
    total_queries: int | None = None  # For percentage calculations


class HierarchyListResponse(PaginatedResponse[HierarchyRunInfo]):
    """Paginated list of hierarchy runs."""

    pass


# ===== Summary Run Models =====


class SummaryRunSummary(BaseModel):
    """Summary of a summarization run."""

    summary_run_id: str
    run_id: str
    alias: str | None = None
    model: str
    generated_at: datetime
    status: Literal["pending", "running", "completed", "failed"] = "completed"

    class Config:
        from_attributes = True


class SummaryRunListResponse(PaginatedResponse[SummaryRunSummary]):
    """Paginated list of summary runs."""

    pass


# ===== Search Models =====


class QuerySearchResult(BaseModel):
    """Query search result with cluster context."""

    query: QueryResponse
    clusters: list[ClusterInfo]
    distance: float | None = None  # For semantic search


class ClusterSearchResult(BaseModel):
    """Cluster search result."""

    run_id: str
    cluster_id: int
    title: str | None = None
    description: str | None = None
    summary: str | None = None
    num_queries: int | None = None
    distance: float | None = None  # For semantic search


class FacetBucket(BaseModel):
    """Single facet bucket with count."""

    key: Any  # cluster_id (int) or string key
    count: int
    percentage: float | None = None
    meta: dict[str, Any] | None = None  # Additional metadata (e.g., cluster title)


class SearchFacets(BaseModel):
    """Faceted search results."""

    clusters: list[FacetBucket] | None = None
    language: list[FacetBucket] | None = None
    model: list[FacetBucket] | None = None


class SearchQueriesResponse(PaginatedResponse[QuerySearchResult]):
    """Query search results with facets."""

    facets: SearchFacets | None = None
    applied_clusters: list[ClusterSearchResult] | None = None  # For within_clusters


class SearchClustersResponse(PaginatedResponse[ClusterSearchResult]):
    """Cluster search results."""

    pass


# ===== Curation Models (Read-Only) =====


class ClusterMetadata(BaseModel):
    """Cluster quality metadata."""

    coherence_score: int | None = None
    quality: Literal["high", "medium", "low"] | None = None
    notes: str | None = None
    flags: list[str] | None = None
    last_edited: datetime | None = None

    class Config:
        from_attributes = True


class EditHistoryRecord(BaseModel):
    """Single edit history record."""

    timestamp: datetime
    cluster_id: int | None = None
    edit_type: str
    editor: str
    reason: str | None = None
    old_value: dict[str, Any] | None = None
    new_value: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class EditHistoryResponse(PaginatedResponse[EditHistoryRecord]):
    """Paginated edit history."""

    pass


class OrphanInfo(BaseModel):
    """Information about an orphaned query."""

    orphan: dict[str, Any]  # OrphanedQuery fields
    query: QueryResponse


class OrphanedQueriesResponse(PaginatedResponse[OrphanInfo]):
    """Paginated orphaned queries."""

    pass


# ===== Error Response =====


class ErrorDetail(BaseModel):
    """Error detail structure."""

    type: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail
