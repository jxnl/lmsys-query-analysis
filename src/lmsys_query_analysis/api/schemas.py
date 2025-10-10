"""Pydantic schemas for FastAPI request/response models.

Based on FASTAPI_SPEC.md with field names using snake_case to match SQLModel layer.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ===== Base Classes & Common Patterns =====


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic pagination wrapper for list responses."""

    items: List[T]
    total: int
    page: int
    pages: int
    limit: int


class OperationResponse(BaseModel):
    """Generic operation response."""

    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# ===== Query Models =====


class QueryResponse(BaseModel):
    """Single query response."""

    id: int
    conversation_id: str
    model: str
    query_text: str
    language: Optional[str] = None
    timestamp: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ClusterInfo(BaseModel):
    """Cluster info attached to a query."""

    run_id: str
    cluster_id: int
    title: Optional[str] = None
    confidence_score: Optional[float] = None


class QueryDetailResponse(BaseModel):
    """Query with its cluster assignments."""

    query: QueryResponse
    clusters: List[ClusterInfo]


class PaginatedQueriesResponse(PaginatedResponse[QueryResponse]):
    """Paginated list of queries."""

    pass


# ===== Clustering Run Models =====


class ClusteringRunSummary(BaseModel):
    """Summary of a clustering run."""

    run_id: str
    algorithm: str
    num_clusters: Optional[int] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    created_at: datetime
    status: Literal["pending", "running", "completed", "failed"] = "completed"

    class Config:
        from_attributes = True


class ClusteringRunDetail(ClusteringRunSummary):
    """Detailed clustering run with metrics."""

    metrics: Optional[Dict[str, Any]] = None
    latest_errors: Optional[List[str]] = None


class ClusteringRunListResponse(PaginatedResponse[ClusteringRunSummary]):
    """Paginated list of clustering runs."""

    pass


class ClusteringRunStatusResponse(BaseModel):
    """Status of a clustering run (for polling)."""

    run_id: str
    status: str
    processed: Optional[int] = None


# ===== Cluster Summary Models =====


class ClusterSummaryResponse(BaseModel):
    """Cluster summary with LLM-generated metadata."""

    run_id: str
    cluster_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    num_queries: Optional[int] = None
    representative_queries: Optional[List[str]] = None
    summary_run_id: Optional[str] = None
    alias: Optional[str] = None
    # Enhanced aggregations
    query_count: Optional[int] = None  # Alias for num_queries
    percentage: Optional[float] = None  # Percentage of total queries

    class Config:
        from_attributes = True


class ClusterListResponse(PaginatedResponse[ClusterSummaryResponse]):
    """Paginated list of clusters with optional aggregations."""

    total_queries: Optional[int] = None  # Total queries in run (for percentage calc)


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
    parent_cluster_id: Optional[int] = None
    level: int
    children_ids: List[int]
    title: Optional[str] = None
    description: Optional[str] = None
    # Enhanced aggregations
    query_count: Optional[int] = None
    percentage: Optional[float] = None

    class Config:
        from_attributes = True


class HierarchyRunInfo(BaseModel):
    """Metadata about a hierarchy run."""

    hierarchy_run_id: str
    run_id: str
    created_at: datetime


class HierarchyTreeResponse(BaseModel):
    """Full hierarchy tree with all nodes."""

    nodes: List[HierarchyNode]
    total_queries: Optional[int] = None  # For percentage calculations


class HierarchyListResponse(PaginatedResponse[HierarchyRunInfo]):
    """Paginated list of hierarchy runs."""

    pass


# ===== Summary Run Models =====


class SummaryRunSummary(BaseModel):
    """Summary of a summarization run."""

    summary_run_id: str
    run_id: str
    alias: Optional[str] = None
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
    clusters: List[ClusterInfo]
    distance: Optional[float] = None  # For semantic search


class ClusterSearchResult(BaseModel):
    """Cluster search result."""

    run_id: str
    cluster_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    num_queries: Optional[int] = None
    distance: Optional[float] = None  # For semantic search


class FacetBucket(BaseModel):
    """Single facet bucket with count."""

    key: Any  # cluster_id (int) or string key
    count: int
    percentage: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None  # Additional metadata (e.g., cluster title)


class SearchFacets(BaseModel):
    """Faceted search results."""

    clusters: Optional[List[FacetBucket]] = None
    language: Optional[List[FacetBucket]] = None
    model: Optional[List[FacetBucket]] = None


class SearchQueriesResponse(PaginatedResponse[QuerySearchResult]):
    """Query search results with facets."""

    facets: Optional[SearchFacets] = None
    applied_clusters: Optional[List[ClusterSearchResult]] = None  # For within_clusters


class SearchClustersResponse(PaginatedResponse[ClusterSearchResult]):
    """Cluster search results."""

    pass


# ===== Curation Models (Read-Only) =====


class ClusterMetadata(BaseModel):
    """Cluster quality metadata."""

    coherence_score: Optional[int] = None
    quality: Optional[Literal["high", "medium", "low"]] = None
    notes: Optional[str] = None
    flags: Optional[List[str]] = None
    last_edited: Optional[datetime] = None

    class Config:
        from_attributes = True


class EditHistoryRecord(BaseModel):
    """Single edit history record."""

    timestamp: datetime
    cluster_id: Optional[int] = None
    edit_type: str
    editor: str
    reason: Optional[str] = None
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class EditHistoryResponse(PaginatedResponse[EditHistoryRecord]):
    """Paginated edit history."""

    pass


class OrphanInfo(BaseModel):
    """Information about an orphaned query."""

    orphan: Dict[str, Any]  # OrphanedQuery fields
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
