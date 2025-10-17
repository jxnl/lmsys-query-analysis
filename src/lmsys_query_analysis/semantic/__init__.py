"""Semantic SDK stubs for cluster and query search.

This package exposes typed client interfaces for semantic workflows.
Implementations will be added incrementally; current stubs define method
signatures and docstrings for discussion and scaffolding.
"""

from .clusters import ClustersClient
from .queries import QueriesClient
from .types import ClusterHit, FacetBucket, QueryHit, RunSpace, SearchResult

__all__ = [
    "RunSpace",
    "ClusterHit",
    "QueryHit",
    "FacetBucket",
    "SearchResult",
    "ClustersClient",
    "QueriesClient",
]
