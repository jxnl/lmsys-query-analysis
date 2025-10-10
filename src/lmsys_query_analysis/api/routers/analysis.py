"""Analysis endpoints for queries and cluster details."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..dependencies import get_db
from ..schemas import (
    PaginatedQueriesResponse,
    ClusterDetailResponse,
    ClusterSummaryResponse,
    QueryResponse,
    ErrorResponse,
)
from ...db.connection import Database
from ...services import query_service, cluster_service

router = APIRouter()


@router.get(
    "/queries",
    response_model=PaginatedQueriesResponse,
    summary="List queries with optional filtering",
)
async def list_queries(
    run_id: Optional[str] = Query(None, description="Filter by run ID"),
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
    model: Optional[str] = Query(None, description="Filter by model name"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(100, ge=1, le=100, description="Items per page"),
    db: Database = Depends(get_db),
):
    """List queries with optional filtering by run_id, cluster_id, and model.

    Returns paginated list of queries with metadata.
    """
    # Get queries from service (uses existing service layer)
    all_queries = query_service.list_queries(db, run_id, cluster_id, model, limit=10000)

    # Calculate pagination
    total = len(all_queries)
    pages = (total + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit

    items = [QueryResponse.model_validate(q) for q in all_queries[start:end]]

    return PaginatedQueriesResponse(
        items=items,
        total=total,
        page=page,
        pages=pages,
        limit=limit,
    )


@router.get(
    "/clustering/runs/{run_id}/clusters/{cluster_id}",
    response_model=ClusterDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get cluster details with paginated queries",
)
async def get_cluster_detail(
    run_id: str,
    cluster_id: int,
    page: int = Query(1, ge=1, description="Page number for queries"),
    limit: int = Query(50, ge=1, le=100, description="Queries per page"),
    db: Database = Depends(get_db),
):
    """Get detailed cluster information including summary and paginated queries.

    Returns:
    - Cluster summary (title, description, representative queries)
    - Paginated list of queries in the cluster
    - Query count and percentage of run
    """
    # Get cluster summary
    summary = cluster_service.get_cluster_summary(db, run_id, cluster_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "NotFound",
                    "message": f"Cluster {cluster_id} not found in run {run_id}",
                }
            },
        )

    # Get queries in cluster
    all_queries = query_service.get_cluster_queries(db, run_id, cluster_id)

    # Calculate pagination
    total = len(all_queries)
    pages = (total + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit

    query_items = [QueryResponse.model_validate(q) for q in all_queries[start:end]]

    # Calculate percentage
    from sqlmodel import select, func
    from ...db.models import QueryCluster

    with db.get_session() as session:
        total_queries_stmt = select(func.count()).where(QueryCluster.run_id == run_id)
        total_queries = session.exec(total_queries_stmt).one()

    percentage = None
    if total_queries > 0:
        percentage = round((total / total_queries) * 100, 2)

    return ClusterDetailResponse(
        cluster=ClusterSummaryResponse(
            run_id=summary.run_id,
            cluster_id=summary.cluster_id,
            title=summary.title,
            description=summary.description,
            summary=summary.summary,
            num_queries=summary.num_queries or total,
            representative_queries=summary.representative_queries,
            summary_run_id=summary.summary_run_id,
            alias=summary.alias,
            query_count=total,
            percentage=percentage,
        ),
        queries=PaginatedQueriesResponse(
            items=query_items,
            total=total,
            page=page,
            pages=pages,
            limit=limit,
        ),
    )
