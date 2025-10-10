"""Clustering endpoints for managing and querying clustering runs."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..dependencies import get_db
from ..schemas import (
    ClusteringRunListResponse,
    ClusteringRunDetail,
    ClusteringRunStatusResponse,
    ClusterListResponse,
    ClusterSummaryResponse,
    ErrorResponse,
)
from ...db.connection import Database
from ...services import run_service, cluster_service
from ...db.models import QueryCluster

router = APIRouter()


@router.get(
    "/runs",
    response_model=ClusteringRunListResponse,
    summary="List all clustering runs",
)
async def list_runs(
    algorithm: Optional[str] = Query(None, description="Filter by algorithm (kmeans, hdbscan)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Database = Depends(get_db),
):
    """List all clustering runs with optional filtering.

    Returns paginated list of clustering run summaries with basic metadata.
    """
    # Get runs from service
    runs_list = run_service.list_runs(db)

    # Filter by algorithm if provided
    if algorithm:
        runs_list = [r for r in runs_list if r.algorithm == algorithm]

    # Calculate pagination
    total = len(runs_list)
    pages = (total + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit

    items = runs_list[start:end]

    return ClusteringRunListResponse(
        items=items,
        total=total,
        page=page,
        pages=pages,
        limit=limit,
    )


@router.get(
    "/runs/{run_id}",
    response_model=ClusteringRunDetail,
    responses={404: {"model": ErrorResponse}},
    summary="Get clustering run details",
)
async def get_run(
    run_id: str,
    db: Database = Depends(get_db),
):
    """Get detailed information about a specific clustering run.

    Returns run metadata including parameters, cluster count, and metrics.
    """
    run = run_service.get_run(db, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"type": "NotFound", "message": f"Run {run_id} not found"}},
        )

    return ClusteringRunDetail(
        run_id=run.run_id,
        algorithm=run.algorithm,
        num_clusters=run.num_clusters,
        description=run.description,
        parameters=run.parameters,
        created_at=run.created_at,
        status="completed",  # Assuming completed for now
        metrics=None,  # Could add metrics if available
        latest_errors=None,
    )


@router.get(
    "/runs/{run_id}/status",
    response_model=ClusteringRunStatusResponse,
    summary="Get clustering run status",
)
async def get_run_status(
    run_id: str,
    db: Database = Depends(get_db),
):
    """Get the current status of a clustering run (for polling).

    Used by clients to poll long-running clustering jobs.
    """
    run = run_service.get_run(db, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"type": "NotFound", "message": f"Run {run_id} not found"}},
        )

    # Count processed queries
    with db.get_session() as session:
        from sqlmodel import select, func

        count_stmt = select(func.count()).where(QueryCluster.run_id == run_id)
        processed = session.exec(count_stmt).one()

    return ClusteringRunStatusResponse(
        run_id=run.run_id,
        status="completed",  # Assuming completed for existing runs
        processed=processed,
    )


@router.get(
    "/runs/{run_id}/clusters",
    response_model=ClusterListResponse,
    summary="List clusters for a run with aggregations",
)
async def list_clusters(
    run_id: str,
    include_counts: bool = Query(True, description="Include query counts per cluster"),
    include_percentages: bool = Query(True, description="Include percentages of total queries"),
    summary_run_id: Optional[str] = Query(None, description="Filter by specific summary run ID"),
    alias: Optional[str] = Query(None, description="Filter by summary alias"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Limit number of clusters"),
    page: int = Query(1, ge=1, description="Page number"),
    page_limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Database = Depends(get_db),
):
    """List all clusters for a run with optional aggregations.

    Supports:
    - Query counts per cluster (include_counts=true)
    - Percentages of total queries (include_percentages=true)
    - Filtering by summary_run_id or alias
    - Pagination
    """
    # Get cluster summaries
    summaries = cluster_service.list_cluster_summaries(
        db, run_id, summary_run_id, alias, limit
    )

    # Get total query count for the run if we need percentages
    total_queries = None
    if include_counts or include_percentages:
        with db.get_session() as session:
            from sqlmodel import select, func

            count_stmt = select(func.count()).where(QueryCluster.run_id == run_id)
            total_queries = session.exec(count_stmt).one()

    # Enhance summaries with counts and percentages
    enriched_summaries = []
    for summary in summaries:
        # Get query count for this cluster if not already set
        query_count = summary.num_queries
        if query_count is None and include_counts:
            with db.get_session() as session:
                from sqlmodel import select, func

                count_stmt = (
                    select(func.count())
                    .where(QueryCluster.run_id == run_id)
                    .where(QueryCluster.cluster_id == summary.cluster_id)
                )
                query_count = session.exec(count_stmt).one()

        # Calculate percentage
        percentage = None
        if include_percentages and query_count and total_queries and total_queries > 0:
            percentage = round((query_count / total_queries) * 100, 2)

        enriched_summaries.append(
            ClusterSummaryResponse(
                run_id=summary.run_id,
                cluster_id=summary.cluster_id,
                title=summary.title,
                description=summary.description,
                summary=summary.summary,
                num_queries=summary.num_queries,
                representative_queries=summary.representative_queries,
                summary_run_id=summary.summary_run_id,
                alias=summary.alias,
                query_count=query_count,
                percentage=percentage,
            )
        )

    # Paginate
    total = len(enriched_summaries)
    pages = (total + page_limit - 1) // page_limit
    start = (page - 1) * page_limit
    end = start + page_limit
    items = enriched_summaries[start:end]

    return ClusterListResponse(
        items=items,
        total=total,
        page=page,
        pages=pages,
        limit=page_limit,
        total_queries=total_queries,
    )


# ===== POST Endpoint Stubs (501 Not Implemented) =====


@router.post(
    "/kmeans",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Create KMeans clustering run (Not Implemented)",
)
async def create_kmeans_run():
    """Create a new KMeans clustering run.

    **Not implemented yet.** This endpoint will be available in Phase 2.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"error": {"type": "NotImplemented", "message": "POST endpoints coming in Phase 2"}},
    )


@router.post(
    "/hdbscan",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Create HDBSCAN clustering run (Not Implemented)",
)
async def create_hdbscan_run():
    """Create a new HDBSCAN clustering run.

    **Not implemented yet.** This endpoint will be available in Phase 2.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"error": {"type": "NotImplemented", "message": "POST endpoints coming in Phase 2"}},
    )
