"""Summary endpoints for LLM-generated cluster summaries."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import select

from ...db.connection import Database
from ...db.models import ClusterSummary
from ..dependencies import get_db
from ..schemas import ClusterSummaryResponse, SummaryRunListResponse, SummaryRunSummary

router = APIRouter()


@router.get(
    "/",
    response_model=SummaryRunListResponse,
    summary="List all summary runs",
)
async def list_summaries(
    run_id: str | None = Query(None, description="Filter by clustering run ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Database = Depends(get_db),
):
    """List all summary runs with optional filtering by clustering run_id."""
    with db.get_session() as session:
        stmt = select(
            ClusterSummary.summary_run_id,
            ClusterSummary.run_id,
            ClusterSummary.alias,
            ClusterSummary.model,
            ClusterSummary.generated_at,
        ).distinct()

        if run_id:
            stmt = stmt.where(ClusterSummary.run_id == run_id)

        stmt = stmt.order_by(ClusterSummary.generated_at.desc())

        all_summaries = session.exec(stmt).all()

        total = len(all_summaries)
        pages = (total + limit - 1) // limit
        start = (page - 1) * limit
        end = start + limit

        items = [
            SummaryRunSummary(
                summary_run_id=s[0],
                run_id=s[1],
                alias=s[2],
                model=s[3],
                generated_at=s[4],
                status="completed",
            )
            for s in all_summaries[start:end]
        ]

    return SummaryRunListResponse(
        items=items,
        total=total,
        page=page,
        pages=pages,
        limit=limit,
    )


@router.get(
    "/{summary_run_id}/clusters/{cluster_id}",
    response_model=ClusterSummaryResponse,
    responses={404: {"model": dict}},
    summary="Get cluster summary",
)
async def get_cluster_summary(
    summary_run_id: str,
    cluster_id: int,
    run_id: str | None = Query(None, description="Clustering run ID (optional but recommended)"),
    db: Database = Depends(get_db),
):
    """Get the summary for a specific cluster in a summary run."""
    with db.get_session() as session:
        stmt = select(ClusterSummary).where(
            ClusterSummary.summary_run_id == summary_run_id,
            ClusterSummary.cluster_id == cluster_id,
        )

        if run_id:
            stmt = stmt.where(ClusterSummary.run_id == run_id)

        summary = session.exec(stmt).first()

        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"type": "NotFound", "message": "Cluster summary not found"}},
            )

    return ClusterSummaryResponse(
        run_id=summary.run_id,
        cluster_id=summary.cluster_id,
        title=summary.title,
        description=summary.description,
        summary=summary.summary,
        num_queries=summary.num_queries,
        representative_queries=summary.representative_queries,
        summary_run_id=summary.summary_run_id,
        alias=summary.alias,
    )


@router.post(
    "/",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Create summary run (Not Implemented)",
)
async def create_summary():
    """Create a new summary run.

    **Not implemented yet.** This endpoint will be available in Phase 2.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"error": {"type": "NotImplemented", "message": "POST endpoints coming in Phase 2"}},
    )
