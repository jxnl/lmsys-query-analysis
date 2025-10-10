"""Curation endpoints for metadata, edit history, and orphaned queries."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..dependencies import get_db
from ..schemas import (
    ClusterMetadata,
    EditHistoryResponse,
    EditHistoryRecord,
    OrphanedQueriesResponse,
    OrphanInfo,
    QueryResponse,
)
from ...db.connection import Database
from ...db.models import ClusterMetadata as ClusterMetadataModel, ClusterEdit, OrphanedQuery, Query as QueryModel
from sqlmodel import select

router = APIRouter()


@router.get(
    "/clusters/{cluster_id}/metadata",
    response_model=ClusterMetadata,
    responses={404: {"model": dict}},
    summary="Get cluster metadata",
)
async def get_cluster_metadata(
    cluster_id: int,
    run_id: str = Query(..., description="Clustering run ID"),
    db: Database = Depends(get_db),
):
    """Get quality metadata for a cluster (coherence score, quality, flags, notes)."""
    with db.get_session() as session:
        stmt = select(ClusterMetadataModel).where(
            ClusterMetadataModel.run_id == run_id,
            ClusterMetadataModel.cluster_id == cluster_id,
        )
        metadata = session.exec(stmt).first()

        if not metadata:
            # Return empty metadata if not found
            return ClusterMetadata(
                coherence_score=None,
                quality=None,
                notes=None,
                flags=None,
                last_edited=None,
            )

    return ClusterMetadata(
        coherence_score=metadata.coherence_score,
        quality=metadata.quality,
        notes=metadata.notes,
        flags=metadata.flags,
        last_edited=metadata.last_edited,
    )


@router.get(
    "/clusters/{cluster_id}/history",
    response_model=EditHistoryResponse,
    summary="Get cluster edit history",
)
async def get_cluster_history(
    cluster_id: int,
    run_id: str = Query(..., description="Clustering run ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Database = Depends(get_db),
):
    """Get the edit history for a specific cluster."""
    with db.get_session() as session:
        stmt = (
            select(ClusterEdit)
            .where(ClusterEdit.run_id == run_id, ClusterEdit.cluster_id == cluster_id)
            .order_by(ClusterEdit.timestamp.desc())
        )

        all_edits = session.exec(stmt).all()

        # Paginate
        total = len(all_edits)
        pages = (total + limit - 1) // limit
        start = (page - 1) * limit
        end = start + limit

        items = [
            EditHistoryRecord(
                timestamp=edit.timestamp,
                cluster_id=edit.cluster_id,
                edit_type=edit.edit_type,
                editor=edit.editor,
                reason=edit.reason,
                old_value=edit.old_value,
                new_value=edit.new_value,
            )
            for edit in all_edits[start:end]
        ]

    return EditHistoryResponse(
        items=items,
        total=total,
        page=page,
        pages=pages,
        limit=limit,
    )


@router.get(
    "/runs/{run_id}/audit",
    response_model=EditHistoryResponse,
    summary="Get full audit log for a run",
)
async def get_run_audit(
    run_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Database = Depends(get_db),
):
    """Get the full edit history (audit log) for a clustering run."""
    with db.get_session() as session:
        stmt = select(ClusterEdit).where(ClusterEdit.run_id == run_id).order_by(ClusterEdit.timestamp.desc())

        all_edits = session.exec(stmt).all()

        # Paginate
        total = len(all_edits)
        pages = (total + limit - 1) // limit
        start = (page - 1) * limit
        end = start + limit

        items = [
            EditHistoryRecord(
                timestamp=edit.timestamp,
                cluster_id=edit.cluster_id,
                edit_type=edit.edit_type,
                editor=edit.editor,
                reason=edit.reason,
                old_value=edit.old_value,
                new_value=edit.new_value,
            )
            for edit in all_edits[start:end]
        ]

    return EditHistoryResponse(
        items=items,
        total=total,
        page=page,
        pages=pages,
        limit=limit,
    )


@router.get(
    "/runs/{run_id}/orphaned",
    response_model=OrphanedQueriesResponse,
    summary="Get orphaned queries for a run",
)
async def get_orphaned_queries(
    run_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Database = Depends(get_db),
):
    """Get all orphaned queries for a clustering run."""
    with db.get_session() as session:
        stmt = (
            select(OrphanedQuery, QueryModel)
            .join(QueryModel, OrphanedQuery.query_id == QueryModel.id)
            .where(OrphanedQuery.run_id == run_id)
            .order_by(OrphanedQuery.orphaned_at.desc())
        )

        all_orphans = session.exec(stmt).all()

        # Paginate
        total = len(all_orphans)
        pages = (total + limit - 1) // limit
        start = (page - 1) * limit
        end = start + limit

        items = [
            OrphanInfo(
                orphan={
                    "query_id": orphan.query_id,
                    "original_cluster_id": orphan.original_cluster_id,
                    "orphaned_at": orphan.orphaned_at.isoformat(),
                    "reason": orphan.reason,
                },
                query=QueryResponse.model_validate(query),
            )
            for orphan, query in all_orphans[start:end]
        ]

    return OrphanedQueriesResponse(
        items=items,
        total=total,
        page=page,
        pages=pages,
        limit=limit,
    )


# ===== POST Endpoint Stubs (501 Not Implemented) =====


@router.post(
    "/queries/{query_id}/move",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Move query to different cluster (Not Implemented)",
)
async def move_query():
    """Move a query to a different cluster.

    **Not implemented yet.** This endpoint will be available in Phase 2.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"error": {"type": "NotImplemented", "message": "POST endpoints coming in Phase 2"}},
    )


@router.post(
    "/clusters/{cluster_id}/rename",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Rename cluster (Not Implemented)",
)
async def rename_cluster():
    """Rename a cluster.

    **Not implemented yet.** This endpoint will be available in Phase 2.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"error": {"type": "NotImplemented", "message": "POST endpoints coming in Phase 2"}},
    )
