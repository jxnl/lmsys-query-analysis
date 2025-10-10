"""Hierarchy endpoints for cluster hierarchies."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..dependencies import get_db
from ..schemas import HierarchyListResponse, HierarchyTreeResponse, HierarchyNode, HierarchyRunInfo
from ...db.connection import Database
from ...db.models import ClusterHierarchy, QueryCluster
from sqlmodel import select, func

router = APIRouter()


@router.get(
    "/",
    response_model=HierarchyListResponse,
    summary="List all hierarchy runs",
)
async def list_hierarchies(
    run_id: Optional[str] = Query(None, description="Filter by clustering run ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Database = Depends(get_db),
):
    """List all hierarchy runs with optional filtering by clustering run_id."""
    with db.get_session() as session:
        # Get distinct hierarchy runs
        stmt = (
            select(
                ClusterHierarchy.hierarchy_run_id,
                ClusterHierarchy.run_id,
                ClusterHierarchy.created_at,
            )
            .distinct()
        )

        if run_id:
            stmt = stmt.where(ClusterHierarchy.run_id == run_id)

        stmt = stmt.order_by(ClusterHierarchy.created_at.desc())

        all_hierarchies = session.exec(stmt).all()

        # Paginate
        total = len(all_hierarchies)
        pages = (total + limit - 1) // limit
        start = (page - 1) * limit
        end = start + limit

        items = [
            HierarchyRunInfo(
                hierarchy_run_id=h[0],
                run_id=h[1],
                created_at=h[2],
            )
            for h in all_hierarchies[start:end]
        ]

    return HierarchyListResponse(
        items=items,
        total=total,
        page=page,
        pages=pages,
        limit=limit,
    )


@router.get(
    "/{hierarchy_run_id}",
    response_model=HierarchyTreeResponse,
    responses={404: {"model": dict}},
    summary="Get full hierarchy tree",
)
async def get_hierarchy_tree(
    hierarchy_run_id: str,
    include_percentages: bool = Query(True, description="Include query count percentages"),
    db: Database = Depends(get_db),
):
    """Get the full hierarchy tree for a hierarchy run.

    Returns all nodes with parent-child relationships, query counts, and optional percentages.
    """
    with db.get_session() as session:
        # Get all nodes
        stmt = (
            select(ClusterHierarchy)
            .where(ClusterHierarchy.hierarchy_run_id == hierarchy_run_id)
            .order_by(ClusterHierarchy.level)
        )
        nodes = session.exec(stmt).all()

        if not nodes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"type": "NotFound", "message": f"Hierarchy {hierarchy_run_id} not found"}},
            )

        run_id = nodes[0].run_id

        # Get total queries for percentage calculation
        total_queries = None
        if include_percentages:
            count_stmt = select(func.count()).where(QueryCluster.run_id == run_id)
            total_queries = session.exec(count_stmt).one()

        # Enhance nodes with query counts
        enriched_nodes = []
        for node in nodes:
            # For leaf nodes (level 0), count queries directly
            query_count = None
            if node.level == 0:
                count_stmt = (
                    select(func.count())
                    .where(QueryCluster.run_id == run_id)
                    .where(QueryCluster.cluster_id == node.cluster_id)
                )
                query_count = session.exec(count_stmt).one()

            # Calculate percentage
            percentage = None
            if include_percentages and query_count and total_queries and total_queries > 0:
                percentage = round((query_count / total_queries) * 100, 2)

            enriched_nodes.append(
                HierarchyNode(
                    hierarchy_run_id=node.hierarchy_run_id,
                    run_id=node.run_id,
                    cluster_id=node.cluster_id,
                    parent_cluster_id=node.parent_cluster_id,
                    level=node.level,
                    children_ids=node.children_ids or [],
                    title=node.title,
                    description=node.description,
                    query_count=query_count,
                    percentage=percentage,
                )
            )

    return HierarchyTreeResponse(
        nodes=enriched_nodes,
        total_queries=total_queries,
    )


@router.post(
    "/",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Create hierarchy (Not Implemented)",
)
async def create_hierarchy():
    """Create a new hierarchy run.

    **Not implemented yet.** This endpoint will be available in Phase 2.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"error": {"type": "NotImplemented", "message": "POST endpoints coming in Phase 2"}},
    )
