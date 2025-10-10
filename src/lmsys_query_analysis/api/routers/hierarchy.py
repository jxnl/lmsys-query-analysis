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

        # Build query count cache for all clusters
        # Get ALL cluster query counts in a single query (much faster!)
        count_stmt = (
            select(QueryCluster.cluster_id, func.count())
            .where(QueryCluster.run_id == run_id)
            .group_by(QueryCluster.cluster_id)
        )
        cluster_counts = session.exec(count_stmt).all()
        query_count_cache = {cluster_id: count for cluster_id, count in cluster_counts}

        # Calculate counts for parent nodes by summing children (bottom-up)
        # Process nodes from LOWEST parent level to HIGHEST (level 0 are leaf nodes, already have counts)
        # This ensures that when we calculate level N, all level N-1 counts are already available
        max_level = max(n.level for n in nodes)
        for level in range(1, max_level + 1):
            level_nodes = [n for n in nodes if n.level == level]
            for node in level_nodes:
                # Sum up children counts
                children_count = 0
                for child_id in (node.children_ids or []):
                    children_count += query_count_cache.get(child_id, 0)
                query_count_cache[node.cluster_id] = children_count

        # Enhance nodes with query counts
        enriched_nodes = []
        for node in nodes:
            query_count = query_count_cache.get(node.cluster_id, 0)

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
