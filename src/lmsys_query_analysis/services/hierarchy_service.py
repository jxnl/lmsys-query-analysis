"""Service for hierarchical cluster operations."""

import time
from typing import List, Optional, Dict, Tuple
from sqlmodel import select
from ..db.connection import Database
from ..db.models import ClusterSummary, ClusterHierarchy, HierarchyRun
from ..clustering.hierarchy import merge_clusters_hierarchical


def list_hierarchy_runs(
    db: Database,
    run_id: Optional[str] = None,
    latest: bool = False,
) -> List[HierarchyRun]:
    """List hierarchy runs.
    
    Args:
        db: Database instance
        run_id: Optional filter by clustering run ID
        latest: If True, return only the most recent run
    
    Returns:
        List of HierarchyRun objects
    """
    with db.get_session() as session:
        statement = select(HierarchyRun).order_by(HierarchyRun.created_at.desc())
        
        if run_id:
            statement = statement.where(HierarchyRun.run_id == run_id)
        
        if latest:
            statement = statement.limit(1)
        
        return session.exec(statement).all()


def get_hierarchy_run(
    db: Database,
    hierarchy_run_id: str,
) -> Optional[HierarchyRun]:
    """Get a specific hierarchy run by ID.
    
    Args:
        db: Database instance
        hierarchy_run_id: Hierarchy run ID to fetch
    
    Returns:
        HierarchyRun object or None if not found
    """
    with db.get_session() as session:
        statement = select(HierarchyRun).where(HierarchyRun.hierarchy_run_id == hierarchy_run_id)
        return session.exec(statement).first()


def get_hierarchy_nodes(
    db: Database,
    hierarchy_run_id: str,
) -> List[ClusterHierarchy]:
    """Get all hierarchy nodes for a hierarchy run.
    
    Args:
        db: Database instance
        hierarchy_run_id: Hierarchy run ID
    
    Returns:
        List of ClusterHierarchy objects
    """
    with db.get_session() as session:
        statement = (
            select(ClusterHierarchy)
            .where(ClusterHierarchy.hierarchy_run_id == hierarchy_run_id)
            .order_by(ClusterHierarchy.level, ClusterHierarchy.cluster_id)
        )
        return session.exec(statement).all()


async def create_hierarchy(
    db: Database,
    run_id: str,
    summary_run_id: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    target_levels: int = 3,
    merge_ratio: float = 0.35,
    neighborhood_size: int = 20,
    concurrency: int = 8,
    rpm: Optional[int] = None,
) -> Tuple[str, List[Dict]]:
    """Create a hierarchical organization of clusters.
    
    Args:
        db: Database instance
        run_id: Clustering run ID to create hierarchy from
        summary_run_id: Optional specific summary run ID (defaults to latest)
        llm_provider: LLM provider for merging
        llm_model: LLM model for merging
        embedding_provider: Embedding provider for neighborhood clustering
        embedding_model: Embedding model for neighborhood clustering
        target_levels: Number of hierarchy levels to create
        merge_ratio: Target merge ratio per level
        neighborhood_size: Average clusters per neighborhood
        concurrency: Max concurrent LLM requests
        rpm: Optional rate limit (requests per minute)
    
    Returns:
        Tuple of (hierarchy_run_id, hierarchy_data)
    """
    with db.get_session() as session:
        # Get base cluster summaries
        if not summary_run_id:
            # Find latest summary run
            statement = (
                select(ClusterSummary.summary_run_id)
                .where(ClusterSummary.run_id == run_id)
                .order_by(ClusterSummary.summary_run_id.desc())
                .limit(1)
            )
            summary_run_id = session.exec(statement).first()
            
            if not summary_run_id:
                raise ValueError(f"No summaries found for run {run_id}")
        
        # Load cluster summaries
        statement = (
            select(ClusterSummary)
            .where(ClusterSummary.run_id == run_id)
            .where(ClusterSummary.summary_run_id == summary_run_id)
            .order_by(ClusterSummary.cluster_id)
        )
        summaries = session.exec(statement).all()
        
        if not summaries:
            raise ValueError(f"No summaries found for run {run_id} with summary_run_id {summary_run_id}")
        
        # Convert to base clusters format
        base_clusters = [
            {
                "cluster_id": s.cluster_id,
                "title": s.title or f"Cluster {s.cluster_id}",
                "description": s.description or "No description"
            }
            for s in summaries
        ]
        
        # Run hierarchical merging
        hierarchy_run_id, hierarchy_data, metadata = await merge_clusters_hierarchical(
            base_clusters=base_clusters,
            run_id=run_id,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            llm_model=llm_model,
            target_levels=target_levels,
            merge_ratio=merge_ratio,
            neighborhood_size=neighborhood_size,
            concurrency=concurrency,
            rpm=rpm,
            summary_run_id=summary_run_id
        )
        
        # Persist hierarchy run metadata
        hierarchy_run = HierarchyRun(**metadata)
        session.add(hierarchy_run)
        
        # Persist hierarchy nodes
        for h in hierarchy_data:
            hierarchy_entry = ClusterHierarchy(**h)
            session.add(hierarchy_entry)
        
        session.commit()
        
        return hierarchy_run_id, hierarchy_data


def get_latest_hierarchy_run_id(
    db: Database,
    run_id: str,
) -> Optional[str]:
    """Get the latest hierarchy run ID for a clustering run.
    
    Args:
        db: Database instance
        run_id: Clustering run ID
    
    Returns:
        Latest hierarchy_run_id or None if no hierarchies exist
    """
    with db.get_session() as session:
        statement = (
            select(HierarchyRun.hierarchy_run_id)
            .where(HierarchyRun.run_id == run_id)
            .order_by(HierarchyRun.created_at.desc())
            .limit(1)
        )
        return session.exec(statement).first()
