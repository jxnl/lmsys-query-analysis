"""Service for cluster operations."""

from typing import List, Optional, Tuple
from sqlmodel import select
from ..db.connection import Database
from ..db.models import ClusterSummary, QueryCluster, Query


def list_cluster_summaries(
    db: Database,
    run_id: str,
    summary_run_id: Optional[str] = None,
    alias: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[ClusterSummary]:
    """List cluster summaries for a run.
    
    Args:
        db: Database manager instance
        run_id: Clustering run ID
        summary_run_id: Optional specific summary run ID filter
        alias: Optional summary alias filter
        limit: Optional limit on number of summaries
    
    Returns:
        List of ClusterSummary objects
    """
    with db.get_session() as session:
        statement = (
            select(ClusterSummary)
            .where(ClusterSummary.run_id == run_id)
            .order_by(
                ClusterSummary.num_queries.desc(), ClusterSummary.cluster_id.asc()
            )
        )
        
        if summary_run_id:
            statement = statement.where(ClusterSummary.summary_run_id == summary_run_id)
        elif alias:
            statement = statement.where(ClusterSummary.alias == alias)
        
        if limit:
            statement = statement.limit(limit)
        
        return session.exec(statement).all()


def get_cluster_summary(
    db: Database,
    run_id: str,
    cluster_id: int,
) -> Optional[ClusterSummary]:
    """Get cluster summary for a specific cluster.
    
    Args:
        db: Database manager instance
        run_id: Clustering run ID
        cluster_id: Cluster ID
    
    Returns:
        ClusterSummary object or None if not found
    """
    with db.get_session() as session:
        statement = select(ClusterSummary).where(
            ClusterSummary.run_id == run_id,
            ClusterSummary.cluster_id == cluster_id,
        )
        return session.exec(statement).first()


def get_cluster_ids_for_run(
    db: Database,
    run_id: str,
) -> List[int]:
    """Get all cluster IDs for a run.
    
    Args:
        db: Database manager instance
        run_id: Clustering run ID
    
    Returns:
        List of cluster IDs
    """
    with db.get_session() as session:
        statement = (
            select(QueryCluster.cluster_id)
            .where(QueryCluster.run_id == run_id)
            .distinct()
        )
        return list(session.exec(statement))


def get_cluster_queries_with_texts(
    db: Database,
    run_id: str,
    cluster_id: int,
) -> Tuple[List[Query], List[str]]:
    """Get queries and their texts for a cluster.
    
    Args:
        db: Database manager instance
        run_id: Clustering run ID
        cluster_id: Cluster ID
    
    Returns:
        Tuple of (queries list, query_texts list)
    """
    with db.get_session() as session:
        statement = (
            select(Query)
            .join(QueryCluster, Query.id == QueryCluster.query_id)
            .where(QueryCluster.run_id == run_id)
            .where(QueryCluster.cluster_id == cluster_id)
        )
        queries = session.exec(statement).all()
        query_texts = [q.query_text for q in queries]
        return queries, query_texts


def get_latest_summary_run_id(
    db: Database,
    run_id: str,
) -> Optional[str]:
    """Get the latest summary run ID for a clustering run.
    
    Args:
        db: Database manager instance
        run_id: Clustering run ID
    
    Returns:
        Latest summary_run_id or None if no summaries exist
    """
    with db.get_session() as session:
        statement = (
            select(ClusterSummary.summary_run_id)
            .where(ClusterSummary.run_id == run_id)
            .order_by(ClusterSummary.summary_run_id.desc())
            .limit(1)
        )
        return session.exec(statement).first()

