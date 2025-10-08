"""Service for query operations."""

from typing import List, Optional
from sqlmodel import select
from ..db.connection import Database
from ..db.models import Query, QueryCluster


def list_queries(
    db: Database,
    run_id: Optional[str] = None,
    cluster_id: Optional[int] = None,
    model: Optional[str] = None,
    limit: int = 50,
) -> List[Query]:
    """List queries with optional filtering.
    
    Args:
        db: Database instance
        run_id: Optional run ID filter
        cluster_id: Optional cluster ID filter
        model: Optional model name filter
        limit: Maximum number of queries to return
    
    Returns:
        List of Query objects
    """
    with db.get_session() as session:
        # Build query with filters
        if run_id and cluster_id:
            # Filter by run and cluster
            statement = (
                select(Query)
                .join(QueryCluster, Query.id == QueryCluster.query_id)
                .where(QueryCluster.run_id == run_id)
                .where(QueryCluster.cluster_id == cluster_id)
                .limit(limit)
            )
        elif run_id:
            # Filter by run only
            statement = (
                select(Query)
                .join(QueryCluster, Query.id == QueryCluster.query_id)
                .where(QueryCluster.run_id == run_id)
                .limit(limit)
            )
        else:
            # No run filter, just list queries
            statement = select(Query).limit(limit)
            if model:
                statement = statement.where(Query.model == model)
        
        return session.exec(statement).all()


def get_cluster_queries(
    db: Database,
    run_id: str,
    cluster_id: int,
) -> List[Query]:
    """Get all queries in a specific cluster.
    
    Args:
        db: Database instance
        run_id: Clustering run ID
        cluster_id: Cluster ID
    
    Returns:
        List of Query objects in the cluster
    """
    with db.get_session() as session:
        statement = (
            select(Query)
            .join(QueryCluster, Query.id == QueryCluster.query_id)
            .where(QueryCluster.run_id == run_id)
            .where(QueryCluster.cluster_id == cluster_id)
        )
        return session.exec(statement).all()

