"""Service for clustering run operations."""

from typing import List, Optional
from sqlmodel import select
from ..db.connection import Database
from ..db.models import ClusteringRun


def list_runs(
    db: Database,
    latest: bool = False,
) -> List[ClusteringRun]:
    """List all clustering runs.
    
    Args:
        db: Database instance
        latest: If True, return only the most recent run
    
    Returns:
        List of ClusteringRun objects
    """
    with db.get_session() as session:
        statement = select(ClusteringRun).order_by(ClusteringRun.created_at.desc())
        
        if latest:
            statement = statement.limit(1)
        
        return session.exec(statement).all()


def get_run(
    db: Database,
    run_id: str,
) -> Optional[ClusteringRun]:
    """Get a specific clustering run by ID.
    
    Args:
        db: Database instance
        run_id: Run ID to fetch
    
    Returns:
        ClusteringRun object or None if not found
    """
    with db.get_session() as session:
        statement = select(ClusteringRun).where(ClusteringRun.run_id == run_id)
        return session.exec(statement).first()

