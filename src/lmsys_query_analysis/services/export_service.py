"""Service for export operations."""

import csv
import json
from typing import List, Tuple
from sqlmodel import select
from ..db.connection import Database
from ..db.models import Query, QueryCluster, ClusterSummary


def get_export_data(
    db: Database,
    run_id: str,
) -> List[Tuple[Query, QueryCluster, ClusterSummary]]:
    """Get all data needed for export.
    
    Args:
        db: Database manager instance
        run_id: Clustering run ID to export
    
    Returns:
        List of tuples (Query, QueryCluster, ClusterSummary)
    """
    with db.get_session() as session:
        statement = (
            select(Query, QueryCluster, ClusterSummary)
            .join(QueryCluster, Query.id == QueryCluster.query_id)
            .outerjoin(
                ClusterSummary,
                (ClusterSummary.run_id == QueryCluster.run_id)
                & (ClusterSummary.cluster_id == QueryCluster.cluster_id),
            )
            .where(QueryCluster.run_id == run_id)
        )
        return session.exec(statement).all()


def export_to_csv(
    output_path: str,
    data: List[Tuple[Query, QueryCluster, ClusterSummary]],
) -> int:
    """Export data to CSV file.
    
    Args:
        output_path: Path to output CSV file
        data: List of tuples (Query, QueryCluster, ClusterSummary)
    
    Returns:
        Number of rows exported
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query_id",
                "cluster_id",
                "query_text",
                "model",
                "language",
                "cluster_title",
                "cluster_description",
            ]
        )
        
        for query, qc, summary in data:
            writer.writerow(
                [
                    query.id,
                    qc.cluster_id,
                    query.query_text,
                    query.model,
                    query.language or "",
                    summary.title if summary else "",
                    summary.description if summary else "",
                ]
            )
    
    return len(data)


def export_to_json(
    output_path: str,
    data: List[Tuple[Query, QueryCluster, ClusterSummary]],
) -> int:
    """Export data to JSON file.
    
    Args:
        output_path: Path to output JSON file
        data: List of tuples (Query, QueryCluster, ClusterSummary)
    
    Returns:
        Number of records exported
    """
    json_data = []
    for query, qc, summary in data:
        json_data.append(
            {
                "query_id": query.id,
                "cluster_id": qc.cluster_id,
                "query_text": query.query_text,
                "model": query.model,
                "language": query.language,
                "cluster_title": summary.title if summary else None,
                "cluster_description": summary.description if summary else None,
            }
        )
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return len(json_data)

