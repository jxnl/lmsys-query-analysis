"""Service for cluster curation operations."""

from datetime import datetime
from typing import Any

from sqlmodel import and_, delete, select

from ..db.connection import Database
from ..db.models import (
    ClusterEdit,
    ClusterMetadata,
    ClusterSummary,
    OrphanedQuery,
    Query,
    QueryCluster,
)



def get_query_details(db: Database, query_id: int) -> dict[str, Any] | None:
    """Get query details with all cluster assignments.

    Args:
        db: Database manager instance
        query_id: Query ID

    Returns:
        Dictionary with query details and cluster assignments, or None if not found
    """
    with db.get_session() as session:
        query = session.get(Query, query_id)
        if not query:
            return None

        assignments = session.exec(
            select(QueryCluster, ClusterSummary)
            .outerjoin(
                ClusterSummary,
                and_(
                    QueryCluster.run_id == ClusterSummary.run_id,
                    QueryCluster.cluster_id == ClusterSummary.cluster_id,
                ),
            )
            .where(QueryCluster.query_id == query_id)
        ).all()

        clusters = []
        for qc, summary in assignments:
            clusters.append(
                {
                    "run_id": qc.run_id,
                    "cluster_id": qc.cluster_id,
                    "confidence_score": qc.confidence_score,
                    "title": summary.title if summary else None,
                }
            )

        return {
            "query": query,
            "clusters": clusters,
        }


def move_query(
    db: Database,
    run_id: str,
    query_id: int,
    to_cluster_id: int,
    editor: str = "cli-user",
    reason: str | None = None,
) -> dict[str, Any]:
    """Move a query from one cluster to another.

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        query_id: Query ID to move
        to_cluster_id: Target cluster ID
        editor: Who is making the edit
        reason: Why the query is being moved

    Returns:
        Dictionary with operation results
    """
    with db.get_session() as session:
        current = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.query_id == query_id,
                )
            )
        ).first()

        if not current:
            raise ValueError(f"Query {query_id} not found in run {run_id}")

        from_cluster_id = current.cluster_id

        if from_cluster_id == to_cluster_id:
            raise ValueError(f"Query {query_id} is already in cluster {to_cluster_id}")

        current.cluster_id = to_cluster_id

        edit = ClusterEdit(
            run_id=run_id,
            cluster_id=from_cluster_id,
            edit_type="move_query",
            editor=editor,
            old_value={"cluster_id": from_cluster_id, "query_id": query_id},
            new_value={"cluster_id": to_cluster_id, "query_id": query_id},
            reason=reason,
        )
        session.add(edit)

        session.commit()
        session.refresh(current)

        return {
            "success": True,
            "query_id": query_id,
            "from_cluster_id": from_cluster_id,
            "to_cluster_id": to_cluster_id,
            "reason": reason,
        }


def move_queries_batch(
    db: Database,
    run_id: str,
    query_ids: list[int],
    to_cluster_id: int,
    editor: str = "cli-user",
    reason: str | None = None,
) -> dict[str, Any]:
    """Move multiple queries to a cluster.

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        query_ids: List of query IDs to move
        to_cluster_id: Target cluster ID
        editor: Who is making the edit
        reason: Why the queries are being moved

    Returns:
        Dictionary with operation results
    """
    results = []
    errors = []

    for query_id in query_ids:
        try:
            result = move_query(db, run_id, query_id, to_cluster_id, editor, reason)
            results.append(result)
        except Exception as e:
            errors.append({"query_id": query_id, "error": str(e)})

    return {
        "success": len(errors) == 0,
        "moved": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }




def rename_cluster(
    db: Database,
    run_id: str,
    cluster_id: int,
    title: str | None = None,
    description: str | None = None,
    editor: str = "cli-user",
) -> dict[str, Any]:
    """Rename a cluster (update title and/or description).

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        cluster_id: Cluster ID to rename
        title: New title (None to keep current)
        description: New description (None to keep current)
        editor: Who is making the edit

    Returns:
        Dictionary with operation results
    """
    with db.get_session() as session:
        summary = session.exec(
            select(ClusterSummary)
            .where(
                and_(
                    ClusterSummary.run_id == run_id,
                    ClusterSummary.cluster_id == cluster_id,
                )
            )
            .order_by(ClusterSummary.generated_at.desc())
        ).first()

        if not summary:
            raise ValueError(f"No summary found for cluster {cluster_id} in run {run_id}")

        old_values = {"title": summary.title, "description": summary.description}

        if title is not None:
            summary.title = title
        if description is not None:
            summary.description = description

        new_values = {"title": summary.title, "description": summary.description}

        edit = ClusterEdit(
            run_id=run_id,
            cluster_id=cluster_id,
            edit_type="rename",
            editor=editor,
            old_value=old_values,
            new_value=new_values,
            reason=f"Renamed cluster {cluster_id}",
        )
        session.add(edit)

        session.commit()

        return {
            "success": True,
            "cluster_id": cluster_id,
            "old_title": old_values["title"],
            "new_title": new_values["title"],
            "old_description": old_values["description"],
            "new_description": new_values["description"],
        }


def merge_clusters(
    db: Database,
    run_id: str,
    source_cluster_ids: list[int],
    target_cluster_id: int,
    new_title: str | None = None,
    new_description: str | None = None,
    editor: str = "cli-user",
) -> dict[str, Any]:
    """Merge multiple clusters into a target cluster.

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        source_cluster_ids: List of source cluster IDs to merge
        target_cluster_id: Target cluster ID
        new_title: New title for merged cluster
        new_description: New description for merged cluster
        editor: Who is making the edit

    Returns:
        Dictionary with operation results
    """
    with db.get_session() as session:
        target_summary = session.exec(
            select(ClusterSummary).where(
                and_(
                    ClusterSummary.run_id == run_id,
                    ClusterSummary.cluster_id == target_cluster_id,
                )
            )
        ).first()

        if not target_summary:
            raise ValueError(f"Target cluster {target_cluster_id} not found in run {run_id}")

        moved_count = 0
        for source_id in source_cluster_ids:
            if source_id == target_cluster_id:
                continue

            queries = session.exec(
                select(QueryCluster).where(
                    and_(
                        QueryCluster.run_id == run_id,
                        QueryCluster.cluster_id == source_id,
                    )
                )
            ).all()

            for qc in queries:
                qc.cluster_id = target_cluster_id
                moved_count += 1

        if new_title:
            target_summary.title = new_title
        if new_description:
            target_summary.description = new_description

        edit = ClusterEdit(
            run_id=run_id,
            cluster_id=target_cluster_id,
            edit_type="merge",
            editor=editor,
            old_value={"source_clusters": source_cluster_ids},
            new_value={
                "target_cluster": target_cluster_id,
                "queries_moved": moved_count,
            },
            reason=f"Merged {len(source_cluster_ids)} clusters into {target_cluster_id}",
        )
        session.add(edit)

        session.commit()

        return {
            "success": True,
            "target_cluster_id": target_cluster_id,
            "source_cluster_ids": source_cluster_ids,
            "queries_moved": moved_count,
            "new_title": target_summary.title,
        }


def split_cluster(
    db: Database,
    run_id: str,
    cluster_id: int,
    query_ids: list[int],
    new_title: str,
    new_description: str,
    editor: str = "cli-user",
) -> dict[str, Any]:
    """Split queries from a cluster into a new cluster.

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        cluster_id: Original cluster ID
        query_ids: List of query IDs to split into new cluster
        new_title: Title for new cluster
        new_description: Description for new cluster
        editor: Who is making the edit

    Returns:
        Dictionary with operation results
    """
    with db.get_session() as session:
        max_cluster = session.exec(
            select(QueryCluster.cluster_id)
            .where(QueryCluster.run_id == run_id)
            .order_by(QueryCluster.cluster_id.desc())
        ).first()
        new_cluster_id = (max_cluster or 0) + 1

        moved = 0
        for query_id in query_ids:
            qc = session.exec(
                select(QueryCluster).where(
                    and_(
                        QueryCluster.run_id == run_id,
                        QueryCluster.query_id == query_id,
                        QueryCluster.cluster_id == cluster_id,
                    )
                )
            ).first()

            if qc:
                qc.cluster_id = new_cluster_id
                moved += 1

        new_summary = ClusterSummary(
            run_id=run_id,
            cluster_id=new_cluster_id,
            title=new_title,
            description=new_description,
            num_queries=moved,
            model="manual-curation",
        )
        session.add(new_summary)

        edit = ClusterEdit(
            run_id=run_id,
            cluster_id=cluster_id,
            edit_type="split",
            editor=editor,
            old_value={"cluster_id": cluster_id, "query_ids": query_ids},
            new_value={"new_cluster_id": new_cluster_id, "queries_moved": moved},
            reason=f"Split {moved} queries from cluster {cluster_id} into new cluster {new_cluster_id}",
        )
        session.add(edit)

        session.commit()

        return {
            "success": True,
            "original_cluster_id": cluster_id,
            "new_cluster_id": new_cluster_id,
            "queries_moved": moved,
            "new_title": new_title,
        }


def delete_cluster(
    db: Database,
    run_id: str,
    cluster_id: int,
    move_to_cluster_id: int | None = None,
    orphan: bool = False,
    editor: str = "cli-user",
    reason: str | None = None,
) -> dict[str, Any]:
    """Delete a cluster, orphaning or reassigning its queries.

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        cluster_id: Cluster ID to delete
        move_to_cluster_id: If provided, move queries to this cluster
        orphan: If True, orphan the queries instead
        editor: Who is making the edit
        reason: Why the cluster is being deleted

    Returns:
        Dictionary with operation results
    """
    with db.get_session() as session:
        queries = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == cluster_id,
                )
            )
        ).all()

        query_count = len(queries)

        if orphan:
            for qc in queries:
                orphaned = OrphanedQuery(
                    run_id=run_id,
                    query_id=qc.query_id,
                    original_cluster_id=cluster_id,
                    reason=reason or "Cluster deleted",
                )
                session.add(orphaned)
                session.delete(qc)
        elif move_to_cluster_id is not None:
            for qc in queries:
                qc.cluster_id = move_to_cluster_id
        else:
            raise ValueError("Must specify either move_to_cluster_id or orphan=True")

        session.exec(
            delete(ClusterSummary).where(
                and_(
                    ClusterSummary.run_id == run_id,
                    ClusterSummary.cluster_id == cluster_id,
                )
            )
        )

        edit = ClusterEdit(
            run_id=run_id,
            cluster_id=cluster_id,
            edit_type="delete",
            editor=editor,
            old_value={"cluster_id": cluster_id, "query_count": query_count},
            new_value={
                "orphaned": orphan,
                "moved_to": move_to_cluster_id,
            },
            reason=reason,
        )
        session.add(edit)

        session.commit()

        return {
            "success": True,
            "cluster_id": cluster_id,
            "query_count": query_count,
            "orphaned": orphan,
            "moved_to": move_to_cluster_id,
            "reason": reason,
        }




def tag_cluster(
    db: Database,
    run_id: str,
    cluster_id: int,
    coherence_score: int | None = None,
    quality: str | None = None,
    flags: list[str] | None = None,
    notes: str | None = None,
    editor: str = "cli-user",
) -> dict[str, Any]:
    """Tag a cluster with metadata (coherence, quality, flags, notes).

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        cluster_id: Cluster ID
        coherence_score: Coherence score (1-5)
        quality: Quality level ('high', 'medium', 'low')
        flags: List of flags
        notes: Free-form notes
        editor: Who is making the edit

    Returns:
        Dictionary with operation results
    """
    with db.get_session() as session:
        metadata = session.exec(
            select(ClusterMetadata).where(
                and_(
                    ClusterMetadata.run_id == run_id,
                    ClusterMetadata.cluster_id == cluster_id,
                )
            )
        ).first()

        old_values = {}
        if metadata:
            old_values = {
                "coherence_score": metadata.coherence_score,
                "quality": metadata.quality,
                "flags": metadata.flags,
                "notes": metadata.notes,
            }
        else:
            metadata = ClusterMetadata(run_id=run_id, cluster_id=cluster_id)
            session.add(metadata)

        if coherence_score is not None:
            metadata.coherence_score = coherence_score
        if quality is not None:
            metadata.quality = quality
        if flags is not None:
            metadata.flags = flags
        if notes is not None:
            metadata.notes = notes

        metadata.last_edited = datetime.utcnow()

        new_values = {
            "coherence_score": metadata.coherence_score,
            "quality": metadata.quality,
            "flags": metadata.flags,
            "notes": metadata.notes,
        }

        edit = ClusterEdit(
            run_id=run_id,
            cluster_id=cluster_id,
            edit_type="tag",
            editor=editor,
            old_value=old_values,
            new_value=new_values,
            reason="Updated cluster metadata",
        )
        session.add(edit)

        session.commit()

        return {
            "success": True,
            "cluster_id": cluster_id,
            "metadata": new_values,
        }


def get_cluster_metadata(db: Database, run_id: str, cluster_id: int) -> ClusterMetadata | None:
    """Get metadata for a cluster.

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        cluster_id: Cluster ID

    Returns:
        ClusterMetadata object or None
    """
    with db.get_session() as session:
        return session.exec(
            select(ClusterMetadata).where(
                and_(
                    ClusterMetadata.run_id == run_id,
                    ClusterMetadata.cluster_id == cluster_id,
                )
            )
        ).first()




def get_cluster_edit_history(
    db: Database, run_id: str, cluster_id: int | None = None
) -> list[ClusterEdit]:
    """Get edit history for a cluster or entire run.

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        cluster_id: Optional cluster ID to filter by

    Returns:
        List of ClusterEdit objects
    """
    with db.get_session() as session:
        stmt = select(ClusterEdit).where(ClusterEdit.run_id == run_id)

        if cluster_id is not None:
            stmt = stmt.where(ClusterEdit.cluster_id == cluster_id)

        stmt = stmt.order_by(ClusterEdit.timestamp.desc())

        return list(session.exec(stmt).all())


def get_orphaned_queries(db: Database, run_id: str) -> list[tuple[OrphanedQuery, Query]]:
    """Get all orphaned queries for a run.

    Args:
        db: Database manager instance
        run_id: Clustering run ID

    Returns:
        List of tuples (OrphanedQuery, Query)
    """
    with db.get_session() as session:
        results = session.exec(
            select(OrphanedQuery, Query)
            .join(Query, OrphanedQuery.query_id == Query.id)
            .where(OrphanedQuery.run_id == run_id)
            .order_by(OrphanedQuery.orphaned_at.desc())
        ).all()

        return list(results)




def find_problematic_clusters(
    db: Database,
    run_id: str,
    max_size: int | None = None,
    min_size: int | None = None,
    min_languages: int | None = None,
    quality: str | None = None,
) -> list[dict[str, Any]]:
    """Find clusters matching quality criteria.

    Args:
        db: Database manager instance
        run_id: Clustering run ID
        max_size: Maximum number of queries
        min_size: Minimum number of queries
        min_languages: Minimum number of languages
        quality: Quality level filter

    Returns:
        List of cluster info dictionaries
    """
    with db.get_session() as session:
        stmt = (
            select(ClusterSummary, ClusterMetadata)
            .outerjoin(
                ClusterMetadata,
                and_(
                    ClusterSummary.run_id == ClusterMetadata.run_id,
                    ClusterSummary.cluster_id == ClusterMetadata.cluster_id,
                ),
            )
            .where(ClusterSummary.run_id == run_id)
        )

        if quality:
            stmt = stmt.where(ClusterMetadata.quality == quality)

        results = session.exec(stmt).all()

        problematic = []
        for summary, metadata in results:
            if max_size and summary.num_queries and summary.num_queries > max_size:
                continue
            if min_size and summary.num_queries and summary.num_queries < min_size:
                continue

            if min_languages:
                langs = session.exec(
                    select(Query.language)
                    .join(QueryCluster, Query.id == QueryCluster.query_id)
                    .where(
                        and_(
                            QueryCluster.run_id == run_id,
                            QueryCluster.cluster_id == summary.cluster_id,
                        )
                    )
                    .distinct()
                ).all()

                if len(langs) < min_languages:
                    continue

            problematic.append(
                {
                    "cluster_id": summary.cluster_id,
                    "title": summary.title,
                    "num_queries": summary.num_queries,
                    "quality": metadata.quality if metadata else None,
                    "coherence_score": metadata.coherence_score if metadata else None,
                    "flags": metadata.flags if metadata else [],
                }
            )

        return problematic
