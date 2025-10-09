"""Unit tests for curation_service."""

import pytest
from datetime import datetime
from sqlmodel import select, and_
from lmsys_query_analysis.services import curation_service
from lmsys_query_analysis.db.models import (
    Query,
    QueryCluster,
    ClusteringRun,
    ClusterSummary,
    ClusterEdit,
    ClusterMetadata,
    OrphanedQuery,
)


@pytest.fixture
def extended_populated_db(populated_db, db_session):
    """Database with additional data for curation testing."""
    run_id = "test-run-001"
    
    # Add some cluster metadata
    metadata1 = ClusterMetadata(
        run_id=run_id,
        cluster_id=0,
        coherence_score=4,
        quality="high",
        flags=["reviewed", "clean"],
        notes="Well-organized Python queries",
    )
    metadata2 = ClusterMetadata(
        run_id=run_id,
        cluster_id=1,
        coherence_score=2,
        quality="low",
        flags=["needs-review"],
        notes="Mixed ML concepts, needs splitting",
    )
    db_session.add(metadata1)
    db_session.add(metadata2)
    
    # Add some cluster edits
    edit1 = ClusterEdit(
        run_id=run_id,
        cluster_id=0,
        edit_type="rename",
        editor="test-user",
        old_value={"title": "Old Title"},
        new_value={"title": "Python Programming Questions"},
        reason="Improved clarity",
    )
    db_session.add(edit1)
    
    # Get the first query from the database to use for orphaning
    first_query = db_session.exec(select(Query).limit(1)).first()
    
    # Add an orphaned query using a real query ID
    orphan = OrphanedQuery(
        run_id=run_id,
        query_id=first_query.id,  # Use real query ID
        original_cluster_id=2,  # Use a different cluster to avoid conflicts
        reason="Quality control removal",
    )
    db_session.add(orphan)
    
    db_session.commit()
    return populated_db


# ============================================================================
# Query Operations Tests
# ============================================================================


def test_get_query_details_found(extended_populated_db):
    """Test getting query details when query exists."""
    # Get first query from populated data
    with extended_populated_db.get_session() as session:
        query = session.exec(select(Query).limit(1)).first()
        query_id = query.id
    
    result = curation_service.get_query_details(extended_populated_db, query_id)
    
    assert result is not None
    assert "query" in result
    assert "clusters" in result
    assert result["query"].id == query_id
    assert len(result["clusters"]) >= 1
    assert result["clusters"][0]["run_id"] == "test-run-001"


def test_get_query_details_not_found(extended_populated_db):
    """Test getting query details when query doesn't exist."""
    result = curation_service.get_query_details(extended_populated_db, 99999)
    
    assert result is None


def test_move_query_success(extended_populated_db):
    """Test successfully moving a query between clusters."""
    run_id = "test-run-001"
    
    # Get a query from cluster 0
    with extended_populated_db.get_session() as session:
        qc = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == 0
                )
            ).limit(1)
        ).first()
        query_id = qc.query_id
    
    result = curation_service.move_query(
        extended_populated_db,
        run_id=run_id,
        query_id=query_id,
        to_cluster_id=1,
        editor="test-editor",
        reason="Better fit in ML cluster",
    )
    
    assert result["success"] is True
    assert result["query_id"] == query_id
    assert result["from_cluster_id"] == 0
    assert result["to_cluster_id"] == 1
    assert result["reason"] == "Better fit in ML cluster"
    
    # Verify the query was actually moved
    with extended_populated_db.get_session() as session:
        moved_qc = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.query_id == query_id
                )
            )
        ).first()
        assert moved_qc.cluster_id == 1


def test_move_query_not_found(extended_populated_db):
    """Test moving a query that doesn't exist."""
    with pytest.raises(ValueError, match="Query 99999 not found"):
        curation_service.move_query(
            extended_populated_db,
            run_id="test-run-001",
            query_id=99999,
            to_cluster_id=1,
        )


def test_move_query_same_cluster(extended_populated_db):
    """Test moving a query to the same cluster it's already in."""
    run_id = "test-run-001"
    
    # Get a query from cluster 0
    with extended_populated_db.get_session() as session:
        qc = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == 0
                )
            ).limit(1)
        ).first()
        query_id = qc.query_id
    
    with pytest.raises(ValueError, match="already in cluster 0"):
        curation_service.move_query(
            extended_populated_db,
            run_id=run_id,
            query_id=query_id,
            to_cluster_id=0,
        )


def test_move_queries_batch_success(extended_populated_db):
    """Test moving multiple queries in batch."""
    run_id = "test-run-001"
    
    # Get all queries from cluster 0
    with extended_populated_db.get_session() as session:
        query_clusters = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == 0
                )
            )
        ).all()
        query_ids = [qc.query_id for qc in query_clusters]
    
    result = curation_service.move_queries_batch(
        extended_populated_db,
        run_id=run_id,
        query_ids=query_ids[:2],  # Move first 2 queries
        to_cluster_id=1,
        editor="batch-editor",
        reason="Batch reorganization",
    )
    
    assert result["success"] is True
    assert result["moved"] == 2
    assert result["failed"] == 0
    assert len(result["results"]) == 2
    assert len(result["errors"]) == 0


def test_move_queries_batch_partial_failure(extended_populated_db):
    """Test batch move with some failures."""
    run_id = "test-run-001"
    
    # Mix valid and invalid query IDs
    query_ids = [1, 99999, 2]  # 99999 doesn't exist
    
    result = curation_service.move_queries_batch(
        extended_populated_db,
        run_id=run_id,
        query_ids=query_ids,
        to_cluster_id=1,
    )
    
    assert result["success"] is False
    assert result["moved"] >= 1  # At least some valid ones moved
    assert result["failed"] >= 1  # At least the invalid one failed
    assert len(result["errors"]) >= 1


# ============================================================================
# Cluster Operations Tests
# ============================================================================


def test_rename_cluster_success(extended_populated_db):
    """Test successfully renaming a cluster."""
    run_id = "test-run-001"
    cluster_id = 0
    
    result = curation_service.rename_cluster(
        extended_populated_db,
        run_id=run_id,
        cluster_id=cluster_id,
        title="New Python Title",
        description="Updated description for Python cluster",
        editor="rename-editor",
    )
    
    assert result["success"] is True
    assert result["cluster_id"] == cluster_id
    assert result["new_title"] == "New Python Title"
    assert result["old_title"] == "Python Programming Questions"
    
    # Verify the change was persisted
    with extended_populated_db.get_session() as session:
        summary = session.exec(
            select(ClusterSummary).where(
                and_(
                    ClusterSummary.run_id == run_id,
                    ClusterSummary.cluster_id == cluster_id,
                )
            )
        ).first()
        assert summary.title == "New Python Title"


def test_rename_cluster_not_found(extended_populated_db):
    """Test renaming a cluster that doesn't exist."""
    with pytest.raises(ValueError, match="No summary found"):
        curation_service.rename_cluster(
            extended_populated_db,
            run_id="test-run-001",
            cluster_id=999,
            title="New Title",
        )


def test_merge_clusters_success(extended_populated_db):
    """Test successfully merging clusters."""
    run_id = "test-run-001"
    
    # Count queries in each cluster before merge
    with extended_populated_db.get_session() as session:
        cluster0_queries = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == 0
                )
            )
        ).all()
        cluster1_queries = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == 1
                )
            )
        ).all()
        cluster0_count = len(cluster0_queries)
        cluster1_count = len(cluster1_queries)
    
    result = curation_service.merge_clusters(
        extended_populated_db,
        run_id=run_id,
        source_cluster_ids=[0],
        target_cluster_id=1,
        new_title="Merged Programming Cluster",
        editor="merge-editor",
    )
    
    assert result["success"] is True
    assert result["target_cluster_id"] == 1
    assert result["source_cluster_ids"] == [0]
    assert result["queries_moved"] == cluster0_count
    assert result["new_title"] == "Merged Programming Cluster"
    
    # Verify all queries are now in target cluster
    with extended_populated_db.get_session() as session:
        remaining_in_0 = len(session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == 0
                )
            )
        ).all())
        now_in_1 = len(session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == 1
                )
            )
        ).all())
        
        assert remaining_in_0 == 0
        assert now_in_1 == cluster0_count + cluster1_count


def test_merge_clusters_target_not_found(extended_populated_db):
    """Test merging into a cluster that doesn't exist."""
    with pytest.raises(ValueError, match="Target cluster 999 not found"):
        curation_service.merge_clusters(
            extended_populated_db,
            run_id="test-run-001",
            source_cluster_ids=[0],
            target_cluster_id=999,
        )


def test_split_cluster_success(extended_populated_db):
    """Test successfully splitting a cluster."""
    run_id = "test-run-001"
    cluster_id = 0
    
    # Get some queries from cluster 0 to split
    with extended_populated_db.get_session() as session:
        cluster_queries = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == cluster_id
                )
            ).limit(2)
        ).all()
        queries_to_split = [qc.query_id for qc in cluster_queries]
    
    result = curation_service.split_cluster(
        extended_populated_db,
        run_id=run_id,
        cluster_id=cluster_id,
        query_ids=queries_to_split,
        new_title="Split Python Cluster",
        new_description="Specialized Python queries",
        editor="split-editor",
    )
    
    assert result["success"] is True
    assert result["original_cluster_id"] == cluster_id
    assert result["queries_moved"] == len(queries_to_split)
    assert result["new_title"] == "Split Python Cluster"
    
    new_cluster_id = result["new_cluster_id"]
    
    # Verify queries moved to new cluster
    with extended_populated_db.get_session() as session:
        new_cluster_queries = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == new_cluster_id
                )
            )
        ).all()
        assert len(new_cluster_queries) == len(queries_to_split)
        
        # Verify summary was created
        new_summary = session.exec(
            select(ClusterSummary).where(
                and_(
                    ClusterSummary.run_id == run_id,
                    ClusterSummary.cluster_id == new_cluster_id,
                )
            )
        ).first()
        assert new_summary is not None
        assert new_summary.title == "Split Python Cluster"


def test_delete_cluster_with_orphaning(extended_populated_db):
    """Test deleting a cluster and orphaning its queries."""
    run_id = "test-run-001"
    cluster_id = 1  # Use cluster 1 instead of 0 to avoid the fixture orphan conflict
    
    # Count queries before deletion
    with extended_populated_db.get_session() as session:
        cluster_queries = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == cluster_id
                )
            )
        ).all()
        query_count = len(cluster_queries)
    
    result = curation_service.delete_cluster(
        extended_populated_db,
        run_id=run_id,
        cluster_id=cluster_id,
        orphan=True,
        editor="delete-editor",
        reason="Quality control",
    )
    
    assert result["success"] is True
    assert result["cluster_id"] == cluster_id
    assert result["query_count"] == query_count
    assert result["orphaned"] is True
    assert result["reason"] == "Quality control"
    
    # Verify queries were orphaned
    with extended_populated_db.get_session() as session:
        orphaned_queries = session.exec(
            select(OrphanedQuery).where(
                and_(
                    OrphanedQuery.run_id == run_id,
                    OrphanedQuery.original_cluster_id == cluster_id,
                )
            )
        ).all()
        assert len(orphaned_queries) == query_count


def test_delete_cluster_with_reassignment(extended_populated_db):
    """Test deleting a cluster and moving queries to another cluster."""
    run_id = "test-run-001"
    source_cluster_id = 0
    target_cluster_id = 1
    
    # Count queries before deletion
    with extended_populated_db.get_session() as session:
        source_queries = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == source_cluster_id
                )
            )
        ).all()
        target_queries_before = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == target_cluster_id
                )
            )
        ).all()
        source_count = len(source_queries)
        target_count_before = len(target_queries_before)
    
    result = curation_service.delete_cluster(
        extended_populated_db,
        run_id=run_id,
        cluster_id=source_cluster_id,
        move_to_cluster_id=target_cluster_id,
        editor="delete-editor",
        reason="Consolidation",
    )
    
    assert result["success"] is True
    assert result["moved_to"] == target_cluster_id
    assert result["orphaned"] is False
    
    # Verify queries were moved
    with extended_populated_db.get_session() as session:
        target_queries_after = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == target_cluster_id
                )
            )
        ).all()
        remaining_in_source = session.exec(
            select(QueryCluster).where(
                and_(
                    QueryCluster.run_id == run_id,
                    QueryCluster.cluster_id == source_cluster_id
                )
            )
        ).all()
        
        assert len(remaining_in_source) == 0
        assert len(target_queries_after) == target_count_before + source_count


def test_delete_cluster_invalid_options(extended_populated_db):
    """Test deleting cluster with invalid options."""
    with pytest.raises(ValueError, match="Must specify either move_to_cluster_id or orphan=True"):
        curation_service.delete_cluster(
            extended_populated_db,
            run_id="test-run-001",
            cluster_id=0,
            # Neither orphan=True nor move_to_cluster_id specified
        )


# ============================================================================
# Metadata Operations Tests
# ============================================================================


def test_tag_cluster_new_metadata(extended_populated_db):
    """Test tagging a cluster that has no existing metadata."""
    run_id = "test-run-001"
    cluster_id = 2  # Cluster without existing metadata
    
    result = curation_service.tag_cluster(
        extended_populated_db,
        run_id=run_id,
        cluster_id=cluster_id,
        coherence_score=5,
        quality="high",
        flags=["excellent", "ready"],
        notes="Perfect cluster organization",
        editor="tag-editor",
    )
    
    assert result["success"] is True
    assert result["cluster_id"] == cluster_id
    assert result["metadata"]["coherence_score"] == 5
    assert result["metadata"]["quality"] == "high"
    assert result["metadata"]["flags"] == ["excellent", "ready"]
    assert result["metadata"]["notes"] == "Perfect cluster organization"


def test_tag_cluster_update_existing(extended_populated_db):
    """Test updating metadata for a cluster that already has it."""
    run_id = "test-run-001"
    cluster_id = 0  # This cluster has existing metadata
    
    result = curation_service.tag_cluster(
        extended_populated_db,
        run_id=run_id,
        cluster_id=cluster_id,
        coherence_score=5,  # Update from 4 to 5
        notes="Updated notes",
        editor="update-editor",
    )
    
    assert result["success"] is True
    assert result["metadata"]["coherence_score"] == 5
    assert result["metadata"]["quality"] == "high"  # Should remain unchanged
    assert result["metadata"]["notes"] == "Updated notes"


def test_get_cluster_metadata_found(extended_populated_db):
    """Test getting metadata for a cluster that has it."""
    run_id = "test-run-001"
    cluster_id = 0
    
    metadata = curation_service.get_cluster_metadata(
        extended_populated_db, run_id, cluster_id
    )
    
    assert metadata is not None
    assert metadata.coherence_score == 4
    assert metadata.quality == "high"
    assert metadata.flags == ["reviewed", "clean"]
    assert metadata.notes == "Well-organized Python queries"


def test_get_cluster_metadata_not_found(extended_populated_db):
    """Test getting metadata for a cluster that doesn't have it."""
    run_id = "test-run-001"
    cluster_id = 999
    
    metadata = curation_service.get_cluster_metadata(
        extended_populated_db, run_id, cluster_id
    )
    
    assert metadata is None


# ============================================================================
# Audit Operations Tests
# ============================================================================


def test_get_cluster_edit_history_all(extended_populated_db):
    """Test getting all edit history for a run."""
    run_id = "test-run-001"
    
    history = curation_service.get_cluster_edit_history(extended_populated_db, run_id)
    
    assert len(history) >= 1  # At least the fixture edit
    assert all(edit.run_id == run_id for edit in history)
    
    # Check that they're ordered by timestamp descending (most recent first)
    timestamps = [edit.timestamp for edit in history]
    assert timestamps == sorted(timestamps, reverse=True)


def test_get_cluster_edit_history_for_cluster(extended_populated_db):
    """Test getting edit history for a specific cluster."""
    run_id = "test-run-001"
    cluster_id = 0
    
    history = curation_service.get_cluster_edit_history(
        extended_populated_db, run_id, cluster_id
    )
    
    assert len(history) >= 1
    assert all(edit.cluster_id == cluster_id for edit in history)
    assert history[0].edit_type == "rename"  # From fixture


def test_get_orphaned_queries(extended_populated_db):
    """Test getting orphaned queries for a run."""
    run_id = "test-run-001"
    
    orphaned = curation_service.get_orphaned_queries(extended_populated_db, run_id)
    
    assert len(orphaned) >= 1  # At least the fixture orphan
    
    orphan_record, query = orphaned[0]
    assert orphan_record.run_id == run_id
    assert orphan_record.original_cluster_id == 2  # Updated to match fixture
    assert orphan_record.reason == "Quality control removal"


def test_get_orphaned_queries_empty(temp_db):
    """Test getting orphaned queries when none exist."""
    orphaned = curation_service.get_orphaned_queries(temp_db, "nonexistent-run")
    
    assert len(orphaned) == 0


# ============================================================================
# Batch Operations Tests
# ============================================================================


def test_find_problematic_clusters_by_quality(extended_populated_db):
    """Test finding clusters by quality level."""
    run_id = "test-run-001"
    
    low_quality = curation_service.find_problematic_clusters(
        extended_populated_db, run_id, quality="low"
    )
    
    assert len(low_quality) >= 1
    assert all(cluster["quality"] == "low" for cluster in low_quality)
    
    high_quality = curation_service.find_problematic_clusters(
        extended_populated_db, run_id, quality="high"
    )
    
    assert len(high_quality) >= 1
    assert all(cluster["quality"] == "high" for cluster in high_quality)


def test_find_problematic_clusters_by_size(extended_populated_db):
    """Test finding clusters by size constraints."""
    run_id = "test-run-001"
    
    # Find large clusters (more than 2 queries)
    large_clusters = curation_service.find_problematic_clusters(
        extended_populated_db, run_id, min_size=3
    )
    
    assert all(cluster["num_queries"] >= 3 for cluster in large_clusters)
    
    # Find small clusters (less than 3 queries)
    small_clusters = curation_service.find_problematic_clusters(
        extended_populated_db, run_id, max_size=2
    )
    
    assert all(cluster["num_queries"] <= 2 for cluster in small_clusters)


def test_find_problematic_clusters_empty_results(extended_populated_db):
    """Test finding clusters with criteria that match nothing."""
    run_id = "test-run-001"
    
    # Look for impossibly large clusters
    results = curation_service.find_problematic_clusters(
        extended_populated_db, run_id, min_size=1000
    )
    
    assert len(results) == 0


def test_find_problematic_clusters_no_run(temp_db):
    """Test finding clusters in a non-existent run."""
    results = curation_service.find_problematic_clusters(
        temp_db, "nonexistent-run"
    )
    
    assert len(results) == 0
