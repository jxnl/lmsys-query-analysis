"""Integration tests for services working together."""

import pytest
from lmsys_query_analysis.services import (
    query_service,
    run_service,
    cluster_service,
    export_service,
)


def test_full_workflow_query_to_export(populated_db, temp_dir):
    """Test complete workflow: query → cluster → summarize → export."""
    # 1. Get the run
    run = run_service.get_run(populated_db, "test-run-001")
    assert run is not None
    assert run.algorithm == "kmeans"
    
    # 2. List queries in the run
    queries = query_service.list_queries(
        populated_db,
        run_id="test-run-001",
        limit=100
    )
    assert len(queries) == 5
    
    # 3. Get cluster summaries
    summaries = cluster_service.list_cluster_summaries(
        populated_db,
        run_id="test-run-001"
    )
    assert len(summaries) == 2
    
    # 4. Get queries in a specific cluster
    cluster_queries = query_service.get_cluster_queries(
        populated_db,
        run_id="test-run-001",
        cluster_id=0
    )
    assert len(cluster_queries) == 3
    
    # 5. Export the data
    export_data = export_service.get_export_data(
        populated_db,
        "test-run-001"
    )
    assert len(export_data) == 5
    
    # 6. Export to CSV
    output_path = temp_dir / "integration_export.csv"
    count = export_service.export_to_csv(str(output_path), export_data)
    assert count == 5
    assert output_path.exists()


def test_query_filtering_across_services(populated_db):
    """Test query filtering works consistently across services."""
    # Get all queries in cluster 0
    cluster_queries = query_service.get_cluster_queries(
        populated_db,
        run_id="test-run-001",
        cluster_id=0
    )
    
    # Get summary for cluster 0
    summary = cluster_service.get_cluster_summary(
        populated_db,
        run_id="test-run-001",
        cluster_id=0
    )
    
    # Summary should reflect the correct number of queries
    assert len(cluster_queries) == summary.num_queries


def test_cluster_summary_matches_queries(populated_db):
    """Test that cluster summaries match actual query content."""
    # Get cluster 0 queries and texts
    queries, texts = cluster_service.get_cluster_queries_with_texts(
        populated_db,
        run_id="test-run-001",
        cluster_id=0
    )
    
    # Get the summary
    summary = cluster_service.get_cluster_summary(
        populated_db,
        run_id="test-run-001",
        cluster_id=0
    )
    
    # Verify consistency
    assert len(queries) == len(texts)
    assert len(queries) == summary.num_queries
    assert summary.title == "Python Programming Questions"
    
    # Check that representative queries are from the actual queries
    rep_queries = summary.representative_queries or []
    if rep_queries:
        assert any(rep in texts for rep in rep_queries)


def test_export_includes_all_data(populated_db):
    """Test that export includes query, cluster, and summary data."""
    export_data = export_service.get_export_data(
        populated_db,
        "test-run-001"
    )
    
    assert len(export_data) > 0
    
    # Check structure of export data
    for query, qc, summary in export_data:
        assert query.id is not None
        assert qc.run_id == "test-run-001"
        assert qc.cluster_id in [0, 1]
        
        # Summary should exist for clusters 0 and 1
        if summary is not None:
            assert summary.run_id == "test-run-001"
            assert summary.cluster_id == qc.cluster_id


def test_latest_summary_run_consistency(populated_db):
    """Test that latest summary run ID works correctly."""
    latest = cluster_service.get_latest_summary_run_id(
        populated_db,
        run_id="test-run-001"
    )
    
    assert latest == "summary-001"
    
    # Get summaries with this run ID
    summaries = cluster_service.list_cluster_summaries(
        populated_db,
        run_id="test-run-001",
        summary_run_id=latest
    )
    
    assert len(summaries) == 2
    assert all(s.summary_run_id == latest for s in summaries)


def test_cluster_ids_match_summaries(populated_db):
    """Test that cluster IDs from run match summary cluster IDs."""
    cluster_ids = cluster_service.get_cluster_ids_for_run(
        populated_db,
        run_id="test-run-001"
    )
    
    summaries = cluster_service.list_cluster_summaries(
        populated_db,
        run_id="test-run-001"
    )
    
    summary_cluster_ids = {s.cluster_id for s in summaries}
    
    # All cluster IDs should have summaries
    assert set(cluster_ids) == summary_cluster_ids

