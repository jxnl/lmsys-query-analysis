"""Unit tests for cluster_service."""

from lmsys_query_analysis.services import cluster_service


def test_list_cluster_summaries(populated_db):
    """Test listing cluster summaries for a run."""
    summaries = cluster_service.list_cluster_summaries(populated_db, run_id="test-run-001")

    assert len(summaries) == 2
    assert summaries[0].cluster_id == 0
    assert summaries[0].title == "Python Programming Questions"
    assert summaries[1].cluster_id == 1


def test_list_cluster_summaries_with_summary_run_id_filter(populated_db):
    """Test filtering by summary_run_id."""
    summaries = cluster_service.list_cluster_summaries(
        populated_db, run_id="test-run-001", summary_run_id="summary-001"
    )

    assert len(summaries) == 2
    assert all(s.summary_run_id == "summary-001" for s in summaries)


def test_list_cluster_summaries_with_alias_filter(populated_db):
    """Test filtering by alias."""
    summaries = cluster_service.list_cluster_summaries(
        populated_db, run_id="test-run-001", alias="test-v1"
    )

    assert len(summaries) == 2
    assert all(s.alias == "test-v1" for s in summaries)


def test_list_cluster_summaries_with_limit(populated_db):
    """Test limiting number of summaries returned."""
    summaries = cluster_service.list_cluster_summaries(populated_db, run_id="test-run-001", limit=1)

    assert len(summaries) == 1


def test_list_cluster_summaries_empty(temp_db):
    """Test listing summaries when none exist."""
    summaries = cluster_service.list_cluster_summaries(temp_db, run_id="nonexistent-run")

    assert len(summaries) == 0


def test_get_cluster_summary(populated_db):
    """Test getting a specific cluster summary."""
    summary = cluster_service.get_cluster_summary(populated_db, run_id="test-run-001", cluster_id=0)

    assert summary is not None
    assert summary.cluster_id == 0
    assert summary.title == "Python Programming Questions"
    assert summary.num_queries == 3


def test_get_cluster_summary_not_found(populated_db):
    """Test getting a summary that doesn't exist."""
    summary = cluster_service.get_cluster_summary(
        populated_db, run_id="test-run-001", cluster_id=999
    )

    assert summary is None


def test_get_cluster_ids_for_run(populated_db):
    """Test getting all cluster IDs for a run."""
    cluster_ids = cluster_service.get_cluster_ids_for_run(populated_db, run_id="test-run-001")

    assert len(cluster_ids) == 2
    assert 0 in cluster_ids
    assert 1 in cluster_ids


def test_get_cluster_ids_for_run_empty(temp_db):
    """Test getting cluster IDs when none exist."""
    cluster_ids = cluster_service.get_cluster_ids_for_run(temp_db, run_id="nonexistent-run")

    assert len(cluster_ids) == 0


def test_get_cluster_queries_with_texts(populated_db):
    """Test getting queries and texts for a cluster."""
    queries, texts = cluster_service.get_cluster_queries_with_texts(
        populated_db, run_id="test-run-001", cluster_id=0
    )

    assert len(queries) == 3
    assert len(texts) == 3
    assert texts[0] == queries[0].query_text
    assert "Python" in " ".join(texts)


def test_get_latest_summary_run_id(populated_db):
    """Test getting the latest summary run ID."""
    latest = cluster_service.get_latest_summary_run_id(populated_db, run_id="test-run-001")

    assert latest == "summary-001"


def test_get_latest_summary_run_id_none(temp_db):
    """Test getting latest summary run ID when none exist."""
    latest = cluster_service.get_latest_summary_run_id(temp_db, run_id="nonexistent-run")

    assert latest is None
