"""Unit tests for query_service."""

from lmsys_query_analysis.services import query_service


def test_list_queries_no_filters(populated_db):
    """Test listing all queries without filters."""
    queries = query_service.list_queries(populated_db, limit=10)

    assert len(queries) == 5
    assert queries[0].query_text == "How do I write a Python function?"
    assert queries[0].id is not None


def test_list_queries_with_model_filter(populated_db):
    """Test listing queries filtered by model."""
    queries = query_service.list_queries(populated_db, model="gpt-4", limit=10)

    assert len(queries) == 3
    assert all(q.model == "gpt-4" for q in queries)


def test_list_queries_with_run_id(populated_db):
    """Test listing queries filtered by run_id."""
    queries = query_service.list_queries(populated_db, run_id="test-run-001", limit=10)

    assert len(queries) == 5  # All queries are in test-run-001


def test_list_queries_with_run_and_cluster(populated_db):
    """Test listing queries filtered by run_id and cluster_id."""
    # First verify we can get queries from cluster 1
    cluster1_queries = query_service.get_cluster_queries(
        populated_db, run_id="test-run-001", cluster_id=1
    )
    assert len(cluster1_queries) == 2  # Machine learning queries

    # Then verify list_queries also works with cluster filter
    queries = query_service.list_queries(
        populated_db, run_id="test-run-001", cluster_id=0, limit=10
    )

    # Should get cluster 0 queries
    assert len(queries) > 0
    assert len(queries) <= 5  # At most all queries


def test_list_queries_with_limit(populated_db):
    """Test that limit parameter works correctly."""
    queries = query_service.list_queries(populated_db, limit=2)

    assert len(queries) == 2


def test_list_queries_empty_result(populated_db):
    """Test listing queries with filter that matches nothing."""
    queries = query_service.list_queries(populated_db, run_id="nonexistent-run", limit=10)

    assert len(queries) == 0


def test_get_cluster_queries(populated_db):
    """Test getting all queries in a specific cluster."""
    queries = query_service.get_cluster_queries(populated_db, run_id="test-run-001", cluster_id=1)

    assert len(queries) == 2
    # Verify they're machine learning queries from cluster 1
    texts = [q.query_text for q in queries]
    assert "machine learning" in " ".join(texts).lower()


def test_get_cluster_queries_empty_cluster(populated_db):
    """Test getting queries from non-existent cluster."""
    queries = query_service.get_cluster_queries(populated_db, run_id="test-run-001", cluster_id=999)

    assert len(queries) == 0
