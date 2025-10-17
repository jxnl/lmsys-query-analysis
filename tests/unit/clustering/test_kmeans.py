"""Unit tests for KMeans clustering functions."""

import pytest

from lmsys_query_analysis.clustering.kmeans import get_cluster_info
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun, Query, QueryCluster


@pytest.fixture
def db_with_clusters():
    """Create a test database with queries and clusters."""
    db = Database(":memory:")
    db.create_tables()

    with db.get_session() as session:
        # Create test queries
        queries = []
        for i in range(10):
            query = Query(
                conversation_id=f"conv-{i}",
                model="gpt-4",
                query_text=f"Test query {i}",
            )
            session.add(query)
            queries.append(query)
        session.commit()

        # Refresh to get IDs
        for q in queries:
            session.refresh(q)

        # Create clustering run
        run = ClusteringRun(
            run_id="test-run",
            algorithm="kmeans",
            num_clusters=3,
            parameters={"n_clusters": 3},
        )
        session.add(run)
        session.commit()

        # Assign queries to clusters
        # Cluster 0: queries 0, 1, 2
        # Cluster 1: queries 3, 4, 5, 6
        # Cluster 2: queries 7, 8, 9
        for i, query in enumerate(queries):
            if i < 3:
                cluster_id = 0
            elif i < 7:
                cluster_id = 1
            else:
                cluster_id = 2

            assignment = QueryCluster(
                run_id="test-run",
                query_id=query.id,
                cluster_id=cluster_id,
            )
            session.add(assignment)
        session.commit()

    return db


def test_get_cluster_info_basic(db_with_clusters):
    """Test getting basic cluster information."""
    result = get_cluster_info(db_with_clusters, "test-run", 0)

    assert result["run_id"] == "test-run"
    assert result["cluster_id"] == 0
    assert result["size"] == 3
    assert len(result["queries"]) == 3


def test_get_cluster_info_queries_content(db_with_clusters):
    """Test that cluster info includes correct query content."""
    result = get_cluster_info(db_with_clusters, "test-run", 0)

    queries = result["queries"]
    assert len(queries) == 3

    # Check structure
    for q in queries:
        assert "id" in q
        assert "text" in q
        assert "model" in q
        assert q["model"] == "gpt-4"


def test_get_cluster_info_different_cluster(db_with_clusters):
    """Test getting info for different cluster sizes."""
    # Cluster 0 has 3 queries
    result0 = get_cluster_info(db_with_clusters, "test-run", 0)
    assert result0["size"] == 3

    # Cluster 1 has 4 queries
    result1 = get_cluster_info(db_with_clusters, "test-run", 1)
    assert result1["size"] == 4

    # Cluster 2 has 3 queries
    result2 = get_cluster_info(db_with_clusters, "test-run", 2)
    assert result2["size"] == 3


def test_get_cluster_info_empty_cluster(db_with_clusters):
    """Test getting info for a cluster with no queries."""
    # Cluster 99 doesn't exist
    result = get_cluster_info(db_with_clusters, "test-run", 99)

    assert result["run_id"] == "test-run"
    assert result["cluster_id"] == 99
    assert result["size"] == 0
    assert result["queries"] == []


def test_get_cluster_info_nonexistent_run(db_with_clusters):
    """Test getting info for a non-existent run."""
    result = get_cluster_info(db_with_clusters, "nonexistent-run", 0)

    assert result["run_id"] == "nonexistent-run"
    assert result["cluster_id"] == 0
    assert result["size"] == 0
    assert result["queries"] == []


def test_get_cluster_info_query_text_preserved(db_with_clusters):
    """Test that query text is correctly preserved."""
    result = get_cluster_info(db_with_clusters, "test-run", 0)

    texts = [q["text"] for q in result["queries"]]

    # Should have queries 0, 1, 2
    assert "Test query 0" in texts
    assert "Test query 1" in texts
    assert "Test query 2" in texts


def test_get_cluster_info_with_multiple_runs():
    """Test that get_cluster_info filters by run_id correctly."""
    db = Database(":memory:")
    db.create_tables()

    with db.get_session() as session:
        # Create queries
        q1 = Query(conversation_id="c1", model="gpt-4", query_text="Query 1")
        q2 = Query(conversation_id="c2", model="gpt-4", query_text="Query 2")
        session.add(q1)
        session.add(q2)
        session.commit()
        session.refresh(q1)
        session.refresh(q2)

        # Create two runs
        run1 = ClusteringRun(run_id="run-1", algorithm="kmeans", num_clusters=2)
        run2 = ClusteringRun(run_id="run-2", algorithm="kmeans", num_clusters=2)
        session.add(run1)
        session.add(run2)
        session.commit()

        # Assign q1 to cluster 0 in run-1
        session.add(QueryCluster(run_id="run-1", query_id=q1.id, cluster_id=0))

        # Assign q2 to cluster 0 in run-2
        session.add(QueryCluster(run_id="run-2", query_id=q2.id, cluster_id=0))
        session.commit()

    # Check run-1 cluster 0 (should have q1)
    result1 = get_cluster_info(db, "run-1", 0)
    assert result1["size"] == 1
    assert result1["queries"][0]["text"] == "Query 1"

    # Check run-2 cluster 0 (should have q2)
    result2 = get_cluster_info(db, "run-2", 0)
    assert result2["size"] == 1
    assert result2["queries"][0]["text"] == "Query 2"


def test_get_cluster_info_with_various_models(db_with_clusters):
    """Test get_cluster_info with queries from different models."""
    db = Database(":memory:")
    db.create_tables()

    with db.get_session() as session:
        # Create queries with different models
        q1 = Query(conversation_id="c1", model="gpt-4", query_text="Query 1")
        q2 = Query(conversation_id="c2", model="claude-3", query_text="Query 2")
        q3 = Query(conversation_id="c3", model="llama", query_text="Query 3")
        session.add_all([q1, q2, q3])
        session.commit()

        for q in [q1, q2, q3]:
            session.refresh(q)

        # Create run and assign all to cluster 0
        run = ClusteringRun(run_id="test", algorithm="kmeans", num_clusters=1)
        session.add(run)
        session.commit()

        for q in [q1, q2, q3]:
            session.add(QueryCluster(run_id="test", query_id=q.id, cluster_id=0))
        session.commit()

    result = get_cluster_info(db, "test", 0)

    assert result["size"] == 3
    models = {q["model"] for q in result["queries"]}
    assert models == {"gpt-4", "claude-3", "llama"}


def test_get_cluster_info_returns_dict(db_with_clusters):
    """Test that get_cluster_info returns a dictionary with expected keys."""
    result = get_cluster_info(db_with_clusters, "test-run", 0)

    assert isinstance(result, dict)
    assert "run_id" in result
    assert "cluster_id" in result
    assert "size" in result
    assert "queries" in result

    # Check queries structure
    for query in result["queries"]:
        assert isinstance(query, dict)
        assert "id" in query
        assert "text" in query
        assert "model" in query


def test_get_cluster_info_large_cluster():
    """Test get_cluster_info with a larger cluster."""
    db = Database(":memory:")
    db.create_tables()

    with db.get_session() as session:
        # Create many queries
        queries = []
        for i in range(100):
            q = Query(
                conversation_id=f"conv-{i}",
                model="gpt-4",
                query_text=f"Query {i}",
            )
            session.add(q)
            queries.append(q)
        session.commit()

        for q in queries:
            session.refresh(q)

        # Create run
        run = ClusteringRun(run_id="test", algorithm="kmeans", num_clusters=1)
        session.add(run)
        session.commit()

        # Assign all to cluster 0
        for q in queries:
            session.add(QueryCluster(run_id="test", query_id=q.id, cluster_id=0))
        session.commit()

    result = get_cluster_info(db, "test", 0)

    assert result["size"] == 100
    assert len(result["queries"]) == 100
