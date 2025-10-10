"""Test hierarchy endpoint query count calculation."""

from fastapi.testclient import TestClient

from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import (
    Query,
    ClusteringRun,
    QueryCluster,
    ClusterHierarchy,
)


def test_hierarchy_query_counts_bottom_up(db: Database, client: TestClient):
    """Test that hierarchy query counts are calculated bottom-up correctly.

    This test creates a 3-level hierarchy:
    - Level 0 (leaf): 3 clusters with 5, 10, and 15 queries
    - Level 1 (mid): 1 cluster containing the 3 leaf clusters (should have 30 queries)
    - Level 2 (root): 1 cluster containing the mid-level cluster (should have 30 queries)

    This ensures the bottom-up aggregation works correctly at all levels.
    """
    with db.get_session() as session:
        # Create test data
        run_id = "test-run-1"

        # Create clustering run
        run = ClusteringRun(
            run_id=run_id,
            algorithm="test",
            num_clusters=5,
            parameters={"test": True},
        )
        session.add(run)

        # Create queries and assign to leaf clusters
        # Leaf cluster 0: 5 queries
        for i in range(5):
            query = Query(
                conversation_id=f"conv-0-{i}",
                model="test-model",
                query_text=f"Query {i}",
            )
            session.add(query)
            session.flush()
            qc = QueryCluster(run_id=run_id, query_id=query.id, cluster_id=0)
            session.add(qc)

        # Leaf cluster 1: 10 queries
        for i in range(10):
            query = Query(
                conversation_id=f"conv-1-{i}",
                model="test-model",
                query_text=f"Query {i}",
            )
            session.add(query)
            session.flush()
            qc = QueryCluster(run_id=run_id, query_id=query.id, cluster_id=1)
            session.add(qc)

        # Leaf cluster 2: 15 queries
        for i in range(15):
            query = Query(
                conversation_id=f"conv-2-{i}",
                model="test-model",
                query_text=f"Query {i}",
            )
            session.add(query)
            session.flush()
            qc = QueryCluster(run_id=run_id, query_id=query.id, cluster_id=2)
            session.add(qc)

        # Create hierarchy
        hierarchy_run_id = "hier-test-run-1"

        # Level 0 (leaf clusters)
        leaf0 = ClusterHierarchy(
            hierarchy_run_id=hierarchy_run_id,
            run_id=run_id,
            cluster_id=0,
            parent_cluster_id=100,  # Parent is mid-level cluster
            level=0,
            children_ids=[],
            title="Leaf Cluster 0",
        )
        leaf1 = ClusterHierarchy(
            hierarchy_run_id=hierarchy_run_id,
            run_id=run_id,
            cluster_id=1,
            parent_cluster_id=100,
            level=0,
            children_ids=[],
            title="Leaf Cluster 1",
        )
        leaf2 = ClusterHierarchy(
            hierarchy_run_id=hierarchy_run_id,
            run_id=run_id,
            cluster_id=2,
            parent_cluster_id=100,
            level=0,
            children_ids=[],
            title="Leaf Cluster 2",
        )

        # Level 1 (mid-level cluster containing all leaf clusters)
        mid = ClusterHierarchy(
            hierarchy_run_id=hierarchy_run_id,
            run_id=run_id,
            cluster_id=100,
            parent_cluster_id=200,  # Parent is root cluster
            level=1,
            children_ids=[0, 1, 2],
            title="Mid-Level Cluster",
        )

        # Level 2 (root cluster containing mid-level cluster)
        root = ClusterHierarchy(
            hierarchy_run_id=hierarchy_run_id,
            run_id=run_id,
            cluster_id=200,
            parent_cluster_id=None,
            level=2,
            children_ids=[100],
            title="Root Cluster",
        )

        session.add_all([leaf0, leaf1, leaf2, mid, root])
        session.commit()

    # Test the API endpoint
    response = client.get(f"/api/hierarchy/{hierarchy_run_id}")
    assert response.status_code == 200

    data = response.json()
    nodes = {node["cluster_id"]: node for node in data["nodes"]}

    # Verify leaf cluster counts (direct query counts)
    assert nodes[0]["query_count"] == 5, "Leaf cluster 0 should have 5 queries"
    assert nodes[1]["query_count"] == 10, "Leaf cluster 1 should have 10 queries"
    assert nodes[2]["query_count"] == 15, "Leaf cluster 2 should have 15 queries"

    # Verify mid-level cluster count (sum of children)
    assert nodes[100]["query_count"] == 30, "Mid-level cluster should have 30 queries (5+10+15)"

    # Verify root cluster count (sum of descendants)
    assert nodes[200]["query_count"] == 30, "Root cluster should have 30 queries (sum of all descendants)"

    # Verify total queries in response
    assert data["total_queries"] == 30, "Total queries should be 30"


def test_hierarchy_empty_clusters(db: Database, client: TestClient):
    """Test that clusters with no queries show 0 count."""
    with db.get_session() as session:
        run_id = "test-run-empty"
        hierarchy_run_id = "hier-test-run-empty"

        # Create clustering run
        run = ClusteringRun(
            run_id=run_id,
            algorithm="test",
            num_clusters=2,
            parameters={"test": True},
        )
        session.add(run)

        # Create hierarchy with no queries
        leaf = ClusterHierarchy(
            hierarchy_run_id=hierarchy_run_id,
            run_id=run_id,
            cluster_id=0,
            parent_cluster_id=100,
            level=0,
            children_ids=[],
            title="Empty Leaf",
        )
        root = ClusterHierarchy(
            hierarchy_run_id=hierarchy_run_id,
            run_id=run_id,
            cluster_id=100,
            parent_cluster_id=None,
            level=1,
            children_ids=[0],
            title="Empty Root",
        )

        session.add_all([leaf, root])
        session.commit()

    # Test the API endpoint
    response = client.get(f"/api/hierarchy/{hierarchy_run_id}")
    assert response.status_code == 200

    data = response.json()
    nodes = {node["cluster_id"]: node for node in data["nodes"]}

    # Both should have 0 queries
    assert nodes[0]["query_count"] == 0, "Empty leaf should have 0 queries"
    assert nodes[100]["query_count"] == 0, "Empty root should have 0 queries"
    assert data["total_queries"] == 0, "Total queries should be 0"
