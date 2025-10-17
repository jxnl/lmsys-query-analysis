"""Test cases for FastAPI endpoints."""

from fastapi.testclient import TestClient

from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import (
    ClusteringRun,
    ClusterSummary,
    Query,
)


def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "docs" in data


def test_list_runs_empty(client: TestClient):
    """Test listing runs when database is empty."""
    response = client.get("/api/clustering/runs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []


def test_list_runs_with_data(client: TestClient, db: Database):
    """Test listing runs with sample data."""
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="kmeans-50-20251010-120000",
            algorithm="kmeans",
            num_clusters=50,
            description="Test run",
            parameters={"embedding_model": "test-model", "embedding_provider": "openai"},
        )
        session.add(run)
        session.commit()

    response = client.get("/api/clustering/runs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["run_id"] == "kmeans-50-20251010-120000"


def test_get_run_not_found(client: TestClient):
    """Test getting a non-existent run."""
    response = client.get("/api/clustering/runs/nonexistent")
    assert response.status_code == 404


def test_get_run_detail(client: TestClient, db: Database):
    """Test getting run details."""
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="kmeans-50-20251010-120000",
            algorithm="kmeans",
            num_clusters=50,
            description="Test run",
            parameters={"embedding_model": "test-model"},
        )
        session.add(run)
        session.commit()

    response = client.get("/api/clustering/runs/kmeans-50-20251010-120000")
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "kmeans-50-20251010-120000"
    assert data["algorithm"] == "kmeans"
    assert data["num_clusters"] == 50


def test_list_queries_empty(client: TestClient):
    """Test listing queries when empty."""
    response = client.get("/api/queries")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0


def test_list_queries_with_data(client: TestClient, db: Database):
    """Test listing queries with sample data."""
    with db.get_session() as session:
        query = Query(
            conversation_id="test-conv-1",
            model="gpt-4",
            query_text="What is Python?",
            language="en",
        )
        session.add(query)
        session.commit()

    response = client.get("/api/queries")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["query_text"] == "What is Python?"


def test_search_queries_fulltext(client: TestClient, db: Database):
    """Test full-text search for queries."""
    with db.get_session() as session:
        query = Query(
            conversation_id="test-conv-1",
            model="gpt-4",
            query_text="How do I write Python code?",
            language="en",
        )
        session.add(query)
        session.commit()

    response = client.get("/api/search/queries?text=Python&mode=fulltext")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1


def test_search_clusters_fulltext(client: TestClient, db: Database):
    """Test full-text search for clusters."""
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="test-run",
            algorithm="kmeans",
            num_clusters=1,
        )
        session.add(run)
        session.commit()

        summary = ClusterSummary(
            run_id="test-run",
            cluster_id=0,
            title="Python Programming",
            description="Questions about Python",
            summary_run_id="summary-1",
        )
        session.add(summary)
        session.commit()

    response = client.get("/api/search/clusters?text=Python&mode=fulltext")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1


def test_list_hierarchies_empty(client: TestClient):
    """Test listing hierarchies when empty."""
    response = client.get("/api/hierarchy/")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0


def test_list_summaries_empty(client: TestClient):
    """Test listing summaries when empty."""
    response = client.get("/api/summaries/")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0


def test_get_cluster_metadata_empty(client: TestClient):
    """Test getting cluster metadata when none exists."""
    response = client.get("/api/curation/clusters/0/metadata?run_id=test-run")
    assert response.status_code == 200
    data = response.json()
    assert data["coherence_score"] is None


def test_post_endpoints_not_implemented(client: TestClient):
    """Test that POST endpoints return 501 Not Implemented."""
    response = client.post("/api/clustering/kmeans")
    assert response.status_code == 501

    response = client.post("/api/clustering/hdbscan")
    assert response.status_code == 501

    response = client.post("/api/hierarchy/")
    assert response.status_code == 501

    response = client.post("/api/summaries/")
    assert response.status_code == 501

    response = client.post("/api/curation/queries/1/move")
    assert response.status_code == 501

    response = client.post("/api/curation/clusters/1/rename")
    assert response.status_code == 501
