"""Unit tests for JSON output formatters."""

from lmsys_query_analysis.cli.formatters.json_output import (
    format_chroma_collections_json,
    format_search_clusters_json,
    format_search_queries_json,
    format_verify_sync_json,
)


def test_format_search_queries_json():
    """Test formatting query search results as JSON."""

    class MockHit:
        def __init__(self, query_id, distance, snippet, model, language, cluster_id):
            self.query_id = query_id
            self.distance = distance
            self.snippet = snippet
            self.model = model
            self.language = language
            self.cluster_id = cluster_id

    hits = [
        MockHit("q1", 0.1, "Sample query 1", "gpt-4", "en", 0),
        MockHit("q2", 0.2, "Sample query 2", "gpt-4", "en", 1),
    ]

    result = format_search_queries_json(text="test query", run_id="test-run-001", hits=hits)

    assert result["text"] == "test query"
    assert result["run_id"] == "test-run-001"
    assert len(result["results"]) == 2
    assert result["results"][0]["query_id"] == "q1"
    assert result["results"][0]["distance"] == 0.1


def test_format_search_queries_json_with_facets():
    """Test formatting with facets."""
    result = format_search_queries_json(
        text="test", run_id="run1", hits=[], facets={"clusters": [{"cluster_id": 0, "count": 10}]}
    )

    assert "facets" in result
    assert "clusters" in result["facets"]


def test_format_search_queries_json_with_applied_clusters():
    """Test formatting with applied clusters."""
    applied = [{"cluster_id": 0, "title": "Test", "distance": 0.1}]

    result = format_search_queries_json(
        text="test", run_id="run1", hits=[], applied_clusters=applied
    )

    assert len(result["applied_clusters"]) == 1
    assert result["applied_clusters"][0]["cluster_id"] == 0


def test_format_search_clusters_json():
    """Test formatting cluster search results as JSON."""

    class MockHit:
        def __init__(self, cluster_id, distance, title, description, num_queries):
            self.cluster_id = cluster_id
            self.distance = distance
            self.title = title
            self.description = description
            self.num_queries = num_queries

    hits = [
        MockHit(0, 0.1, "Test Cluster", "Description", 100),
        MockHit(1, 0.2, "Another Cluster", "Another desc", 50),
    ]

    result = format_search_clusters_json(text="test query", run_id="test-run-001", hits=hits)

    assert result["text"] == "test query"
    assert result["run_id"] == "test-run-001"
    assert len(result["results"]) == 2
    assert result["results"][0]["cluster_id"] == 0
    assert result["results"][0]["title"] == "Test Cluster"


def test_format_chroma_collections_json():
    """Test formatting Chroma collections as JSON."""
    collections = [
        {
            "name": "queries",
            "count": 1000,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        },
        {
            "name": "summaries",
            "count": 50,
            "embedding_provider": "cohere",
            "embedding_model": "embed-v4.0",
            "embedding_dimension": 256,
        },
    ]

    result = format_chroma_collections_json(collections)

    assert "collections" in result
    assert len(result["collections"]) == 2
    assert result["collections"][0]["name"] == "queries"
    assert result["collections"][0]["count"] == 1000


def test_format_verify_sync_json():
    """Test formatting verification sync report as JSON."""
    space = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
    }
    sqlite = {"summary_count": 100}
    chroma = {
        "summary_count": 100,
        "runs_in_summaries": ["run1", "run2"],
        "summaries_collection": "summaries",
    }

    result = format_verify_sync_json(
        run_id="test-run-001", space=space, sqlite=sqlite, chroma=chroma, status="ok", issues=[]
    )

    assert result["run_id"] == "test-run-001"
    assert result["status"] == "ok"
    assert result["space"]["embedding_provider"] == "openai"
    assert result["sqlite"]["summary_count"] == 100
    assert result["chroma"]["summary_count"] == 100
    assert len(result["issues"]) == 0


def test_format_verify_sync_json_with_issues():
    """Test formatting verification with issues."""
    result = format_verify_sync_json(
        run_id="test-run-001",
        space={},
        sqlite={"summary_count": 100},
        chroma={"summary_count": 95},
        status="mismatch",
        issues=["Count mismatch", "Missing summaries"],
    )

    assert result["status"] == "mismatch"
    assert len(result["issues"]) == 2
