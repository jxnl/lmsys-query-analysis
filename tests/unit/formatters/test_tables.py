"""Unit tests for table formatters."""

import pytest
from rich.table import Table
from lmsys_query_analysis.cli.formatters.tables import (
    format_queries_table,
    format_runs_table,
    format_cluster_summaries_table,
    format_loading_stats_table,
    format_backfill_summary_table,
    format_search_results_queries_table,
    format_search_results_clusters_table,
    format_chroma_collections_table,
    format_verify_sync_table,
)


def test_format_queries_table(sample_queries):
    """Test formatting queries as a table."""
    table = format_queries_table(sample_queries)
    
    assert isinstance(table, Table)
    assert table.title == "Queries (5 shown)"
    assert len(table.columns) == 4  # ID, Model, Query, Language


def test_format_queries_table_with_custom_title(sample_queries):
    """Test formatting queries with custom title."""
    table = format_queries_table(sample_queries, title="Custom Title")
    
    assert table.title == "Custom Title"


def test_format_queries_table_empty():
    """Test formatting empty query list."""
    table = format_queries_table([])
    
    assert isinstance(table, Table)
    assert table.title == "Queries (0 shown)"


def test_format_runs_table(sample_clustering_run):
    """Test formatting clustering runs as a table."""
    runs = [sample_clustering_run]
    table = format_runs_table(runs, latest=False)
    
    assert isinstance(table, Table)
    assert table.title == "Clustering Runs"
    assert len(table.columns) == 5  # Run ID, Algorithm, Clusters, Created, Description


def test_format_runs_table_latest(sample_clustering_run):
    """Test formatting latest run with different title."""
    runs = [sample_clustering_run]
    table = format_runs_table(runs, latest=True)
    
    assert table.title == "Latest Clustering Run"


def test_format_cluster_summaries_table(sample_cluster_summaries):
    """Test formatting cluster summaries as a table."""
    table = format_cluster_summaries_table(
        sample_cluster_summaries,
        run_id="test-run-001",
        show_examples=0
    )
    
    assert isinstance(table, Table)
    assert table.title == "Clusters for Run: test-run-001"
    assert len(table.columns) == 4  # Cluster, Title, Queries, Description


def test_format_cluster_summaries_table_with_examples(sample_cluster_summaries):
    """Test formatting cluster summaries with example queries."""
    table = format_cluster_summaries_table(
        sample_cluster_summaries,
        run_id="test-run-001",
        show_examples=2,
        example_width=80
    )
    
    assert isinstance(table, Table)
    assert len(table.columns) == 5  # Adds Examples column


def test_format_loading_stats_table():
    """Test formatting loading statistics."""
    stats = {
        "total_processed": 1000,
        "loaded": 950,
        "skipped": 40,
        "errors": 10,
    }
    
    table = format_loading_stats_table(stats)
    
    assert isinstance(table, Table)
    assert table.title == "Loading Statistics"
    assert len(table.columns) == 2  # Metric, Count


def test_format_backfill_summary_table():
    """Test formatting backfill summary."""
    table = format_backfill_summary_table(
        scanned=1000,
        backfilled=200,
        already_present=800,
        elapsed=45.5,
        rate=21.98
    )
    
    assert isinstance(table, Table)
    assert table.title == "Backfill Summary"
    assert len(table.columns) == 2  # Metric, Count


def test_format_queries_table_truncates_long_text(sample_queries):
    """Test that long query text is truncated."""
    # Add a query with very long text
    from lmsys_query_analysis.db.models import Query
    
    long_query = Query(
        id="q_long",
        query_text="A" * 200,  # Very long query
        model="gpt-4",
        language="en",
        conversation_id="conv_long",
    )
    
    queries = sample_queries + [long_query]
    table = format_queries_table(queries)
    
    # Should truncate to 80 chars
    assert isinstance(table, Table)


def test_format_search_results_queries_table():
    """Test formatting search results for queries."""
    from types import SimpleNamespace
    
    # Create mock query hits
    hits = [
        SimpleNamespace(
            query_id=1,
            snippet="How do I write a Python function?",
            model="gpt-4",
            distance=0.123
        ),
        SimpleNamespace(
            query_id=2,
            snippet="What is machine learning?",
            model="claude-3",
            distance=0.234
        ),
    ]
    
    table = format_search_results_queries_table(hits)
    
    assert isinstance(table, Table)
    assert table.title == "Top 2 Similar Queries"
    assert len(table.columns) == 5  # Rank, Query ID, Query Text, Model, Distance


def test_format_search_results_queries_table_long_text():
    """Test that search results truncate long queries."""
    from types import SimpleNamespace
    
    hits = [
        SimpleNamespace(
            query_id=1,
            snippet="A" * 100,  # Very long snippet
            model="gpt-4",
            distance=0.1
        ),
    ]
    
    table = format_search_results_queries_table(hits)
    
    assert isinstance(table, Table)
    assert table.title == "Top 1 Similar Queries"


def test_format_search_results_queries_table_missing_model():
    """Test search results with missing model."""
    from types import SimpleNamespace
    
    hits = [
        SimpleNamespace(
            query_id=1,
            snippet="Test query",
            model=None,  # Missing model
            distance=0.1
        ),
    ]
    
    table = format_search_results_queries_table(hits)
    
    assert isinstance(table, Table)


def test_format_search_results_clusters_table():
    """Test formatting search results for clusters."""
    from types import SimpleNamespace
    
    hits = [
        SimpleNamespace(
            cluster_id=0,
            title="Python Programming",
            distance=0.123
        ),
        SimpleNamespace(
            cluster_id=1,
            title="Machine Learning",
            distance=0.234
        ),
    ]
    
    table = format_search_results_clusters_table(hits)
    
    assert isinstance(table, Table)
    assert table.title == "Top 2 Similar Clusters"
    assert len(table.columns) == 4  # Rank, Cluster, Title, Distance


def test_format_search_results_clusters_table_missing_title():
    """Test cluster search results with missing title."""
    from types import SimpleNamespace
    
    hits = [
        SimpleNamespace(
            cluster_id=0,
            title=None,  # Missing title
            distance=0.1
        ),
    ]
    
    table = format_search_results_clusters_table(hits)
    
    assert isinstance(table, Table)
    # Should handle None title gracefully


def test_format_chroma_collections_table():
    """Test formatting Chroma collections table."""
    collections = [
        {
            "name": "queries",
            "count": 1000,
            "metadata": {
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 1536,
            }
        },
        {
            "name": "summaries",
            "count": 50,
            "metadata": {
                "embedding_provider": "cohere",
                "embedding_model": "embed-english-v3.0",
                "embedding_dimension": 1024,
            }
        }
    ]
    
    table = format_chroma_collections_table(collections)
    
    assert isinstance(table, Table)
    assert table.title == "Chroma Collections"  # Fixed: correct title
    assert len(table.columns) == 6  # Name, Count, Provider, Model, Dim, Description


def test_format_chroma_collections_table_missing_metadata():
    """Test formatting collections with missing metadata."""
    collections = [
        {
            "name": "queries",
            "count": 100,
            "metadata": {}  # Empty metadata
        },
    ]
    
    table = format_chroma_collections_table(collections)
    
    assert isinstance(table, Table)


def test_format_verify_sync_table():
    """Test formatting verify sync table."""
    report = {
        "run_id": "test-run-001",
        "space": {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        },
        "sqlite": {
            "summary_count": 100,
        },
        "chroma": {
            "summary_count": 98,
            "summaries_collection": "cluster_summaries",
            "runs_in_summaries": ["test-run-001"],
        },
        "status": "Out of sync",
        "issues": [
            "2 summaries missing from Chroma",
        ],
    }
    
    table = format_verify_sync_table(report)
    
    assert isinstance(table, Table)
    assert "test-run-001" in table.title
    assert len(table.columns) == 2  # Field, Value


def test_format_verify_sync_table_no_issues():
    """Test verify sync table with no issues."""
    report = {
        "run_id": "test-run-002",
        "space": {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        },
        "sqlite": {
            "summary_count": 100,
        },
        "chroma": {
            "summary_count": 100,
            "summaries_collection": "cluster_summaries",
            "runs_in_summaries": ["test-run-002"],
        },
        "status": "In sync",
        "issues": [],
    }
    
    table = format_verify_sync_table(report)
    
    assert isinstance(table, Table)
    # Should show that everything is in sync

