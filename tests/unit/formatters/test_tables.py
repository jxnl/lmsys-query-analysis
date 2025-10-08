"""Unit tests for table formatters."""

import pytest
from rich.table import Table
from lmsys_query_analysis.cli.formatters.tables import (
    format_queries_table,
    format_runs_table,
    format_cluster_summaries_table,
    format_loading_stats_table,
    format_backfill_summary_table,
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

