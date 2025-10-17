"""Unit tests for search CLI commands."""

from unittest.mock import Mock, patch

import pytest
import typer


@patch("lmsys_query_analysis.cli.commands.search.create_queries_client")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_command(mock_get_db, mock_create_clusters, mock_create_queries):
    """Test search command."""
    from lmsys_query_analysis.cli.commands.search import search

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_client = Mock()
    mock_client.find.return_value = []
    mock_create_queries.return_value = mock_client

    mock_cclient = Mock()
    mock_cclient.find.return_value = []
    mock_create_clusters.return_value = mock_cclient

    # Execute command
    search(
        text="test query",
        search_type="queries",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=False,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Verify
    mock_get_db.assert_called_once()
    mock_create_queries.assert_called_once()
    mock_client.find.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_cluster_command(mock_get_db, mock_create_clusters):
    """Test search-cluster command."""
    from lmsys_query_analysis.cli.commands.search import search_cluster

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_client = Mock()
    mock_client.find.return_value = []
    mock_create_clusters.return_value = mock_client

    # Execute command
    search_cluster(
        text="test query",
        run_id="test-run",
        alias=None,
        summary_run_id=None,
        top_k=5,
        json_out=False,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Verify
    mock_get_db.assert_called_once()
    mock_create_clusters.assert_called_once()
    mock_client.find.assert_called_once()


def test_search_multiple_format_error():
    """Test search command with multiple output formats raises error."""
    from lmsys_query_analysis.cli.commands.search import search

    with pytest.raises(typer.Exit) as exc_info:
        search(
            text="test",
            search_type="queries",
            run_id=None,
            cluster_ids=None,
            within_clusters=None,
            top_clusters=10,
            n_results=10,
            n_candidates=250,
            by=None,
            facets=None,
            json_out=True,
            table=True,  # Both JSON and table
            xml=False,
            chroma_path="/tmp/chroma",
            embedding_model="openai/text-embedding-3-small",
        )
    assert exc_info.value.exit_code == 1


def test_search_cluster_multiple_format_error():
    """Test search_cluster command with multiple output formats raises error."""
    from lmsys_query_analysis.cli.commands.search import search_cluster

    with pytest.raises(typer.Exit) as exc_info:
        search_cluster(
            text="test",
            run_id=None,
            alias=None,
            summary_run_id=None,
            top_k=5,
            json_out=True,
            table=False,
            xml=True,  # Both JSON and XML
            chroma_path="/tmp/chroma",
            embedding_model="openai/text-embedding-3-small",
        )
    assert exc_info.value.exit_code == 1


def test_search_invalid_cluster_ids():
    """Test search command with invalid cluster IDs raises error."""
    from lmsys_query_analysis.cli.commands.search import search

    with pytest.raises(typer.Exit) as exc_info:
        search(
            text="test",
            search_type="queries",
            run_id="test-run",
            cluster_ids="abc,def",  # Invalid - not integers
            within_clusters=None,
            top_clusters=10,
            n_results=10,
            n_candidates=250,
            by=None,
            facets=None,
            json_out=False,
            table=False,
            xml=False,
            chroma_path="/tmp/chroma",
            embedding_model="openai/text-embedding-3-small",
        )
    assert exc_info.value.exit_code == 1


@patch("lmsys_query_analysis.cli.commands.search.create_queries_client")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_with_cluster_ids(mock_get_db, mock_create_clusters, mock_create_queries):
    """Test search command with cluster IDs filter."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_client = Mock()
    mock_client.find.return_value = []
    mock_create_queries.return_value = mock_client

    mock_cclient = Mock()
    mock_cclient.find.return_value = []
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="queries",
        run_id="test-run",
        cluster_ids="1,2,3",
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=False,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Verify cluster_ids were parsed
    mock_client.find.assert_called_once()
    call_kwargs = mock_client.find.call_args[1]
    assert call_kwargs["cluster_ids"] == [1, 2, 3]


@patch("lmsys_query_analysis.cli.commands.search.console")
@patch("lmsys_query_analysis.cli.commands.search.create_queries_client")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_with_json_output(mock_get_db, mock_create_clusters, mock_create_queries, mock_console):
    """Test search command with JSON output format."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.query_id = 1
    mock_hit.snippet = "test"  # Note: snippet, not query_text
    mock_hit.distance = 0.5
    mock_hit.cluster_id = None
    mock_hit.language = "en"
    mock_hit.model = "gpt-4"

    mock_client = Mock()
    mock_client.find.return_value = [mock_hit]
    mock_create_queries.return_value = mock_client

    mock_cclient = Mock()
    mock_cclient.find.return_value = []
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="queries",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=True,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find and print_json
    mock_client.find.assert_called_once()
    mock_console.print_json.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.create_queries_client")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_with_within_clusters(mock_get_db, mock_create_clusters, mock_create_queries):
    """Test search command with within-clusters semantic filtering."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_client = Mock()
    mock_client.find.return_value = []
    mock_create_queries.return_value = mock_client

    mock_cluster_hit = Mock()
    mock_cluster_hit.cluster_id = 1
    mock_cluster_hit.title = "Test Cluster"
    mock_cluster_hit.description = "Test Description"
    mock_cluster_hit.num_queries = 10
    mock_cluster_hit.distance = 0.5

    mock_cclient = Mock()
    mock_cclient.find.return_value = [mock_cluster_hit]
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="queries",
        run_id="test-run",
        cluster_ids=None,
        within_clusters="semantic filter",
        top_clusters=5,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=True,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call cluster find for within_clusters
    assert mock_cclient.find.call_count == 1


@patch("lmsys_query_analysis.cli.commands.search.create_queries_client")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_with_facets(mock_get_db, mock_create_clusters, mock_create_queries):
    """Test search command with facets."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_client = Mock()
    mock_client.find.return_value = []
    mock_client.facets.return_value = {}
    mock_create_queries.return_value = mock_client

    mock_cclient = Mock()
    mock_cclient.find.return_value = []
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="queries",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets="cluster,language",
        json_out=True,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call facets
    mock_client.facets.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.create_queries_client")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_with_groupby(mock_get_db, mock_create_clusters, mock_create_queries):
    """Test search command with group-by counts."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_client = Mock()
    mock_client.find.return_value = []
    mock_client.count.return_value = {"cluster_1": 5, "cluster_2": 3}
    mock_create_queries.return_value = mock_client

    mock_cclient = Mock()
    mock_cclient.find.return_value = []
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="queries",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by="cluster",
        facets=None,
        json_out=True,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call count
    mock_client.count.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.create_queries_client")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_queries_with_xml_output(mock_get_db, mock_create_clusters, mock_create_queries):
    """Test search command with XML output format."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.query_id = 1
    mock_hit.query_text = "test"
    mock_hit.distance = 0.5

    mock_client = Mock()
    mock_client.find.return_value = [mock_hit]
    mock_create_queries.return_value = mock_client

    mock_cclient = Mock()
    mock_cclient.find.return_value = []
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="queries",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=False,
        table=False,
        xml=True,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find
    mock_client.find.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.console")
@patch("lmsys_query_analysis.cli.commands.search.create_queries_client")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_queries_with_table_output(mock_get_db, mock_create_clusters, mock_create_queries, mock_console):
    """Test search command with table output format (default)."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.query_id = 1
    mock_hit.snippet = "test"  # Use snippet for tables
    mock_hit.model = "gpt-4"
    mock_hit.distance = 0.5

    mock_client = Mock()
    mock_client.find.return_value = [mock_hit]
    mock_create_queries.return_value = mock_client

    mock_cclient = Mock()
    mock_cclient.find.return_value = []
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="queries",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=False,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find and print (for table)
    mock_client.find.assert_called_once()
    mock_console.print.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.console")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_clusters_with_json_output(mock_get_db, mock_create_clusters, mock_console):
    """Test search clusters with JSON output format."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.cluster_id = 1
    mock_hit.title = "Test"
    mock_hit.distance = 0.5

    mock_cclient = Mock()
    mock_cclient.find.return_value = [mock_hit]
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="clusters",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=True,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find and print_json
    mock_cclient.find.assert_called_once()
    mock_console.print_json.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_clusters_with_xml_output(mock_get_db, mock_create_clusters):
    """Test search clusters with XML output format."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.cluster_id = 1
    mock_hit.title = "Test"
    mock_hit.distance = 0.5

    mock_cclient = Mock()
    mock_cclient.find.return_value = [mock_hit]
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="clusters",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=False,
        table=False,
        xml=True,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find
    mock_cclient.find.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_clusters_with_table_output(mock_get_db, mock_create_clusters):
    """Test search clusters with table output format (default)."""
    from lmsys_query_analysis.cli.commands.search import search

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.cluster_id = 1
    mock_hit.title = "Test"
    mock_hit.distance = 0.5

    mock_cclient = Mock()
    mock_cclient.find.return_value = [mock_hit]
    mock_create_clusters.return_value = mock_cclient

    search(
        text="test query",
        search_type="clusters",
        run_id="test-run",
        cluster_ids=None,
        within_clusters=None,
        top_clusters=10,
        n_results=10,
        n_candidates=250,
        by=None,
        facets=None,
        json_out=False,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find
    mock_cclient.find.assert_called_once()


def test_search_invalid_search_type():
    """Test search command with invalid search_type raises error."""
    from lmsys_query_analysis.cli.commands.search import search

    with pytest.raises(typer.Exit) as exc_info:
        search(
            text="test",
            search_type="invalid",
            run_id=None,
            cluster_ids=None,
            within_clusters=None,
            top_clusters=10,
            n_results=10,
            n_candidates=250,
            by=None,
            facets=None,
            json_out=False,
            table=False,
            xml=False,
            chroma_path="/tmp/chroma",
            embedding_model="openai/text-embedding-3-small",
        )
    assert exc_info.value.exit_code == 1


@patch("lmsys_query_analysis.cli.commands.search.console")
@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_cluster_with_json_output(mock_get_db, mock_create_clusters, mock_console):
    """Test search_cluster command with JSON output format."""
    from lmsys_query_analysis.cli.commands.search import search_cluster

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.cluster_id = 1
    mock_hit.title = "Test"
    mock_hit.distance = 0.5

    mock_client = Mock()
    mock_client.find.return_value = [mock_hit]
    mock_create_clusters.return_value = mock_client

    search_cluster(
        text="test query",
        run_id="test-run",
        alias=None,
        summary_run_id=None,
        top_k=5,
        json_out=True,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find and print_json
    mock_client.find.assert_called_once()
    mock_console.print_json.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_cluster_with_xml_output(mock_get_db, mock_create_clusters):
    """Test search_cluster command with XML output format."""
    from lmsys_query_analysis.cli.commands.search import search_cluster

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.cluster_id = 1
    mock_hit.title = "Test"
    mock_hit.distance = 0.5

    mock_client = Mock()
    mock_client.find.return_value = [mock_hit]
    mock_create_clusters.return_value = mock_client

    search_cluster(
        text="test query",
        run_id="test-run",
        alias=None,
        summary_run_id=None,
        top_k=5,
        json_out=False,
        table=False,
        xml=True,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find
    mock_client.find.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.search.create_clusters_client")
@patch("lmsys_query_analysis.cli.commands.search.get_db")
def test_search_cluster_with_table_output(mock_get_db, mock_create_clusters):
    """Test search_cluster command with table output format (default)."""
    from lmsys_query_analysis.cli.commands.search import search_cluster

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_hit = Mock()
    mock_hit.cluster_id = 1
    mock_hit.title = "Test"
    mock_hit.distance = 0.5

    mock_client = Mock()
    mock_client.find.return_value = [mock_hit]
    mock_create_clusters.return_value = mock_client

    search_cluster(
        text="test query",
        run_id="test-run",
        alias=None,
        summary_run_id=None,
        top_k=5,
        json_out=False,
        table=False,
        xml=False,
        chroma_path="/tmp/chroma",
        embedding_model="openai/text-embedding-3-small",
    )

    # Should call find
    mock_client.find.assert_called_once()
