"""Unit tests for search CLI commands."""

from unittest.mock import Mock, patch


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
