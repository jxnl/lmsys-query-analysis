"""Unit tests for clustering CLI commands."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path


@patch('lmsys_query_analysis.cli.commands.clustering.run_kmeans_clustering')
@patch('lmsys_query_analysis.cli.commands.clustering.get_db')
@patch('lmsys_query_analysis.cli.commands.clustering.create_chroma_client')
@patch('lmsys_query_analysis.cli.commands.clustering.parse_embedding_model')
def test_kmeans_command_basic(mock_parse, mock_chroma_client, mock_get_db, mock_run_kmeans):
    """Test basic kmeans clustering command."""
    from lmsys_query_analysis.cli.commands.clustering import cluster_kmeans
    
    # Setup mocks
    mock_parse.return_value = ("all-MiniLM-L6-v2", "sentence-transformers")
    mock_db = Mock()
    mock_get_db.return_value = mock_db
    mock_run_kmeans.return_value = "test-run-id"
    
    # Execute command
    cluster_kmeans(
        n_clusters=200,
        db_path="/tmp/test.db",
        description="Test clustering",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_chroma=False,
        chroma_path="/tmp/chroma"
    )
    
    # Verify
    mock_get_db.assert_called_once()
    mock_run_kmeans.assert_called_once()


@patch('lmsys_query_analysis.cli.commands.clustering.run_hdbscan_clustering')
@patch('lmsys_query_analysis.cli.commands.clustering.get_db')
@patch('lmsys_query_analysis.cli.commands.clustering.parse_embedding_model')
def test_hdbscan_command_basic(mock_parse, mock_get_db, mock_run_hdbscan):
    """Test basic HDBSCAN clustering command."""
    from lmsys_query_analysis.cli.commands.clustering import cluster_hdbscan
    
    # Setup mocks
    mock_parse.return_value = ("all-MiniLM-L6-v2", "sentence-transformers")
    mock_db = Mock()
    mock_get_db.return_value = mock_db
    mock_run_hdbscan.return_value = "test-run-id"
    
    # Execute command
    cluster_hdbscan(
        min_cluster_size=15,
        db_path="/tmp/test.db",
        description="Test HDBSCAN",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        metric="euclidean",
        use_chroma=False,
        chroma_path="/tmp/chroma"
    )
    
    # Verify
    mock_get_db.assert_called_once()
    mock_run_hdbscan.assert_called_once()
