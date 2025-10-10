"""Unit tests for verify CLI commands."""

import pytest
from unittest.mock import Mock, patch


@patch('lmsys_query_analysis.cli.commands.verify.get_chroma')
@patch('lmsys_query_analysis.cli.commands.verify.get_db')
def test_verify_sync_command(mock_get_db, mock_get_chroma):
    """Test verify-sync command."""
    from lmsys_query_analysis.cli.commands.verify import verify_sync
    from lmsys_query_analysis.db.models import ClusteringRun, ClusterSummary
    
    # Setup mocks
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    mock_get_db.return_value = mock_db
    
    # Mock run and summaries
    mock_run = ClusteringRun(
        run_id="test-run",
        algorithm="kmeans",
        num_clusters=100,
        parameters={"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"}
    )
    mock_summaries = []
    
    mock_result_run = Mock()
    mock_result_run.first.return_value = mock_run
    mock_result_summaries = Mock()
    mock_result_summaries.all.return_value = mock_summaries
    
    # Set up exec to return different results based on call count
    mock_session.exec.side_effect = [mock_result_run, mock_result_summaries]
    
    # Mock chroma
    mock_chroma = Mock()
    mock_chroma.count_summaries.return_value = 0
    mock_chroma.list_runs_in_summaries.return_value = []
    mock_chroma.summaries_collection.name = "summaries_collection"
    mock_get_chroma.return_value = mock_chroma
    
    # Execute command
    verify_sync(
        run_id="test-run",
        chroma_path="/tmp/chroma",
        db_path="/tmp/test.db",
        json_out=False
    )
    
    # Verify
    mock_get_db.assert_called_once()
    mock_get_chroma.assert_called_once()

