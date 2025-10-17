"""Unit tests for summarization CLI commands."""

from unittest.mock import Mock, patch


@patch("lmsys_query_analysis.cli.commands.summarization.cluster_service")
@patch("lmsys_query_analysis.cli.commands.summarization.run_service")
@patch("lmsys_query_analysis.cli.commands.summarization.ClusterSummarizer")
@patch("lmsys_query_analysis.cli.commands.summarization.get_db")
@patch("lmsys_query_analysis.cli.commands.summarization.create_chroma_client")
def test_summarize_command(
    mock_chroma_client, mock_get_db, mock_summarizer_class, mock_run_service, mock_cluster_service
):
    """Test summarize command."""
    from lmsys_query_analysis.cli.commands.summarization import summarize
    from lmsys_query_analysis.db.models import ClusteringRun

    # Setup mocks
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    mock_get_db.return_value = mock_db

    # Mock run
    mock_run = ClusteringRun(
        run_id="test-run",
        algorithm="kmeans",
        num_clusters=10,
        parameters={"embedding_model": "test", "embedding_provider": "test"},
    )
    mock_run_service.get_run.return_value = mock_run

    # Mock cluster service
    mock_cluster_service.get_cluster_ids_for_run.return_value = [1, 2]
    mock_cluster_service.get_cluster_queries_with_texts.return_value = ([], ["query1", "query2"])

    # Mock session operations
    mock_session.exec.return_value.first.return_value = None
    mock_session.add = Mock()
    mock_session.commit = Mock()

    # Mock chroma
    mock_chroma = Mock()
    mock_chroma_client.return_value = mock_chroma

    # Mock summarizer
    mock_summarizer = Mock()
    mock_summarizer.generate_batch_summaries.return_value = {
        1: {"title": "Cluster 1", "description": "Desc 1", "sample_queries": ["q1"]},
        2: {"title": "Cluster 2", "description": "Desc 2", "sample_queries": ["q2"]},
    }
    mock_summarizer_class.return_value = mock_summarizer

    # Execute command
    summarize(
        run_id="test-run",
        cluster_id=None,
        alias="v1",
        db_path="/tmp/test.db",
        use_chroma=False,
        chroma_path="/tmp/chroma",
    )

    # Verify
    mock_get_db.assert_called_once()
    mock_summarizer_class.assert_called_once()
    mock_cluster_service.get_cluster_ids_for_run.assert_called_once()
    mock_cluster_service.get_cluster_queries_with_texts.assert_called()
