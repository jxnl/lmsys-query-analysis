"""Unit tests for hierarchy CLI commands."""

from unittest.mock import Mock, patch

import pytest
import typer


@patch("lmsys_query_analysis.cli.commands.hierarchy.anyio")
@patch("lmsys_query_analysis.cli.commands.hierarchy.cluster_service")
@patch("lmsys_query_analysis.cli.commands.hierarchy.get_db")
def test_merge_clusters_command(mock_get_db, mock_cluster_service, mock_anyio):
    """Test merge-clusters command."""
    from lmsys_query_analysis.cli.commands.hierarchy import merge_clusters_cmd
    from lmsys_query_analysis.db.models import ClusterSummary

    # Setup mocks
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    mock_get_db.return_value = mock_db

    # Mock cluster service
    mock_cluster_service.get_latest_summary_run_id.return_value = "summary-1"

    # Mock summaries
    mock_summaries = [
        ClusterSummary(
            run_id="test-run",
            cluster_id=1,
            summary_run_id="summary-1",
            title="Test 1",
            description="Desc 1",
        )
    ]
    mock_session.exec.return_value.all.return_value = mock_summaries
    mock_session.add = Mock()
    mock_session.commit = Mock()

    # Mock anyio.run to return hierarchy data
    mock_anyio.run.return_value = ("hierarchy-1", [])

    # Execute command
    merge_clusters_cmd(
        run_id="test-run",
        db_path="/tmp/test.db",
        target_levels=3,
        embedding_model="openai/text-embedding-3-small",
        model="openai/gpt-4o-mini",
    )

    # Verify
    mock_get_db.assert_called_once()
    mock_anyio.run.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.hierarchy.get_db")
def test_show_hierarchy_command(mock_get_db):
    """Test show-hierarchy command."""
    from lmsys_query_analysis.cli.commands.hierarchy import show_hierarchy_cmd

    # Setup mocks
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    mock_get_db.return_value = mock_db

    # Mock hierarchy
    mock_hierarchies = []
    mock_result = Mock()
    mock_result.all.return_value = mock_hierarchies
    mock_session.exec.return_value = mock_result

    # Execute command - expected to raise Exit when no hierarchies found
    with pytest.raises(typer.Exit):
        show_hierarchy_cmd(hierarchy_run_id="test-hierarchy", db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()
    mock_session.exec.assert_called_once()
