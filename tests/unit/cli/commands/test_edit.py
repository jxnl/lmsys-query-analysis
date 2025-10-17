"""Unit tests for edit CLI commands."""

from unittest.mock import Mock, patch


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_view_query_command(mock_get_db, mock_curation_service):
    """Test view-query command."""
    from lmsys_query_analysis.cli.commands.edit import view_query
    from lmsys_query_analysis.db.models import Query

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock query details
    mock_query = Query(id=1, query_text="test", model="gpt-4", conversation_id="c1")
    mock_curation_service.get_query_details.return_value = {"query": mock_query, "clusters": []}

    # Execute command
    view_query(query_id=1, db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()
    mock_curation_service.get_query_details.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_move_query_command(mock_get_db, mock_curation_service):
    """Test move-query command."""
    from lmsys_query_analysis.cli.commands.edit import move_query

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock move result
    mock_curation_service.move_query.return_value = {"from_cluster_id": 1, "to_cluster_id": 2}

    # Execute command
    move_query(run_id="test-run", query_id=1, to_cluster=2, db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()
    mock_curation_service.move_query.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_move_queries_command(mock_get_db, mock_curation_service):
    """Test move-queries command."""
    from lmsys_query_analysis.cli.commands.edit import move_queries

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock batch move result
    mock_curation_service.move_queries_batch.return_value = {"moved": 2, "failed": 0, "errors": []}

    # Execute command
    move_queries(run_id="test-run", query_ids="1,2", to_cluster=2, db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()
    mock_curation_service.move_queries_batch.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_rename_cluster_command(mock_get_db, mock_curation_service):
    """Test rename-cluster command."""
    from lmsys_query_analysis.cli.commands.edit import rename_cluster

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock rename result
    mock_curation_service.rename_cluster.return_value = {
        "old_title": "Old Title",
        "new_title": "New Title",
    }

    # Execute command
    rename_cluster(run_id="test-run", cluster_id=1, title="New Title", db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()
    mock_curation_service.rename_cluster.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_merge_clusters_command(mock_get_db, mock_curation_service):
    """Test merge-clusters command."""
    from lmsys_query_analysis.cli.commands.edit import merge_clusters

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock merge result
    mock_curation_service.merge_clusters.return_value = {
        "queries_moved": 10,
        "new_title": "Merged Cluster",
    }

    # Execute command
    merge_clusters(run_id="test-run", source="2,3", target=1, db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()
    mock_curation_service.merge_clusters.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_split_cluster_command(mock_get_db, mock_curation_service):
    """Test split-cluster command."""
    from lmsys_query_analysis.cli.commands.edit import split_cluster

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock split result
    mock_curation_service.split_cluster.return_value = {
        "new_cluster_id": 10,
        "new_title": "Split Cluster",
        "queries_moved": 3,
    }

    # Execute command
    split_cluster(
        run_id="test-run",
        cluster_id=1,
        query_ids="1,2,3",
        new_title="Split Cluster",
        new_description="Description",
        db_path="/tmp/test.db",
    )

    # Verify
    mock_get_db.assert_called_once()
    mock_curation_service.split_cluster.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_delete_cluster_command(mock_get_db):
    """Test delete-cluster command."""
    from lmsys_query_analysis.cli.commands.edit import delete_cluster
    from lmsys_query_analysis.db.models import ClusterSummary

    # Setup mocks
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    mock_get_db.return_value = mock_db

    # Mock queries and summary
    mock_queries = []
    mock_summary = ClusterSummary(run_id="test", cluster_id=1, title="Test")

    def exec_side_effect(query):
        result = Mock()
        result.all.return_value = mock_queries
        result.first.return_value = mock_summary
        return result

    mock_session.exec.side_effect = exec_side_effect
    mock_session.delete = Mock()
    mock_session.commit = Mock()

    # Execute command
    delete_cluster(run_id="test-run", cluster_id=1, db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_tag_cluster_command(mock_get_db, mock_curation_service):
    """Test tag-cluster command."""
    from lmsys_query_analysis.cli.commands.edit import tag_cluster

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock tag result
    mock_curation_service.tag_cluster.return_value = {
        "metadata": {"coherence_score": 4, "quality": "high", "notes": "Good cluster", "flags": []}
    }

    # Execute command
    tag_cluster(
        run_id="test-run",
        cluster_id=1,
        coherence=4,
        quality="high",
        notes="Good cluster",
        db_path="/tmp/test.db",
    )

    # Verify
    mock_get_db.assert_called_once()
    mock_curation_service.tag_cluster.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_flag_cluster_command(mock_get_db, mock_curation_service):
    """Test flag-cluster command."""
    from lmsys_query_analysis.cli.commands.edit import flag_cluster

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock metadata with no existing flags
    mock_metadata = Mock()
    mock_metadata.flags = []
    mock_curation_service.get_cluster_metadata.return_value = mock_metadata
    mock_curation_service.tag_cluster.return_value = {
        "metadata": {
            "flags": ["needs_review"],
            "coherence_score": None,
            "quality": None,
            "notes": None,
        }
    }

    # Execute command
    flag_cluster(run_id="test-run", cluster_id=1, flag="needs_review", db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()
    mock_curation_service.get_cluster_metadata.assert_called_once()
    mock_curation_service.tag_cluster.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.curation_service")
@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_history_command(mock_get_db, mock_curation_service):
    """Test history command."""
    from lmsys_query_analysis.cli.commands.edit import history

    # Setup mocks
    mock_db = Mock()
    mock_get_db.return_value = mock_db

    # Mock edits
    mock_curation_service.get_cluster_edit_history.return_value = []

    # Execute command
    history(run_id="test-run", db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_audit_command(mock_get_db):
    """Test audit command."""
    from lmsys_query_analysis.cli.commands.edit import audit

    # Setup mocks
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    mock_get_db.return_value = mock_db

    # Mock queries
    mock_queries = []
    mock_result = Mock()
    mock_result.all.return_value = mock_queries
    mock_session.exec.return_value = mock_result

    # Execute command
    audit(run_id="test-run", db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_orphaned_command(mock_get_db):
    """Test orphaned command."""
    from lmsys_query_analysis.cli.commands.edit import orphaned

    # Setup mocks
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    mock_get_db.return_value = mock_db

    # Mock queries
    mock_queries = []
    mock_result = Mock()
    mock_result.all.return_value = mock_queries
    mock_session.exec.return_value = mock_result

    # Execute command
    orphaned(run_id="test-run", db_path="/tmp/test.db")

    # Verify
    mock_get_db.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.edit.get_db")
def test_select_bad_clusters_command(mock_get_db):
    """Test select-bad-clusters command."""
    from lmsys_query_analysis.cli.commands.edit import select_bad_clusters

    # Setup mocks
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    mock_get_db.return_value = mock_db

    # Mock summaries
    mock_summaries = []
    mock_result = Mock()
    mock_result.all.return_value = mock_summaries
    mock_session.exec.return_value = mock_result

    # Execute command
    select_bad_clusters(run_id="test-run", db_path="/tmp/test.db", min_size=5)

    # Verify
    mock_get_db.assert_called_once()
