"""Unit tests for analysis CLI commands."""

from unittest.mock import Mock, patch


@patch("lmsys_query_analysis.cli.commands.analysis.get_db")
def test_runs_command(mock_get_db):
    """Test runs listing command."""
    from lmsys_query_analysis.cli.commands.analysis import runs
    from lmsys_query_analysis.db.models import ClusteringRun

    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)

    mock_runs = [ClusteringRun(run_id="run-1", algorithm="kmeans", num_clusters=100, parameters={})]
    mock_result = Mock()
    mock_result.all.return_value = mock_runs
    mock_session.exec.return_value = mock_result

    mock_get_db.return_value = mock_db

    runs(db_path="/tmp/test.db", latest=False, table=False, xml=False)

    mock_get_db.assert_called_once()
    mock_session.exec.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.analysis.get_db")
def test_list_clusters_command(mock_get_db):
    """Test list-clusters command."""
    from lmsys_query_analysis.cli.commands.analysis import list_clusters
    from lmsys_query_analysis.db.models import ClusterSummary

    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)

    mock_summaries = [
        ClusterSummary(
            run_id="test-run",
            cluster_id=1,
            title="Test Cluster",
            description="Test description",
            num_queries=100,
            representative_queries=["query1", "query2"],
        )
    ]
    mock_result = Mock()
    mock_result.all.return_value = mock_summaries
    mock_session.exec.return_value = mock_result

    mock_get_db.return_value = mock_db

    with patch(
        "lmsys_query_analysis.cli.commands.analysis.cluster_service"
    ) as mock_cluster_service:
        mock_cluster_service.list_cluster_summaries.return_value = mock_summaries
        list_clusters(
            run_id="test-run",
            db_path="/tmp/test.db",
            show_examples=0,
            example_width=50,
            table=False,
            xml=False,
        )

    mock_get_db.assert_called_once()
    mock_cluster_service.list_cluster_summaries.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.analysis.query_service")
@patch("lmsys_query_analysis.cli.commands.analysis.cluster_service")
@patch("lmsys_query_analysis.cli.commands.analysis.get_db")
def test_inspect_cluster_command(mock_get_db, mock_cluster_service, mock_query_service):
    """Test inspect command for a specific cluster."""
    from lmsys_query_analysis.cli.commands.analysis import inspect
    from lmsys_query_analysis.db.models import ClusterSummary, Query

    mock_db = Mock()
    mock_get_db.return_value = mock_db

    mock_queries = [
        Query(id=1, query_text="Test query", model="gpt-4", conversation_id="c1", language="en")
    ]
    mock_query_service.get_cluster_queries.return_value = mock_queries

    mock_summary = ClusterSummary(
        run_id="test-run", cluster_id=1, title="Test Cluster", description="Test description"
    )
    mock_cluster_service.get_cluster_summary.return_value = mock_summary

    inspect(run_id="test-run", cluster_id=1, show_queries=10, db_path="/tmp/test.db")

    mock_get_db.assert_called_once()
    mock_query_service.get_cluster_queries.assert_called_once()
    mock_cluster_service.get_cluster_summary.assert_called_once()


@patch("lmsys_query_analysis.cli.commands.analysis.export_service")
@patch("lmsys_query_analysis.cli.commands.analysis.get_db")
def test_export_command(mock_get_db, mock_export_service):
    """Test export command."""
    from lmsys_query_analysis.cli.commands.analysis import export

    mock_db = Mock()
    mock_get_db.return_value = mock_db
    mock_export_service.get_export_data.return_value = [{"cluster_id": 1, "query": "test"}]
    mock_export_service.export_to_csv.return_value = 1

    export(run_id="test-run", output="/tmp/export.csv", format="csv", db_path="/tmp/test.db")

    mock_get_db.assert_called_once()
    mock_export_service.get_export_data.assert_called_once()
    mock_export_service.export_to_csv.assert_called_once()
