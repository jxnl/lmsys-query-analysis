"""Tests for CLI commands."""

from unittest.mock import Mock, patch
from typer.testing import CliRunner
from lmsys_query_analysis.cli.main import app
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun, ClusterSummary, Query

runner = CliRunner()


def test_cli_help():
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LMSYS Query Analysis CLI" in result.stdout


def test_load_help():
    """Test load command help."""
    result = runner.invoke(app, ["load", "--help"])
    assert result.exit_code == 0
    assert "Download and load LMSYS-1M dataset" in result.stdout


def test_list_help():
    """Test list command help."""
    result = runner.invoke(app, ["list", "--help"])
    assert result.exit_code == 0
    assert "List queries" in result.stdout


def test_cluster_help():
    """Test cluster command help."""
    result = runner.invoke(app, ["cluster", "--help"])
    assert result.exit_code == 0
    assert "clustering" in result.stdout.lower()


def test_runs_command():
    """Test runs command."""
    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 0


def test_list_clusters_sorted_by_num_queries(tmp_path):
    """Ensure list-clusters sorts clusters by num_queries descending."""
    db_path = tmp_path / "queries.db"
    db = Database(db_path)
    db.create_tables()
    session = db.get_session()
    try:
        # Seed a run and several cluster summaries with varying counts
        run_id = "run-sort-test"
        run = ClusteringRun(run_id=run_id, algorithm="kmeans", num_clusters=3)
        session.add(run)
        session.commit()

        summaries = [
            ClusterSummary(
                run_id=run_id,
                cluster_id=1,
                title="first",
                num_queries=10,
                representative_queries=["ex-a", "ex-b"],
            ),
            ClusterSummary(
                run_id=run_id,
                cluster_id=2,
                title="second",
                num_queries=25,
                representative_queries=["ex-c", "ex-d"],
            ),
            ClusterSummary(
                run_id=run_id,
                cluster_id=3,
                title="none",
                num_queries=None,
                representative_queries=["ex-e"],
            ),
            ClusterSummary(
                run_id=run_id,
                cluster_id=4,
                title="tie10a",
                num_queries=10,
                representative_queries=["ex-f"],
            ),
        ]
        for s in summaries:
            session.add(s)
        session.commit()

        # Invoke CLI
        result = runner.invoke(
            app, ["list-clusters", run_id, "--db-path", str(db_path)]
        )
        assert result.exit_code == 0

        out = result.stdout
        # Expect order: 25 ("second"), then 10s (cluster_id ascending among ties), then None ("none")
        pos_second = out.find("second")
        pos_first = out.find("first")
        pos_tie10a = out.find("tie10a")
        pos_none = out.find("none")

        assert (
            pos_second != -1 and pos_first != -1 and pos_tie10a != -1 and pos_none != -1
        )
        assert pos_second < pos_first  # 25 before 10
        assert pos_first < pos_tie10a  # cluster_id 1 before 4 among ties
        assert pos_tie10a < pos_none  # None last
    finally:
        session.close()


def test_list_clusters_show_examples(tmp_path):
    """Ensure list-clusters shows example queries when requested."""
    db_path = tmp_path / "queries.db"
    db = Database(db_path)
    db.create_tables()
    session = db.get_session()
    try:
        run_id = "run-examples"
        run = ClusteringRun(run_id=run_id, algorithm="kmeans", num_clusters=1)
        session.add(run)
        session.commit()

        reps = [
            "First example query about pandas DataFrame indexing",
            "Second example query regarding Docker build cache",
            "Third example mentioning SQL window functions",
        ]
        s = ClusterSummary(
            run_id=run_id,
            cluster_id=7,
            title="Mixed tech clusters",
            description="Mixed queries across Python, DevOps, and SQL",
            num_queries=42,
            representative_queries=reps,
        )
        session.add(s)
        session.commit()

        result = runner.invoke(
            app,
            [
                "list-clusters",
                run_id,
                "--db-path",
                str(db_path),
                "--show-examples",
                "2",
                "--example-width",
                "50",
            ],
        )
        assert result.exit_code == 0
        out = result.stdout
        assert "Clusters for Run: run-examples" in out
        # First two examples should appear, truncated width respected (with ellipsis)
        expected1 = "First example query about pandas DataFrame indexing"
        expected2 = "Second example query regarding Docker build cache"
        printed1 = expected1 if len(expected1) <= 50 else expected1[: 50 - 3] + "..."
        printed2 = expected2 if len(expected2) <= 50 else expected2[: 50 - 3] + "..."
        assert printed1 in out
        assert printed2 in out
        # Third example should not be shown due to limit=2
        assert "Third example mentioning SQL window functions" not in out
        assert "Total: 1 clusters" in out
    finally:
        session.close()


def test_load_command_with_adapter(tmp_path):
    """Test that load command works with adapter refactor."""
    db_path = tmp_path / "adapter-test.db"
    
    # Mock the HuggingFaceAdapter to avoid actual HF download
    with patch('lmsys_query_analysis.db.loader.HuggingFaceAdapter') as mock_adapter_class:
        mock_data = [
            {
                "conversation_id": "test1",
                "query_text": "What is Python?",
                "model": "gpt-4",
                "language": "en",
                "timestamp": None,
                "extra_metadata": {},
            },
            {
                "conversation_id": "test2",
                "query_text": "How does async/await work?",
                "model": "claude-3",
                "language": "en",
                "timestamp": None,
                "extra_metadata": {},
            },
            {
                "conversation_id": "test3",
                "query_text": "Explain decorators",
                "model": "gpt-4",
                "language": "en",
                "timestamp": None,
                "extra_metadata": {},
            },
        ]
        mock_adapter_instance = Mock()
        mock_adapter_instance.__iter__ = Mock(return_value=iter(mock_data))
        mock_adapter_instance.__len__ = Mock(return_value=len(mock_data))
        mock_adapter_class.return_value = mock_adapter_instance
        
        result = runner.invoke(
            app,
            ["load", "--limit", "10", "--db-path", str(db_path)]
        )
        
        assert result.exit_code == 0
        assert "loaded" in result.stdout.lower() or "complete" in result.stdout.lower()
        
        # Verify adapter was called with expected parameters
        mock_adapter_class.assert_called_once()
        call_kwargs = mock_adapter_class.call_args[1]
        assert call_kwargs["limit"] == 10
        
        # Verify data was actually loaded into the database
        db = Database(db_path)
        session = db.get_session()
        try:
            queries = session.query(Query).all()
            assert len(queries) == 3
            assert queries[0].conversation_id == "test1"
            assert queries[0].query_text == "What is Python?"
            assert queries[1].conversation_id == "test2"
            assert queries[2].conversation_id == "test3"
        finally:
            session.close()
