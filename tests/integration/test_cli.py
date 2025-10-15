"""Tests for CLI commands."""

import pytest
from typer.testing import CliRunner
from lmsys_query_analysis.cli.main import app
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun, ClusterSummary

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
    assert "Download and load dataset from Hugging Face" in result.stdout


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
