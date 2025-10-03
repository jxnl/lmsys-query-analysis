"""Tests for CLI commands."""
import pytest
from typer.testing import CliRunner
from lmsys_query_analysis.cli.main import app

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
