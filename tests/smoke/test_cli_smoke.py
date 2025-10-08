"""Smoke tests for CLI commands with real operations."""

import pytest
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from lmsys_query_analysis.cli.main import app

runner = CliRunner()


@pytest.mark.smoke
def test_cli_help_commands():
    """Test that all CLI commands show help properly."""
    # Main help
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LMSYS Query Analysis CLI" in result.stdout
    
    # Subcommand helps
    for cmd in ["cluster", "chroma", "verify"]:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0


@pytest.mark.smoke
def test_cli_runs_command_empty_db():
    """Test runs command with empty database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        result = runner.invoke(app, ["runs", "--db-path", str(db_path)])
        
        # Should succeed but show no runs
        assert result.exit_code == 0
        assert "No clustering runs found" in result.stdout or "runs found" in result.stdout.lower()


@pytest.mark.smoke
def test_cli_list_command_empty_db():
    """Test list command with empty database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        result = runner.invoke(app, ["list", "--db-path", str(db_path), "--limit", "10"])
        
        # Should succeed but show no queries
        assert result.exit_code == 0


@pytest.mark.smoke
def test_cli_chroma_info():
    """Test chroma info command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_path = Path(tmpdir) / "chroma"
        
        result = runner.invoke(app, ["chroma", "info", "--chroma-path", str(chroma_path)])
        
        # Should succeed (might be empty)
        assert result.exit_code == 0

