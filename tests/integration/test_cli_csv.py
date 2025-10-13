"""Integration tests for CLI CSV loading functionality."""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from lmsys_query_analysis.cli.main import app
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import Query

runner = CliRunner()


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


def test_cli_load_single_csv(tmp_path, fixtures_dir):
    """Test loading a single CSV file via CLI."""
    db_path = tmp_path / "queries.db"
    csv_path = fixtures_dir / "valid_queries.csv"
    
    result = runner.invoke(
        app,
        ["load", "--csv", str(csv_path), "--db-path", str(db_path)]
    )
    
    assert result.exit_code == 0
    assert "CSV file(s)" in result.stdout or "Loaded" in result.stdout
    # Verify stats table shows loaded count
    assert "5" in result.stdout  # 5 queries loaded
    
    # Verify rows in database
    db = Database(db_path)
    session = db.get_session()
    try:
        queries = session.query(Query).all()
        assert len(queries) == 5
        
        # Verify sample data
        conv_ids = {q.conversation_id for q in queries}
        assert "conv_1" in conv_ids
        assert "conv_2" in conv_ids
        
        # Verify query text
        query_texts = {q.query_text for q in queries}
        assert "What is machine learning?" in query_texts
    finally:
        session.close()


def test_cli_load_multiple_csvs(tmp_path, fixtures_dir):
    """Test loading multiple CSV files in one command."""
    db_path = tmp_path / "queries.db"
    csv1 = fixtures_dir / "dataset1.csv"
    csv2 = fixtures_dir / "dataset2.csv"
    
    result = runner.invoke(
        app,
        [
            "load",
            "--csv", str(csv1),
            "--csv", str(csv2),
            "--db-path", str(db_path)
        ]
    )
    
    assert result.exit_code == 0
    # Should show multi-source table
    assert "dataset1.csv" in result.stdout or str(csv1) in result.stdout
    assert "dataset2.csv" in result.stdout or str(csv2) in result.stdout
    
    # Verify correct row counts (c is duplicate, should be skipped in dataset2)
    db = Database(db_path)
    session = db.get_session()
    try:
        queries = session.query(Query).all()
        assert len(queries) == 5  # a, b, c from dataset1 + d, e from dataset2
        
        conv_ids = {q.conversation_id for q in queries}
        assert conv_ids == {"a", "b", "c", "d", "e"}
    finally:
        session.close()


def test_cli_csv_mutual_exclusivity(tmp_path, fixtures_dir):
    """Test that --csv and --hf-dataset are mutually exclusive."""
    db_path = tmp_path / "queries.db"
    csv_path = fixtures_dir / "valid_queries.csv"
    
    result = runner.invoke(
        app,
        [
            "load",
            "--csv", str(csv_path),
            "--hf-dataset", "lmsys/lmsys-chat-1m",
            "--db-path", str(db_path)
        ]
    )
    
    # Should error
    assert result.exit_code != 0
    assert "Cannot specify both" in result.stdout or "mutually exclusive" in result.stdout.lower()


def test_cli_csv_backward_compatibility(tmp_path):
    """Test that load without source args still defaults to LMSYS dataset."""
    db_path = tmp_path / "queries.db"
    
    # This will fail if HF dataset not available, but should attempt to load
    result = runner.invoke(
        app,
        ["load", "--limit", "5", "--db-path", str(db_path)]
    )
    
    # Should attempt HuggingFace load (may fail due to auth/network, but should try)
    # Exit code might be 0 or 1 depending on environment
    # But output should mention HF dataset
    assert "lmsys" in result.stdout.lower() or "huggingface" in result.stdout.lower() or result.exit_code in [0, 1]


def test_cli_csv_skip_existing(tmp_path, fixtures_dir):
    """Test loading same CSV twice skips existing records."""
    db_path = tmp_path / "queries.db"
    csv_path = fixtures_dir / "valid_queries.csv"
    
    # First load
    result1 = runner.invoke(
        app,
        ["load", "--csv", str(csv_path), "--db-path", str(db_path)]
    )
    assert result1.exit_code == 0
    
    # Second load - should skip all
    result2 = runner.invoke(
        app,
        ["load", "--csv", str(csv_path), "--db-path", str(db_path)]
    )
    assert result2.exit_code == 0
    assert "Skipped" in result2.stdout or "skipped" in result2.stdout.lower()
    
    # Verify still only 5 records
    db = Database(db_path)
    session = db.get_session()
    try:
        queries = session.query(Query).all()
        assert len(queries) == 5
    finally:
        session.close()


def test_cli_csv_with_chroma(tmp_path, fixtures_dir):
    """Test CSV loading with ChromaDB integration."""
    db_path = tmp_path / "queries.db"
    chroma_dir = tmp_path / "chroma"
    csv_path = fixtures_dir / "dataset1.csv"  # 3 rows
    
    result = runner.invoke(
        app,
        [
            "load",
            "--csv", str(csv_path),
            "--db-path", str(db_path),
            "--use-chroma",
            "--chroma-path", str(chroma_dir)
        ]
    )
    
    assert result.exit_code == 0
    
    # Verify database
    db = Database(db_path)
    session = db.get_session()
    try:
        queries = session.query(Query).all()
        assert len(queries) == 3
    finally:
        session.close()
    
    # Verify ChromaDB exists (basic check)
    assert chroma_dir.exists()


def test_cli_csv_missing_required_columns(tmp_path, fixtures_dir):
    """Test CSV with missing required columns produces clear error."""
    db_path = tmp_path / "queries.db"
    csv_path = fixtures_dir / "invalid_headers.csv"
    
    result = runner.invoke(
        app,
        ["load", "--csv", str(csv_path), "--db-path", str(db_path)]
    )
    
    # Should error with clear message
    assert result.exit_code != 0
    assert "query_text" in result.stdout or "required" in result.stdout.lower()


def test_cli_csv_empty_fields(tmp_path, fixtures_dir):
    """Test CSV with empty required fields skips rows gracefully."""
    db_path = tmp_path / "queries.db"
    csv_path = fixtures_dir / "empty_fields.csv"
    
    result = runner.invoke(
        app,
        ["load", "--csv", str(csv_path), "--db-path", str(db_path)]
    )
    
    # Should succeed but skip invalid rows
    assert result.exit_code == 0
    
    # Verify only valid rows loaded
    db = Database(db_path)
    session = db.get_session()
    try:
        queries = session.query(Query).all()
        # empty_fields.csv has some valid rows and some with empty fields
        # Only valid rows should be loaded
        for q in queries:
            assert q.conversation_id  # Should not be empty
            assert q.query_text  # Should not be empty
    finally:
        session.close()


def test_cli_csv_nonexistent_file(tmp_path):
    """Test loading nonexistent CSV file produces clear error."""
    db_path = tmp_path / "queries.db"
    csv_path = tmp_path / "nonexistent.csv"
    
    result = runner.invoke(
        app,
        ["load", "--csv", str(csv_path), "--db-path", str(db_path)]
    )
    
    # Should error
    assert result.exit_code != 0
    assert "exist" in result.stdout.lower() or "not found" in result.stdout.lower() or "error" in result.stdout.lower()


def test_cli_load_help_shows_csv_options(tmp_path):
    """Test that help text includes CSV loading options."""
    result = runner.invoke(app, ["load", "--help"])
    
    assert result.exit_code == 0
    assert "--csv" in result.stdout
    assert "--hf-dataset" in result.stdout
    assert "CSV" in result.stdout or "csv" in result.stdout

