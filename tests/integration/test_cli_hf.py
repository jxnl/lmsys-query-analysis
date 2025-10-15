"""Integration tests for CLI with Hugging Face dataset loading."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import shutil

from lmsys_query_analysis.cli.main import app
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import Query

runner = CliRunner()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield str(db_path)
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@patch('lmsys_query_analysis.cli.commands.data.HuggingFaceAdapter')
@patch('lmsys_query_analysis.cli.commands.data.load_dataset')
def test_load_with_default_hf_dataset(mock_load_dataset, mock_adapter, temp_db_path):
    """Test load command without --hf flag uses default dataset (backwards compatibility)."""
    # Setup mocks
    mock_adapter_instance = Mock()
    mock_adapter.return_value = mock_adapter_instance
    mock_load_dataset.return_value = {
        "total_processed": 10,
        "loaded": 10,
        "skipped": 0,
        "errors": 0
    }
    
    # Run CLI command without --hf flag
    result = runner.invoke(app, [
        "load",
        "--limit", "10",
        "--db-path", temp_db_path
    ])
    
    # Verify success
    assert result.exit_code == 0, f"Command failed with: {result.stdout}"
    
    # Verify adapter was created with default dataset
    mock_adapter.assert_called_once()
    call_kwargs = mock_adapter.call_args[1]
    assert call_kwargs['dataset_name'] == "lmsys/lmsys-chat-1m"
    assert call_kwargs['use_streaming'] == False
    
    # Verify load_dataset was called
    assert mock_load_dataset.called


@patch('lmsys_query_analysis.cli.commands.data.HuggingFaceAdapter')
@patch('lmsys_query_analysis.cli.commands.data.load_dataset')
def test_load_with_explicit_hf_dataset(mock_load_dataset, mock_adapter, temp_db_path):
    """Test load command with explicit --hf flag."""
    # Setup mocks
    mock_adapter_instance = Mock()
    mock_adapter.return_value = mock_adapter_instance
    mock_load_dataset.return_value = {
        "total_processed": 10,
        "loaded": 10,
        "skipped": 0,
        "errors": 0
    }
    
    # Run CLI command with custom HF dataset
    result = runner.invoke(app, [
        "load",
        "--hf", "custom/test-dataset",
        "--limit", "10",
        "--db-path", temp_db_path
    ])
    
    # Verify success
    assert result.exit_code == 0, f"Command failed with: {result.stdout}"
    
    # Verify adapter was created with custom dataset
    mock_adapter.assert_called_once()
    call_kwargs = mock_adapter.call_args[1]
    assert call_kwargs['dataset_name'] == "custom/test-dataset"
    assert call_kwargs['use_streaming'] == False
    
    # Verify load_dataset was called
    assert mock_load_dataset.called


@patch('lmsys_query_analysis.cli.commands.data.HuggingFaceAdapter')
@patch('lmsys_query_analysis.cli.commands.data.load_dataset')
def test_load_with_hf_and_streaming(mock_load_dataset, mock_adapter, temp_db_path):
    """Test load command with --hf flag and streaming enabled."""
    # Setup mocks
    mock_adapter_instance = Mock()
    mock_adapter.return_value = mock_adapter_instance
    mock_load_dataset.return_value = {
        "total_processed": 10,
        "loaded": 10,
        "skipped": 0,
        "errors": 0
    }
    
    # Run CLI command with streaming
    result = runner.invoke(app, [
        "load",
        "--hf", "lmsys/lmsys-chat-1m",
        "--limit", "10",
        "--streaming",
        "--db-path", temp_db_path
    ])
    
    # Verify success
    assert result.exit_code == 0, f"Command failed with: {result.stdout}"
    
    # Verify adapter was created with streaming enabled
    mock_adapter.assert_called_once()
    call_kwargs = mock_adapter.call_args[1]
    assert call_kwargs['dataset_name'] == "lmsys/lmsys-chat-1m"
    assert call_kwargs['use_streaming'] == True
    
    # Verify load_dataset was called
    assert mock_load_dataset.called


@patch('lmsys_query_analysis.cli.commands.data.HuggingFaceAdapter')
@patch('lmsys_query_analysis.cli.commands.data.load_dataset')
@patch('lmsys_query_analysis.cli.commands.data.create_chroma_client')
def test_load_with_hf_and_chroma(mock_chroma_client, mock_load_dataset, mock_adapter, temp_db_path):
    """Test load command with --hf flag and ChromaDB enabled."""
    # Setup mocks
    mock_adapter_instance = Mock()
    mock_adapter.return_value = mock_adapter_instance
    
    mock_chroma = Mock()
    mock_chroma.count_queries.return_value = 10
    mock_chroma.persist_directory = "/tmp/chroma"
    mock_chroma_client.return_value = mock_chroma
    
    mock_load_dataset.return_value = {
        "total_processed": 10,
        "loaded": 10,
        "skipped": 0,
        "errors": 0
    }
    
    # Run CLI command with ChromaDB
    result = runner.invoke(app, [
        "load",
        "--hf", "lmsys/lmsys-chat-1m",
        "--limit", "10",
        "--use-chroma",
        "--db-path", temp_db_path,
        "--chroma-path", "/tmp/chroma"
    ])
    
    # Verify success
    assert result.exit_code == 0, f"Command failed with: {result.stdout}"
    
    # Verify adapter was created
    mock_adapter.assert_called_once()
    call_kwargs = mock_adapter.call_args[1]
    assert call_kwargs['dataset_name'] == "lmsys/lmsys-chat-1m"
    
    # Verify ChromaDB was created
    assert mock_chroma_client.called
    
    # Verify load_dataset was called with chroma
    assert mock_load_dataset.called
    load_call_kwargs = mock_load_dataset.call_args[1]
    assert load_call_kwargs['chroma'] == mock_chroma

