"""Unit tests for data CLI commands."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@patch('lmsys_query_analysis.cli.commands.data.load_lmsys_dataset')
@patch('lmsys_query_analysis.cli.commands.data.get_db')
@patch('lmsys_query_analysis.cli.commands.data.create_chroma_client')
@patch('lmsys_query_analysis.cli.commands.data.parse_embedding_model')
def test_load_command_basic(mock_parse, mock_chroma_client, mock_get_db, mock_load_dataset):
    """Test basic load command without Chroma."""
    from lmsys_query_analysis.cli.commands.data import load
    
    # Setup mocks
    mock_parse.return_value = ("test-model", "test-provider")
    mock_db = Mock()
    mock_db.db_path = Path("/tmp/test.db")
    mock_get_db.return_value = mock_db
    mock_load_dataset.return_value = {
        "total_processed": 100,
        "loaded": 90,
        "skipped": 10,
        "errors": 0
    }
    
    # Execute command
    load(
        limit=100,
        db_path="/tmp/test.db",
        use_chroma=False,
        chroma_path="/tmp/chroma",
        embedding_model="test-model",
        db_batch_size=1000,
        streaming=False,
        no_pragmas=False,
        force_reload=False
    )
    
    # Verify
    mock_get_db.assert_called_once_with("/tmp/test.db")
    mock_load_dataset.assert_called_once()


@patch('lmsys_query_analysis.cli.commands.data.Path')
@patch('lmsys_query_analysis.cli.commands.data.shutil.rmtree')
@patch('lmsys_query_analysis.cli.commands.data.get_chroma')
@patch('lmsys_query_analysis.cli.commands.data.get_db')
def test_clear_command_with_yes_flag(mock_get_db, mock_get_chroma, mock_rmtree, mock_path_class):
    """Test clear command with --yes flag."""
    from lmsys_query_analysis.cli.commands.data import clear
    
    # Setup mocks
    mock_db = Mock()
    mock_db.db_path = "/tmp/test.db"
    mock_get_db.return_value = mock_db
    
    mock_chroma = Mock()
    mock_chroma.persist_directory = "/tmp/chroma"
    mock_get_chroma.return_value = mock_chroma
    
    # Mock Path to say files exist
    mock_db_path = Mock()
    mock_db_path.exists.return_value = True
    mock_chroma_path = Mock()
    mock_chroma_path.exists.return_value = True
    
    def path_side_effect(arg):
        if "test.db" in str(arg):
            return mock_db_path
        return mock_chroma_path
    
    mock_path_class.side_effect = path_side_effect
    
    # Execute command with confirm=True (--yes flag)
    clear(
        db_path="/tmp/test.db",
        chroma_path="/tmp/chroma",
        confirm=True
    )
    
    # Verify files were deleted
    mock_db_path.unlink.assert_called_once()
    mock_rmtree.assert_called_once()


@patch('lmsys_query_analysis.cli.commands.data.create_embedding_generator')
@patch('lmsys_query_analysis.cli.commands.data.create_chroma_client')
@patch('lmsys_query_analysis.cli.commands.data.get_db')
@patch('lmsys_query_analysis.cli.commands.data.parse_embedding_model')
def test_backfill_chroma_command(mock_parse, mock_get_db, mock_chroma_client, mock_embed_gen):
    """Test backfill-chroma command."""
    import numpy as np
    from lmsys_query_analysis.cli.commands.data import backfill_chroma
    from lmsys_query_analysis.db.models import Query
    
    # Setup mocks
    mock_parse.return_value = ("test-model", "test-provider")
    
    mock_db = Mock()
    mock_session = Mock()
    mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_db.get_session.return_value.__exit__ = Mock(return_value=False)
    
    # Mock queries and count
    mock_queries = [
        Query(id=1, conversation_id="c1", model="gpt-4", query_text="Test 1", language="en"),
    ]
    mock_result_all = Mock()
    mock_result_all.all.return_value = mock_queries
    
    # Mock the one() result for count query - returns a single row
    mock_result_one = Mock()
    mock_result_one.one.return_value = (1,)  # Tuple with count value
    
    # Set up exec to return different results for different queries
    mock_session.exec.side_effect = [mock_result_one, mock_result_all]
    
    mock_get_db.return_value = mock_db
    
    # Mock Chroma
    mock_chroma = Mock()
    mock_chroma.get_query_embeddings_map.return_value = {}
    mock_chroma_client.return_value = mock_chroma
    
    # Mock embedder
    mock_embedder = Mock()
    mock_embedder.generate_embeddings.return_value = np.array([[0.1] * 10])
    mock_embed_gen.return_value = mock_embedder
    
    # Execute command
    backfill_chroma(
        db_path="/tmp/test.db",
        chroma_path="/tmp/chroma",
        embedding_model="test-model",
        chunk_size=5000,
        embed_batch_size=64
    )
    
    # Verify
    mock_get_db.assert_called_once()
    mock_chroma_client.assert_called_once()
