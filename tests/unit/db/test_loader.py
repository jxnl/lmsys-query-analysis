"""Tests for data loader."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from lmsys_query_analysis.db.loader import extract_first_query, load_lmsys_dataset
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import Query
from sqlmodel import select


def test_extract_first_query_basic():
    """Test extracting first query from conversation."""
    conversation = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more"},
    ]

    result = extract_first_query(conversation)
    assert result == "What is Python?"


def test_extract_first_query_whitespace():
    """Test that whitespace is stripped."""
    conversation = [
        {"role": "user", "content": "  What is Python?  \n"},
    ]

    result = extract_first_query(conversation)
    assert result == "What is Python?"


def test_extract_first_query_no_user():
    """Test handling conversation with no user messages."""
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": "Hello!"},
    ]

    result = extract_first_query(conversation)
    assert result is None


def test_extract_first_query_empty():
    """Test handling empty conversation."""
    assert extract_first_query([]) is None
    assert extract_first_query(None) is None


def test_extract_first_query_empty_content():
    """Test handling user message with empty content."""
    conversation = [
        {"role": "user", "content": ""},
        {"role": "user", "content": "Second message"},
    ]

    result = extract_first_query(conversation)
    assert result == ""  # First user message even if empty


def test_extract_first_query_missing_content():
    """Test handling message without content field."""
    conversation = [
        {"role": "user"},
        {"role": "user", "content": "Second message"},
    ]

    result = extract_first_query(conversation)
    assert result == ""  # get() returns empty string


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.create_tables()
    return db


def test_database_creation(temp_db):
    """Test that database and tables are created."""
    assert temp_db.db_path.exists()

    # Test we can create a session
    session = temp_db.get_session()
    assert session is not None
    session.close()


def test_add_query_to_db(temp_db):
    """Test adding a query to the database."""
    session = temp_db.get_session()

    query = Query(
        conversation_id="test-789",
        model="claude-3",
        query_text="How does clustering work?",
        language="en",
    )
    session.add(query)
    session.commit()

    # Verify it was added
    statement = select(Query).where(Query.conversation_id == "test-789")
    result = session.exec(statement).first()

    assert result is not None
    assert result.query_text == "How does clustering work?"
    assert result.model == "claude-3"

    session.close()


def test_skip_duplicate_conversation_id(temp_db):
    """Test that duplicate conversation_ids are handled."""
    session = temp_db.get_session()

    query1 = Query(
        conversation_id="dup-123",
        model="gpt-4",
        query_text="First",
    )
    session.add(query1)
    session.commit()

    # Check for existing
    statement = select(Query).where(Query.conversation_id == "dup-123")
    existing = session.exec(statement).first()
    assert existing is not None

    session.close()


def test_load_lmsys_dataset_basic(temp_db):
    """Test basic dataset loading with mocked data."""
    # Create mock dataset
    mock_data = [
        {
            "conversation_id": "conv1",
            "model": "gpt-4",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."},
            ],
        },
        {
            "conversation_id": "conv2",
            "model": "claude-3",
            "language": "Spanish",
            "conversation": [
                {"role": "user", "content": "Hola mundo"},
            ],
        },
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=None, skip_existing=True, apply_pragmas=False)
    
    assert stats["total_processed"] == 2
    assert stats["loaded"] == 2
    assert stats["skipped"] == 0
    assert stats["errors"] == 0
    
    # Verify data in database
    with temp_db.get_session() as session:
        queries = session.exec(select(Query)).all()
        assert len(queries) == 2
        assert queries[0].query_text in ["What is AI?", "Hola mundo"]


def test_load_lmsys_dataset_skip_existing(temp_db):
    """Test that existing conversations are skipped."""
    # Add existing query
    with temp_db.get_session() as session:
        existing = Query(
            conversation_id="conv1",
            model="gpt-4",
            query_text="Existing query",
        )
        session.add(existing)
        session.commit()
    
    # Mock dataset with one existing, one new
    mock_data = [
        {
            "conversation_id": "conv1",  # Existing
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": "What is AI?"}],
        },
        {
            "conversation_id": "conv2",  # New
            "model": "claude-3",
            "conversation": [{"role": "user", "content": "Hello"}],
        },
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=None, skip_existing=True, apply_pragmas=False)
    
    assert stats["total_processed"] == 2
    assert stats["loaded"] == 1  # Only conv2 loaded
    assert stats["skipped"] == 1  # conv1 skipped
    
    # Verify only 2 queries total (1 existing + 1 new)
    with temp_db.get_session() as session:
        count = len(session.exec(select(Query)).all())
        assert count == 2


def test_load_lmsys_dataset_handles_errors(temp_db):
    """Test that loader handles various error conditions."""
    mock_data = [
        # Missing conversation_id
        {
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": "Test"}],
        },
        # Invalid JSON in conversation
        {
            "conversation_id": "conv2",
            "model": "gpt-4",
            "conversation": "invalid json string {",
        },
        # No user messages
        {
            "conversation_id": "conv3",
            "model": "gpt-4",
            "conversation": [{"role": "system", "content": "System msg"}],
        },
        # Valid one
        {
            "conversation_id": "conv4",
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": "Valid query"}],
        },
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=None, apply_pragmas=False)
    
    assert stats["total_processed"] == 4
    assert stats["loaded"] == 1  # Only conv4 loaded
    assert stats["errors"] == 3  # Three errors


def test_load_lmsys_dataset_with_limit(temp_db):
    """Test loading with a limit."""
    mock_data = [
        {
            "conversation_id": f"conv{i}",
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": f"Query {i}"}],
        }
        for i in range(10)
    ]
    
    # Create limited dataset
    limited_data = mock_data[:5]
    mock_dataset_limited = Mock()
    mock_dataset_limited.__iter__ = Mock(return_value=iter(limited_data))
    mock_dataset_limited.__len__ = Mock(return_value=len(limited_data))
    
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    mock_dataset.select = Mock(return_value=mock_dataset_limited)
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=5, apply_pragmas=False)
    
    # Should process exactly 5
    assert stats["total_processed"] == 5
    assert stats["loaded"] == 5


def test_load_lmsys_dataset_deduplicates_within_batch(temp_db):
    """Test that duplicate conversation IDs within a batch are handled."""
    mock_data = [
        {
            "conversation_id": "dup",
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": "First"}],
        },
        {
            "conversation_id": "dup",  # Duplicate in same batch
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": "Second"}],
        },
        {
            "conversation_id": "unique",
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": "Unique"}],
        },
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=None, apply_pragmas=False)
    
    assert stats["total_processed"] == 3
    assert stats["loaded"] == 2  # Only first "dup" and "unique"
    assert stats["skipped"] == 1  # Second "dup" skipped


def test_load_lmsys_dataset_handles_json_conversation(temp_db):
    """Test that JSON string conversations are parsed correctly."""
    import json
    
    mock_data = [
        {
            "conversation_id": "conv1",
            "model": "gpt-4",
            # Conversation as JSON string (as it might come from dataset)
            "conversation": json.dumps([
                {"role": "user", "content": "Parsed from JSON"},
                {"role": "assistant", "content": "Response"},
            ]),
        },
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=None, apply_pragmas=False)
    
    assert stats["loaded"] == 1
    
    # Verify the query text was extracted correctly
    with temp_db.get_session() as session:
        query = session.exec(select(Query)).first()
        assert query.query_text == "Parsed from JSON"


def test_load_lmsys_dataset_stores_metadata(temp_db):
    """Test that extra metadata is stored correctly."""
    mock_data = [
        {
            "conversation_id": "conv1",
            "model": "gpt-4",
            "language": "English",
            "redacted": True,
            "openai_moderation": {"flagged": False},
            "conversation": [
                {"role": "user", "content": "Test query"},
                {"role": "assistant", "content": "Response"},
            ],
        },
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=None, apply_pragmas=False)
    
    assert stats["loaded"] == 1
    
    # Verify metadata
    with temp_db.get_session() as session:
        query = session.exec(select(Query)).first()
        assert query.model == "gpt-4"
        assert query.language == "English"
        assert query.extra_metadata["turn_count"] == 2
        assert query.extra_metadata["redacted"] is True
        assert query.extra_metadata["openai_moderation"]["flagged"] is False


def test_load_lmsys_dataset_with_chroma(temp_db):
    """Test loading with ChromaDB integration."""
    from lmsys_query_analysis.db.chroma import ChromaManager
    
    mock_data = [
        {
            "conversation_id": "conv1",
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": "Test with chroma"}],
        },
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    # Mock ChromaDB
    mock_chroma = Mock(spec=ChromaManager)
    mock_chroma.add_queries_batch = Mock()
    
    # Mock EmbeddingGenerator - it's imported inside the function
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset), \
         patch('lmsys_query_analysis.clustering.embeddings.EmbeddingGenerator') as mock_emb_gen:
        
        # Mock embeddings
        mock_embedder = Mock()
        mock_embedder.generate_embeddings = Mock(return_value=[[0.1] * 10])
        mock_emb_gen.return_value = mock_embedder
        
        stats = load_lmsys_dataset(
            db=temp_db,
            limit=None,
            chroma=mock_chroma,
            embedding_model="test-model",
            embedding_provider="test-provider",
            apply_pragmas=False,
        )
    
    assert stats["loaded"] == 1
    
    # Verify embeddings were generated and stored
    mock_emb_gen.assert_called_once()
    mock_embedder.generate_embeddings.assert_called_once()
    mock_chroma.add_queries_batch.assert_called_once()


def test_load_lmsys_dataset_large_batch(temp_db):
    """Test loading a large batch of queries."""
    # Create 100 mock queries
    mock_data = [
        {
            "conversation_id": f"conv{i}",
            "model": "gpt-4",
            "conversation": [{"role": "user", "content": f"Query {i}"}],
        }
        for i in range(100)
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=None, batch_size=20, apply_pragmas=False)
    
    assert stats["total_processed"] == 100
    assert stats["loaded"] == 100
    
    # Verify all queries are in database
    with temp_db.get_session() as session:
        count = len(session.exec(select(Query)).all())
        assert count == 100


def test_load_lmsys_dataset_missing_language(temp_db):
    """Test that missing language field is handled correctly."""
    mock_data = [
        {
            "conversation_id": "conv1",
            "model": "gpt-4",
            # No language field
            "conversation": [{"role": "user", "content": "Test"}],
        },
    ]
    
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=len(mock_data))
    
    with patch('lmsys_query_analysis.db.loader.load_dataset', return_value=mock_dataset):
        stats = load_lmsys_dataset(db=temp_db, limit=None, apply_pragmas=False)
    
    assert stats["loaded"] == 1
    
    # Verify language is None in database
    with temp_db.get_session() as session:
        query = session.exec(select(Query)).first()
        assert query.language is None
