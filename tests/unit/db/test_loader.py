"""Tests for data loader."""

from unittest.mock import Mock, patch

import pytest
from sqlmodel import select

from lmsys_query_analysis.db.adapters import extract_first_query
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.loader import load_dataset
from lmsys_query_analysis.db.models import Query


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
    # Create mock normalized records (as returned by HuggingFaceAdapter)
    mock_normalized_data = [
        {
            "conversation_id": "conv1",
            "query_text": "What is AI?",
            "model": "gpt-4",
            "language": "English",
            "timestamp": None,
            "extra_metadata": {"turn_count": 2, "redacted": False},
        },
        {
            "conversation_id": "conv2",
            "query_text": "Hola mundo",
            "model": "claude-3",
            "language": "Spanish",
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=None, skip_existing=True, apply_pragmas=False)

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

    # Mock normalized records with one existing, one new
    mock_normalized_data = [
        {
            "conversation_id": "conv1",  # Existing
            "query_text": "What is AI?",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
        {
            "conversation_id": "conv2",  # New
            "query_text": "Hello",
            "model": "claude-3",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=None, skip_existing=True, apply_pragmas=False)

    assert stats["total_processed"] == 2
    assert stats["loaded"] == 1  # Only conv2 loaded
    assert stats["skipped"] == 1  # conv1 skipped

    # Verify only 2 queries total (1 existing + 1 new)
    with temp_db.get_session() as session:
        count = len(session.exec(select(Query)).all())
        assert count == 2


def test_load_lmsys_dataset_handles_errors(temp_db):
    """Test that loader handles various error conditions.

    Note: The adapter now handles error filtering, so this test
    only receives valid normalized records. The adapter's error
    handling is tested in test_adapters.py.
    """
    # Mock normalized data - adapter only yields valid records
    mock_normalized_data = [
        {
            "conversation_id": "conv4",
            "query_text": "Valid query",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=None, apply_pragmas=False)

    assert stats["total_processed"] == 1
    assert stats["loaded"] == 1  # Only conv4 loaded
    assert stats["errors"] == 0  # Errors filtered by adapter


def test_load_lmsys_dataset_with_limit(temp_db):
    """Test loading with a limit.

    Note: The adapter handles the limit internally, so we just mock
    it returning the limited number of records.
    """
    # Mock normalized data with limit applied (adapter handles this)
    mock_normalized_data = [
        {
            "conversation_id": f"conv{i}",
            "query_text": f"Query {i}",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        }
        for i in range(5)
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=5, apply_pragmas=False)

    # Should process exactly 5
    assert stats["total_processed"] == 5
    assert stats["loaded"] == 5


def test_load_lmsys_dataset_deduplicates_within_batch(temp_db):
    """Test that duplicate conversation IDs within a batch are handled."""
    # Mock normalized data with duplicates
    mock_normalized_data = [
        {
            "conversation_id": "dup",
            "query_text": "First",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
        {
            "conversation_id": "dup",  # Duplicate in same batch
            "query_text": "Second",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
        {
            "conversation_id": "unique",
            "query_text": "Unique",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=None, apply_pragmas=False)

    assert stats["total_processed"] == 3
    assert stats["loaded"] == 2  # Only first "dup" and "unique"
    assert stats["skipped"] == 1  # Second "dup" skipped


def test_load_lmsys_dataset_handles_json_conversation(temp_db):
    """Test that JSON string conversations are parsed correctly.

    Note: The adapter now handles JSON parsing, so this test
    verifies the loader works with adapter's output.
    """
    # Mock normalized data (adapter already parsed JSON)
    mock_normalized_data = [
        {
            "conversation_id": "conv1",
            "query_text": "Parsed from JSON",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 2, "redacted": False},
        },
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=None, apply_pragmas=False)

    assert stats["loaded"] == 1

    # Verify the query text was extracted correctly
    with temp_db.get_session() as session:
        query = session.exec(select(Query)).first()
        assert query.query_text == "Parsed from JSON"


def test_load_lmsys_dataset_stores_metadata(temp_db):
    """Test that extra metadata is stored correctly."""
    # Mock normalized data with metadata
    mock_normalized_data = [
        {
            "conversation_id": "conv1",
            "query_text": "Test query",
            "model": "gpt-4",
            "language": "English",
            "timestamp": None,
            "extra_metadata": {
                "turn_count": 2,
                "redacted": True,
                "openai_moderation": {"flagged": False},
            },
        },
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=None, apply_pragmas=False)

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

    # Mock normalized data
    mock_normalized_data = [
        {
            "conversation_id": "conv1",
            "query_text": "Test with chroma",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    # Mock ChromaDB
    mock_chroma = Mock(spec=ChromaManager)
    mock_chroma.add_queries_batch = Mock()

    # Mock EmbeddingGenerator - it's imported inside the function
    with (
        patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter),
        patch("lmsys_query_analysis.clustering.embeddings.EmbeddingGenerator") as mock_emb_gen,
    ):
        # Mock embeddings
        mock_embedder = Mock()
        mock_embedder.generate_embeddings = Mock(return_value=[[0.1] * 10])
        mock_emb_gen.return_value = mock_embedder

        stats = load_dataset(
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
    # Create 100 mock normalized records
    mock_normalized_data = [
        {
            "conversation_id": f"conv{i}",
            "query_text": f"Query {i}",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        }
        for i in range(100)
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=None, batch_size=20, apply_pragmas=False)

    assert stats["total_processed"] == 100
    assert stats["loaded"] == 100

    # Verify all queries are in database
    with temp_db.get_session() as session:
        count = len(session.exec(select(Query)).all())
        assert count == 100


def test_load_lmsys_dataset_missing_language(temp_db):
    """Test that missing language field is handled correctly."""
    # Mock normalized data with None language
    mock_normalized_data = [
        {
            "conversation_id": "conv1",
            "query_text": "Test",
            "model": "gpt-4",
            "language": None,  # No language field
            "timestamp": None,
            "extra_metadata": {"turn_count": 1, "redacted": False},
        },
    ]

    mock_adapter = Mock()
    mock_adapter.__iter__ = Mock(return_value=iter(mock_normalized_data))
    mock_adapter.__len__ = Mock(return_value=len(mock_normalized_data))

    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter", return_value=mock_adapter):
        stats = load_dataset(db=temp_db, limit=None, apply_pragmas=False)

    assert stats["loaded"] == 1

    # Verify language is None in database
    with temp_db.get_session() as session:
        query = session.exec(select(Query)).first()
        assert query.language is None


def test_load_with_custom_dataset_name(temp_db):
    """Test loader accepts custom dataset name and passes it to adapter."""
    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter") as mock_adapter_class:
        # Mock the adapter instance
        mock_adapter = Mock()
        mock_adapter.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "conversation_id": "test1",
                        "query_text": "What is Python?",
                        "model": "gpt-4",
                        "language": "en",
                        "timestamp": None,
                        "extra_metadata": {},
                    }
                ]
            )
        )
        mock_adapter.__len__ = Mock(return_value=1)
        mock_adapter_class.return_value = mock_adapter

        # Call load_dataset with custom dataset name
        stats = load_dataset(
            db=temp_db,
            dataset_name="custom/dataset",
            limit=10,
            apply_pragmas=False,
        )

        # Verify adapter was initialized with custom dataset
        mock_adapter_class.assert_called_once()
        call_kwargs = mock_adapter_class.call_args[1]
        assert call_kwargs["dataset_name"] == "custom/dataset"
        assert call_kwargs["split"] == "train"
        assert call_kwargs["limit"] == 10

        # Verify data was loaded
        assert stats["loaded"] == 1


def test_load_defaults_to_lmsys_dataset(temp_db):
    """Test loader defaults to lmsys/lmsys-chat-1m when no dataset specified."""
    with patch("lmsys_query_analysis.db.loader.HuggingFaceAdapter") as mock_adapter_class:
        # Mock the adapter instance
        mock_adapter = Mock()
        mock_adapter.__iter__ = Mock(return_value=iter([]))
        mock_adapter.__len__ = Mock(return_value=0)
        mock_adapter_class.return_value = mock_adapter

        # Call load_dataset without dataset_name parameter
        stats = load_dataset(
            db=temp_db,
            limit=10,
            apply_pragmas=False,
        )

        # Verify adapter was initialized with default dataset
        mock_adapter_class.assert_called_once()
        call_kwargs = mock_adapter_class.call_args[1]
        assert call_kwargs["dataset_name"] == "lmsys/lmsys-chat-1m"
