"""Tests for data loader."""

import pytest
from unittest.mock import Mock, patch
from lmsys_query_analysis.db.loader import load_queries
from lmsys_query_analysis.db.sources import BaseSource, extract_first_query
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import Query
from sqlmodel import select
from typing import Iterator, Any


class MockSource(BaseSource):
    """Mock data source for testing."""
    
    def __init__(self, records: list[dict], label: str = "mock:test"):
        self.records = records
        self.label = label
    
    def validate_source(self) -> None:
        """Mock validation (always passes)."""
        pass
    
    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield mock records."""
        for record in self.records:
            yield record
    
    def get_source_label(self) -> str:
        """Return mock label."""
        return self.label


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


def test_load_queries_basic(temp_db):
    """Test basic query loading with mock source."""
    # Create mock records (already normalized)
    mock_records = [
        {
            "conversation_id": "conv1",
            "query_text": "What is AI?",
            "model": "gpt-4",
            "language": "English",
            "timestamp": None,
            "extra_metadata": {"turn_count": 2},
        },
        {
            "conversation_id": "conv2",
            "query_text": "Hola mundo",
            "model": "claude-3",
            "language": "Spanish",
            "timestamp": None,
            "extra_metadata": {"turn_count": 1},
        },
    ]
    
    source = MockSource(mock_records)
    stats = load_queries(db=temp_db, source=source, skip_existing=True, apply_pragmas=False)
    
    assert stats["source"] == "mock:test"
    assert stats["total_processed"] == 2
    assert stats["loaded"] == 2
    assert stats["skipped"] == 0
    assert stats["errors"] == 0
    
    # Verify data in database
    with temp_db.get_session() as session:
        queries = session.exec(select(Query)).all()
        assert len(queries) == 2
        assert queries[0].query_text in ["What is AI?", "Hola mundo"]


def test_load_queries_skip_existing(temp_db):
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
    
    # Mock records with one existing, one new
    mock_records = [
        {
            "conversation_id": "conv1",  # Existing
            "query_text": "What is AI?",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
        {
            "conversation_id": "conv2",  # New
            "query_text": "Hello",
            "model": "claude-3",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
    ]
    
    source = MockSource(mock_records)
    stats = load_queries(db=temp_db, source=source, skip_existing=True, apply_pragmas=False)
    
    assert stats["total_processed"] == 2
    assert stats["loaded"] == 1  # Only conv2 loaded
    assert stats["skipped"] == 1  # conv1 skipped
    
    # Verify only 2 queries total (1 existing + 1 new)
    with temp_db.get_session() as session:
        count = len(session.exec(select(Query)).all())
        assert count == 2


def test_load_queries_handles_errors(temp_db):
    """Test that loader handles various error conditions."""
    mock_records = [
        # Missing conversation_id
        {
            "query_text": "Test",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
        # Missing query_text
        {
            "conversation_id": "conv2",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
        # Empty query_text
        {
            "conversation_id": "conv3",
            "query_text": "",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
        # Valid one
        {
            "conversation_id": "conv4",
            "query_text": "Valid query",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
    ]
    
    source = MockSource(mock_records)
    stats = load_queries(db=temp_db, source=source, apply_pragmas=False)
    
    assert stats["total_processed"] == 4
    assert stats["loaded"] == 1  # Only conv4 loaded
    assert stats["errors"] == 3  # Three errors


def test_load_queries_deduplicates_within_batch(temp_db):
    """Test that duplicate conversation IDs within a batch are handled."""
    mock_records = [
        {
            "conversation_id": "dup",
            "query_text": "First",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
        {
            "conversation_id": "dup",  # Duplicate in same batch
            "query_text": "Second",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
        {
            "conversation_id": "unique",
            "query_text": "Unique",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
    ]
    
    source = MockSource(mock_records)
    stats = load_queries(db=temp_db, source=source, apply_pragmas=False)
    
    assert stats["total_processed"] == 3
    assert stats["loaded"] == 2  # Only first "dup" and "unique"
    assert stats["skipped"] == 1  # Second "dup" skipped


def test_load_queries_stores_metadata(temp_db):
    """Test that extra metadata is stored correctly."""
    mock_records = [
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
    
    source = MockSource(mock_records)
    stats = load_queries(db=temp_db, source=source, apply_pragmas=False)
    
    assert stats["loaded"] == 1
    
    # Verify metadata
    with temp_db.get_session() as session:
        query = session.exec(select(Query)).first()
        assert query.model == "gpt-4"
        assert query.language == "English"
        assert query.extra_metadata["turn_count"] == 2
        assert query.extra_metadata["redacted"] is True
        assert query.extra_metadata["openai_moderation"]["flagged"] is False


def test_load_queries_with_chroma(temp_db):
    """Test loading with ChromaDB integration."""
    from lmsys_query_analysis.db.chroma import ChromaManager
    
    mock_records = [
        {
            "conversation_id": "conv1",
            "query_text": "Test with chroma",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
    ]
    
    # Mock ChromaDB
    mock_chroma = Mock(spec=ChromaManager)
    mock_chroma.add_queries_batch = Mock()
    
    # Mock EmbeddingGenerator - it's imported inside the function
    with patch('lmsys_query_analysis.clustering.embeddings.EmbeddingGenerator') as mock_emb_gen:
        # Mock embeddings
        mock_embedder = Mock()
        mock_embedder.generate_embeddings = Mock(return_value=[[0.1] * 10])
        mock_emb_gen.return_value = mock_embedder
        
        source = MockSource(mock_records)
        stats = load_queries(
            db=temp_db,
            source=source,
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


def test_load_queries_large_batch(temp_db):
    """Test loading a large batch of queries."""
    # Create 100 mock queries
    mock_records = [
        {
            "conversation_id": f"conv{i}",
            "query_text": f"Query {i}",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        }
        for i in range(100)
    ]
    
    source = MockSource(mock_records)
    stats = load_queries(db=temp_db, source=source, batch_size=20, apply_pragmas=False)
    
    assert stats["total_processed"] == 100
    assert stats["loaded"] == 100
    
    # Verify all queries are in database
    with temp_db.get_session() as session:
        count = len(session.exec(select(Query)).all())
        assert count == 100


def test_load_queries_missing_language(temp_db):
    """Test that missing language field is handled correctly."""
    mock_records = [
        {
            "conversation_id": "conv1",
            "query_text": "Test",
            "model": "gpt-4",
            "language": None,  # No language
            "timestamp": None,
            "extra_metadata": None,
        },
    ]
    
    source = MockSource(mock_records)
    stats = load_queries(db=temp_db, source=source, apply_pragmas=False)
    
    assert stats["loaded"] == 1
    
    # Verify language is None in database
    with temp_db.get_session() as session:
        query = session.exec(select(Query)).first()
        assert query.language is None


def test_load_queries_stats_include_source_label(temp_db):
    """Test that stats include source label."""
    mock_records = [
        {
            "conversation_id": "conv1",
            "query_text": "Test",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": None,
        },
    ]
    
    source = MockSource(mock_records, label="test:custom-label")
    stats = load_queries(db=temp_db, source=source, apply_pragmas=False)
    
    assert stats["source"] == "test:custom-label"
    assert stats["loaded"] == 1


def test_load_queries_with_mock_base_source(temp_db):
    """Test load_queries with a pure mock BaseSource."""
    # Test that load_queries works with any BaseSource implementation
    mock_records = [
        {
            "conversation_id": "mock1",
            "query_text": "From mock source",
            "model": "test-model",
            "language": "en",
            "timestamp": None,
            "extra_metadata": {"custom": "data"},
        },
    ]
    
    source = MockSource(mock_records, label="mock:base-source")
    stats = load_queries(db=temp_db, source=source, apply_pragmas=False)
    
    assert stats["source"] == "mock:base-source"
    assert stats["total_processed"] == 1
    assert stats["loaded"] == 1
    
    # Verify record in database
    with temp_db.get_session() as session:
        query = session.exec(select(Query)).first()
        assert query.conversation_id == "mock1"
        assert query.query_text == "From mock source"
        assert query.model == "test-model"
        assert query.extra_metadata["custom"] == "data"
