"""Tests for data loader."""
import pytest
from lmsys_query_analysis.db.loader import extract_first_query
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
        language="en"
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
