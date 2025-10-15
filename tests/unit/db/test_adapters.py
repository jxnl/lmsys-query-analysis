"""Tests for data source adapters."""

from typing import Iterator
from datetime import datetime
from lmsys_query_analysis.db.adapters import (
    DataSourceAdapter,
    extract_first_query,
)


# ============================================================================
# Mock Adapter for Testing
# ============================================================================


class MockDataSourceAdapter:
    """Mock adapter implementation for testing the protocol interface."""
    
    _NOT_PROVIDED = object()  # Sentinel value
    
    def __init__(self, data: list[dict] | None = None, length=_NOT_PROVIDED):
        """Initialize mock adapter with test data.
        
        Args:
            data: List of normalized records to yield. If None, uses empty list.
            length: Override for __len__. If not provided, returns len(data).
                   If explicitly None, returns None (for streaming sources).
        """
        self._data = data or []
        if length is self._NOT_PROVIDED:
            self._length = len(self._data)
        else:
            self._length = length
    
    def __iter__(self) -> Iterator[dict]:
        """Yield normalized records."""
        return iter(self._data)
    
    def __len__(self) -> int | None:
        """Return record count or None."""
        return self._length


# ============================================================================
# Tests for extract_first_query Helper
# ============================================================================


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


def test_extract_first_query_multiple_users():
    """Test that only first user message is extracted."""
    conversation = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "First user query"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Second user query"},
    ]

    result = extract_first_query(conversation)
    assert result == "First user query"


# ============================================================================
# Tests for DataSourceAdapter Protocol
# ============================================================================


def test_adapter_protocol_interface():
    """Test that MockDataSourceAdapter conforms to DataSourceAdapter protocol."""
    adapter = MockDataSourceAdapter()
    
    # Check that it's recognized as implementing the protocol
    assert isinstance(adapter, DataSourceAdapter)


def test_mock_adapter_iteration():
    """Test iterating over mock adapter."""
    test_data = [
        {
            "conversation_id": "conv1",
            "query_text": "What is AI?",
            "model": "gpt-4",
            "language": "en",
            "timestamp": None,
            "extra_metadata": {},
        },
        {
            "conversation_id": "conv2",
            "query_text": "Hola mundo",
            "model": "claude-3",
            "language": "es",
            "timestamp": None,
            "extra_metadata": {},
        },
    ]
    
    adapter = MockDataSourceAdapter(data=test_data)
    
    # Collect all records
    records = list(adapter)
    
    assert len(records) == 2
    assert records[0]["conversation_id"] == "conv1"
    assert records[0]["query_text"] == "What is AI?"
    assert records[1]["conversation_id"] == "conv2"
    assert records[1]["query_text"] == "Hola mundo"


def test_mock_adapter_length():
    """Test that mock adapter returns correct length."""
    test_data = [
        {
            "conversation_id": f"conv{i}",
            "query_text": f"Query {i}",
            "model": "gpt-4",
            "language": "en",
            "timestamp": None,
            "extra_metadata": {},
        }
        for i in range(5)
    ]
    
    adapter = MockDataSourceAdapter(data=test_data)
    assert len(adapter) == 5


def test_mock_adapter_length_none():
    """Test that mock adapter can return None for streaming sources."""
    test_data = [{"conversation_id": "conv1", "query_text": "Test", "model": "gpt-4", 
                  "language": None, "timestamp": None, "extra_metadata": {}}]
    
    adapter = MockDataSourceAdapter(data=test_data, length=None)
    # Call __len__ directly since Python's len() requires an int
    assert adapter.__len__() is None


def test_mock_adapter_empty():
    """Test mock adapter with no data."""
    adapter = MockDataSourceAdapter()
    
    records = list(adapter)
    assert len(records) == 0
    assert len(adapter) == 0


def test_adapter_normalized_output_schema():
    """Test that adapter output conforms to expected schema."""
    now = datetime.now()
    test_data = [
        {
            "conversation_id": "test123",
            "query_text": "What is machine learning?",
            "model": "gpt-4",
            "language": "en",
            "timestamp": now,
            "extra_metadata": {
                "turn_count": 2,
                "redacted": False,
            },
        },
    ]
    
    adapter = MockDataSourceAdapter(data=test_data)
    record = next(iter(adapter))
    
    # Verify all required fields are present
    assert "conversation_id" in record
    assert "query_text" in record
    assert "model" in record
    assert "language" in record
    assert "timestamp" in record
    assert "extra_metadata" in record
    
    # Verify types
    assert isinstance(record["conversation_id"], str)
    assert isinstance(record["query_text"], str)
    assert isinstance(record["model"], str)
    assert record["language"] is None or isinstance(record["language"], str)
    assert record["timestamp"] is None or isinstance(record["timestamp"], datetime)
    assert isinstance(record["extra_metadata"], dict)


def test_adapter_normalized_output_schema_optional_fields():
    """Test that optional fields can be None."""
    test_data = [
        {
            "conversation_id": "test123",
            "query_text": "Test query",
            "model": "gpt-4",
            "language": None,  # Optional
            "timestamp": None,  # Optional
            "extra_metadata": {},
        },
    ]
    
    adapter = MockDataSourceAdapter(data=test_data)
    record = next(iter(adapter))
    
    # These should be allowed to be None
    assert record["language"] is None
    assert record["timestamp"] is None
    
    # These must be present
    assert isinstance(record["conversation_id"], str)
    assert isinstance(record["query_text"], str)
    assert isinstance(record["model"], str)
    assert isinstance(record["extra_metadata"], dict)


def test_adapter_multiple_iterations():
    """Test that adapter can be iterated multiple times."""
    test_data = [
        {
            "conversation_id": "conv1",
            "query_text": "Query 1",
            "model": "gpt-4",
            "language": "en",
            "timestamp": None,
            "extra_metadata": {},
        },
    ]
    
    adapter = MockDataSourceAdapter(data=test_data)
    
    # First iteration
    records1 = list(adapter)
    assert len(records1) == 1
    
    # Second iteration
    records2 = list(adapter)
    assert len(records2) == 1
    
    # Both should yield same data
    assert records1[0]["conversation_id"] == records2[0]["conversation_id"]

