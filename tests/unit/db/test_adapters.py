"""Tests for data source adapters."""

import pytest
from datetime import datetime
from typing import Iterator, Optional

from lmsys_query_analysis.db.adapters import BaseAdapter, RecordDict


class MockAdapter:
    """Mock adapter implementation for testing the adapter interface."""
    
    def __init__(self, records: list[RecordDict]):
        """Initialize with a list of pre-defined records.
        
        Args:
            records: List of RecordDict instances to yield
        """
        self.records = records
    
    def iter_records(self, limit: Optional[int] = None) -> Iterator[RecordDict]:
        """Yield records up to the limit."""
        for i, record in enumerate(self.records):
            if limit is not None and i >= limit:
                break
            yield record


@pytest.fixture
def sample_records() -> list[RecordDict]:
    """Sample normalized records for testing."""
    return [
        RecordDict(
            conversation_id="conv1",
            query_text="How do I write a Python function?",
            model="gpt-4",
            language="en",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            extra_metadata={"turn_count": 2, "redacted": False},
        ),
        RecordDict(
            conversation_id="conv2",
            query_text="Explain async/await in Python",
            model="gpt-4",
            language="en",
            timestamp=datetime(2024, 1, 2, 12, 0, 0),
            extra_metadata={"turn_count": 4, "redacted": False},
        ),
        RecordDict(
            conversation_id="conv3",
            query_text="What is machine learning?",
            model="claude-3",
            language="en",
            timestamp=datetime(2024, 1, 3, 12, 0, 0),
            extra_metadata={"turn_count": 2, "redacted": False},
        ),
    ]


@pytest.fixture
def mock_adapter(sample_records: list[RecordDict]) -> MockAdapter:
    """Create a mock adapter with sample records."""
    return MockAdapter(sample_records)


def test_record_dict_structure():
    """Test that RecordDict has the expected structure."""
    # Create a minimal valid record
    record: RecordDict = RecordDict(
        conversation_id="test123",
        query_text="Test query",
    )
    
    assert record["conversation_id"] == "test123"
    assert record["query_text"] == "Test query"


def test_record_dict_with_all_fields():
    """Test RecordDict with all optional fields."""
    record: RecordDict = RecordDict(
        conversation_id="test123",
        query_text="Test query",
        model="gpt-4",
        language="en",
        timestamp=datetime(2024, 1, 1),
        extra_metadata={"key": "value"},
    )
    
    assert record["conversation_id"] == "test123"
    assert record["query_text"] == "Test query"
    assert record["model"] == "gpt-4"
    assert record["language"] == "en"
    assert record["timestamp"] == datetime(2024, 1, 1)
    assert record["extra_metadata"] == {"key": "value"}


def test_mock_adapter_implements_protocol(mock_adapter: MockAdapter):
    """Test that MockAdapter conforms to BaseAdapter protocol."""
    # This should type-check and work at runtime
    adapter: BaseAdapter = mock_adapter
    
    # Should be able to call iter_records
    records = list(adapter.iter_records())
    assert len(records) == 3


def test_adapter_iter_records_no_limit(mock_adapter: MockAdapter):
    """Test adapter returns all records when no limit specified."""
    records = list(mock_adapter.iter_records())
    
    assert len(records) == 3
    assert records[0]["conversation_id"] == "conv1"
    assert records[1]["conversation_id"] == "conv2"
    assert records[2]["conversation_id"] == "conv3"


def test_adapter_iter_records_with_limit(mock_adapter: MockAdapter):
    """Test adapter respects limit parameter."""
    records = list(mock_adapter.iter_records(limit=2))
    
    assert len(records) == 2
    assert records[0]["conversation_id"] == "conv1"
    assert records[1]["conversation_id"] == "conv2"


def test_adapter_iter_records_limit_exceeds_total(mock_adapter: MockAdapter):
    """Test adapter when limit exceeds total records."""
    records = list(mock_adapter.iter_records(limit=10))
    
    # Should return all 3 records, not error
    assert len(records) == 3


def test_adapter_iter_records_zero_limit(mock_adapter: MockAdapter):
    """Test adapter with zero limit returns nothing."""
    records = list(mock_adapter.iter_records(limit=0))
    
    assert len(records) == 0


def test_adapter_iter_records_is_iterator(mock_adapter: MockAdapter):
    """Test that iter_records returns an iterator, not a list."""
    result = mock_adapter.iter_records()
    
    # Should be an iterator/generator, not a list
    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')
    
    # Consuming it should work
    first = next(result)
    assert first["conversation_id"] == "conv1"
    
    second = next(result)
    assert second["conversation_id"] == "conv2"


def test_adapter_multiple_iterations(mock_adapter: MockAdapter):
    """Test that adapter can be iterated multiple times."""
    # First iteration
    records1 = list(mock_adapter.iter_records())
    assert len(records1) == 3
    
    # Second iteration should work
    records2 = list(mock_adapter.iter_records())
    assert len(records2) == 3
    
    # Results should be the same
    assert records1[0]["conversation_id"] == records2[0]["conversation_id"]


def test_empty_adapter():
    """Test adapter with no records."""
    empty_adapter = MockAdapter([])
    
    records = list(empty_adapter.iter_records())
    assert len(records) == 0


def test_adapter_with_minimal_records():
    """Test adapter with records having only required fields."""
    minimal_records = [
        RecordDict(
            conversation_id="min1",
            query_text="Query 1",
        ),
        RecordDict(
            conversation_id="min2",
            query_text="Query 2",
        ),
    ]
    
    adapter = MockAdapter(minimal_records)
    records = list(adapter.iter_records())
    
    assert len(records) == 2
    assert records[0]["conversation_id"] == "min1"
    assert records[0]["query_text"] == "Query 1"
    assert records[1]["conversation_id"] == "min2"
    assert records[1]["query_text"] == "Query 2"

