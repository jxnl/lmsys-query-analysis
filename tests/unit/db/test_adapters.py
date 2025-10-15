"""Tests for data source adapters."""

import pytest
from datetime import datetime
from typing import Iterator, Optional
from unittest.mock import Mock, MagicMock

from lmsys_query_analysis.db.adapters import (
    BaseAdapter,
    RecordDict,
    HuggingFaceAdapter,
    extract_first_query,
)


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


# Tests for extract_first_query helper


def test_extract_first_query_basic():
    """Test extracting first user query from conversation."""
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]
    
    query = extract_first_query(conversation)
    assert query == "Hello, how are you?"


def test_extract_first_query_multiple_user_turns():
    """Test that it extracts only the FIRST user query."""
    conversation = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]
    
    query = extract_first_query(conversation)
    assert query == "First question"


def test_extract_first_query_no_user_message():
    """Test with conversation that has no user messages."""
    conversation = [
        {"role": "assistant", "content": "Hello"},
        {"role": "system", "content": "System message"},
    ]
    
    query = extract_first_query(conversation)
    assert query is None


def test_extract_first_query_empty_conversation():
    """Test with empty conversation."""
    assert extract_first_query([]) is None
    assert extract_first_query(None) is None


def test_extract_first_query_strips_whitespace():
    """Test that query text is stripped of whitespace."""
    conversation = [
        {"role": "user", "content": "  Whitespace query  \n"},
    ]
    
    query = extract_first_query(conversation)
    assert query == "Whitespace query"


# Tests for HuggingFaceAdapter


@pytest.fixture
def mock_hf_dataset():
    """Create a mock Hugging Face dataset."""
    # Mock dataset with sample records
    records = [
        {
            "conversation_id": "hf_conv1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            "model": "gpt-4",
            "language": "en",
            "timestamp": "2024-01-01T12:00:00",
            "redacted": False,
        },
        {
            "conversation_id": "hf_conv2",
            "conversation": [
                {"role": "user", "content": "Explain machine learning"},
                {"role": "assistant", "content": "Machine learning is..."},
            ],
            "model": "claude-3",
            "language": "en",
            "timestamp": "2024-01-02T12:00:00",
            "redacted": False,
        },
        {
            "conversation_id": "hf_conv3",
            "conversation": [
                {"role": "user", "content": "How do I code?"},
            ],
            "model": "gpt-3.5-turbo",
            "language": "en",
            "timestamp": None,
            "redacted": True,
        },
    ]
    
    # Create mock dataset object
    mock_dataset = MagicMock()
    mock_dataset.__iter__ = Mock(return_value=iter(records))
    mock_dataset.__len__ = Mock(return_value=len(records))
    mock_dataset.select = Mock(side_effect=lambda indices: [records[i] for i in indices])
    
    return mock_dataset


def test_huggingface_adapter_initialization():
    """Test HuggingFaceAdapter can be initialized."""
    adapter = HuggingFaceAdapter()
    
    assert adapter.dataset_name == "lmsys/lmsys-chat-1m"
    assert adapter.split == "train"
    assert adapter.use_streaming is False


def test_huggingface_adapter_custom_dataset():
    """Test HuggingFaceAdapter with custom dataset name."""
    adapter = HuggingFaceAdapter(
        dataset_name="custom/dataset",
        split="test",
        use_streaming=True
    )
    
    assert adapter.dataset_name == "custom/dataset"
    assert adapter.split == "test"
    assert adapter.use_streaming is True


def test_huggingface_adapter_iter_records(mock_hf_dataset, monkeypatch):
    """Test iterating records from HuggingFaceAdapter."""
    # Mock load_dataset to return our mock
    def mock_load_dataset(*args, **kwargs):
        return mock_hf_dataset
    
    monkeypatch.setattr("lmsys_query_analysis.db.adapters.load_dataset", mock_load_dataset)
    
    adapter = HuggingFaceAdapter()
    records = list(adapter.iter_records())
    
    assert len(records) == 3
    assert records[0]["conversation_id"] == "hf_conv1"
    assert records[0]["query_text"] == "What is Python?"
    assert records[0]["model"] == "gpt-4"
    assert records[0]["language"] == "en"
    assert records[0]["extra_metadata"]["turn_count"] == 2
    assert records[0]["extra_metadata"]["redacted"] is False


def test_huggingface_adapter_with_limit(mock_hf_dataset, monkeypatch):
    """Test HuggingFaceAdapter respects limit parameter."""
    def mock_load_dataset(*args, **kwargs):
        return mock_hf_dataset
    
    monkeypatch.setattr("lmsys_query_analysis.db.adapters.load_dataset", mock_load_dataset)
    
    adapter = HuggingFaceAdapter()
    records = list(adapter.iter_records(limit=2))
    
    assert len(records) == 2
    assert records[0]["conversation_id"] == "hf_conv1"
    assert records[1]["conversation_id"] == "hf_conv2"


def test_huggingface_adapter_skips_invalid_records(monkeypatch):
    """Test that HuggingFaceAdapter skips records with missing data."""
    invalid_records = [
        # Missing conversation_id
        {
            "conversation": [{"role": "user", "content": "Query"}],
            "model": "gpt-4",
        },
        # Valid record
        {
            "conversation_id": "valid1",
            "conversation": [{"role": "user", "content": "Valid query"}],
            "model": "gpt-4",
        },
        # No user message in conversation
        {
            "conversation_id": "no_user",
            "conversation": [{"role": "assistant", "content": "Answer"}],
            "model": "gpt-4",
        },
        # Valid record
        {
            "conversation_id": "valid2",
            "conversation": [{"role": "user", "content": "Another valid query"}],
            "model": "claude-3",
        },
    ]
    
    mock_dataset = MagicMock()
    mock_dataset.__iter__ = Mock(return_value=iter(invalid_records))
    
    def mock_load_dataset(*args, **kwargs):
        return mock_dataset
    
    monkeypatch.setattr("lmsys_query_analysis.db.adapters.load_dataset", mock_load_dataset)
    
    adapter = HuggingFaceAdapter()
    records = list(adapter.iter_records())
    
    # Should only get the 2 valid records
    assert len(records) == 2
    assert records[0]["conversation_id"] == "valid1"
    assert records[1]["conversation_id"] == "valid2"


def test_huggingface_adapter_parses_json_conversation(monkeypatch):
    """Test that HuggingFaceAdapter parses JSON string conversations."""
    import json
    
    records_with_json = [
        {
            "conversation_id": "json_conv",
            "conversation": json.dumps([
                {"role": "user", "content": "JSON query"},
            ]),
            "model": "gpt-4",
        },
    ]
    
    mock_dataset = MagicMock()
    mock_dataset.__iter__ = Mock(return_value=iter(records_with_json))
    
    def mock_load_dataset(*args, **kwargs):
        return mock_dataset
    
    monkeypatch.setattr("lmsys_query_analysis.db.adapters.load_dataset", mock_load_dataset)
    
    adapter = HuggingFaceAdapter()
    records = list(adapter.iter_records())
    
    assert len(records) == 1
    assert records[0]["query_text"] == "JSON query"


def test_huggingface_adapter_skips_malformed_json(monkeypatch):
    """Test that HuggingFaceAdapter skips malformed JSON conversations."""
    records_with_bad_json = [
        {
            "conversation_id": "bad_json",
            "conversation": "{invalid json}",
            "model": "gpt-4",
        },
        {
            "conversation_id": "good",
            "conversation": [{"role": "user", "content": "Good query"}],
            "model": "gpt-4",
        },
    ]
    
    mock_dataset = MagicMock()
    mock_dataset.__iter__ = Mock(return_value=iter(records_with_bad_json))
    
    def mock_load_dataset(*args, **kwargs):
        return mock_dataset
    
    monkeypatch.setattr("lmsys_query_analysis.db.adapters.load_dataset", mock_load_dataset)
    
    adapter = HuggingFaceAdapter()
    records = list(adapter.iter_records())
    
    # Should only get the good record
    assert len(records) == 1
    assert records[0]["conversation_id"] == "good"


def test_huggingface_adapter_handles_missing_optional_fields(monkeypatch):
    """Test HuggingFaceAdapter handles missing optional fields gracefully."""
    minimal_records = [
        {
            "conversation_id": "minimal",
            "conversation": [{"role": "user", "content": "Minimal query"}],
            # No model, language, timestamp, etc.
        },
    ]
    
    mock_dataset = MagicMock()
    mock_dataset.__iter__ = Mock(return_value=iter(minimal_records))
    
    def mock_load_dataset(*args, **kwargs):
        return mock_dataset
    
    monkeypatch.setattr("lmsys_query_analysis.db.adapters.load_dataset", mock_load_dataset)
    
    adapter = HuggingFaceAdapter()
    records = list(adapter.iter_records())
    
    assert len(records) == 1
    assert records[0]["conversation_id"] == "minimal"
    assert records[0]["query_text"] == "Minimal query"
    assert records[0]["model"] == "unknown"  # Default value
    assert records[0]["language"] is None
    assert records[0]["timestamp"] is None

