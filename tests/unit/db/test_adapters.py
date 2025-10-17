"""Tests for data source adapters."""

from collections.abc import Iterator
from datetime import datetime
from unittest.mock import Mock, patch

from lmsys_query_analysis.db.adapters import (
    DataSourceAdapter,
    HuggingFaceAdapter,
    extract_first_query,
)



class MockDataSourceAdapter:
    """Mock adapter implementation for testing the protocol interface."""

    _NOT_PROVIDED = object()

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
    assert result == ""


def test_extract_first_query_missing_content():
    """Test handling message without content field."""
    conversation = [
        {"role": "user"},
        {"role": "user", "content": "Second message"},
    ]

    result = extract_first_query(conversation)
    assert result == ""


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




def test_adapter_protocol_interface():
    """Test that MockDataSourceAdapter conforms to DataSourceAdapter protocol."""
    adapter = MockDataSourceAdapter()

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
    test_data = [
        {
            "conversation_id": "conv1",
            "query_text": "Test",
            "model": "gpt-4",
            "language": None,
            "timestamp": None,
            "extra_metadata": {},
        }
    ]

    adapter = MockDataSourceAdapter(data=test_data, length=None)
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

    assert "conversation_id" in record
    assert "query_text" in record
    assert "model" in record
    assert "language" in record
    assert "timestamp" in record
    assert "extra_metadata" in record

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
            "language": None,
            "timestamp": None,
            "extra_metadata": {},
        },
    ]

    adapter = MockDataSourceAdapter(data=test_data)
    record = next(iter(adapter))

    assert record["language"] is None
    assert record["timestamp"] is None

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

    records1 = list(adapter)
    assert len(records1) == 1

    records2 = list(adapter)
    assert len(records2) == 1

    assert records1[0]["conversation_id"] == records2[0]["conversation_id"]




def test_hf_adapter_initialization():
    """Test that HuggingFaceAdapter initializes with correct parameters."""
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=1000)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(
            dataset_name="test/dataset",
            split="train",
            limit=100,
            use_streaming=False,
        )

        assert adapter.dataset_name == "test/dataset"
        assert adapter.split == "train"
        assert adapter.limit == 100
        assert adapter.use_streaming is False
        assert adapter.query_column == "conversation"
        assert adapter.is_conversation_format is True


def test_hf_adapter_iteration_basic():
    """Test basic iteration over HuggingFace adapter."""
    mock_data = [
        {
            "conversation_id": "conv1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            "model": "gpt-4",
            "language": "en",
            "timestamp": "2024-01-01T00:00:00",
        },
        {
            "conversation_id": "conv2",
            "conversation": [
                {"role": "user", "content": "Hello world"},
            ],
            "model": "claude-3",
            "language": "en",
            "timestamp": None,
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=2)
    mock_dataset.select = Mock(return_value=mock_dataset)
    mock_dataset.column_names = [
        "conversation_id",
        "conversation",
        "model",
        "language",
        "timestamp",
    ]

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset", limit=None)

        records = list(adapter)

        assert len(records) == 2
        assert records[0]["conversation_id"] == "conv1"
        assert records[0]["query_text"] == "What is Python?"
        assert records[0]["model"] == "gpt-4"
        assert records[0]["language"] == "en"
        assert records[1]["conversation_id"] == "conv2"
        assert records[1]["query_text"] == "Hello world"


def test_hf_adapter_with_limit():
    """Test HuggingFaceAdapter respects limit parameter."""
    mock_data = [
        {
            "conversation_id": f"conv{i}",
            "conversation": [{"role": "user", "content": f"Query {i}"}],
            "model": "gpt-4",
            "language": "en",
        }
        for i in range(10)
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data[:3]))
    mock_dataset.__len__ = Mock(return_value=3)
    mock_dataset.select = Mock(side_effect=lambda indices: mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset", limit=3)

        records = list(adapter)

        mock_dataset.select.assert_called_once()
        assert len(records) == 3


def test_hf_adapter_normalized_output_schema():
    """Test that HuggingFaceAdapter outputs conform to expected schema."""
    mock_data = [
        {
            "conversation_id": "test123",
            "conversation": [
                {"role": "user", "content": "What is machine learning?"},
            ],
            "model": "gpt-4",
            "language": "en",
            "timestamp": "2024-01-01T00:00:00",
            "redacted": False,
            "openai_moderation": {"flagged": False},
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=1)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset")

        record = next(iter(adapter))

        assert "conversation_id" in record
        assert "query_text" in record
        assert "model" in record
        assert "language" in record
        assert "timestamp" in record
        assert "extra_metadata" in record

        assert record["conversation_id"] == "test123"
        assert record["query_text"] == "What is machine learning?"
        assert record["model"] == "gpt-4"
        assert record["language"] == "en"
        assert record["timestamp"] == "2024-01-01T00:00:00"

        assert isinstance(record["extra_metadata"], dict)
        assert "turn_count" in record["extra_metadata"]
        assert "redacted" in record["extra_metadata"]
        assert "openai_moderation" in record["extra_metadata"]
        assert record["extra_metadata"]["turn_count"] == 1
        assert record["extra_metadata"]["redacted"] is False


def test_hf_adapter_handles_json_conversations():
    """Test that HuggingFaceAdapter handles JSON string conversations."""
    import json

    conversation = [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is..."},
    ]

    mock_data = [
        {
            "conversation_id": "conv1",
            "conversation": json.dumps(conversation),
            "model": "gpt-4",
            "language": "en",
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=1)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset")

        records = list(adapter)

        assert len(records) == 1
        assert records[0]["query_text"] == "What is AI?"
        assert records[0]["extra_metadata"]["turn_count"] == 2


def test_hf_adapter_handles_missing_fields():
    """Test that HuggingFaceAdapter handles missing optional fields."""
    mock_data = [
        {
            "conversation_id": "conv1",
            "conversation": [{"role": "user", "content": "Test query"}],
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=1)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset")

        records = list(adapter)

        assert len(records) == 1
        assert records[0]["conversation_id"] == "conv1"
        assert records[0]["query_text"] == "Test query"
        assert records[0]["model"] == "unknown"
        assert records[0]["language"] is None
        assert records[0]["timestamp"] is None
        assert records[0]["extra_metadata"]["redacted"] is False


def test_hf_adapter_skips_invalid_records():
    """Test that HuggingFaceAdapter skips records with no valid query."""
    mock_data = [
        {
            "conversation": [{"role": "user", "content": "Test"}],
            "model": "gpt-4",
        },
        {
            "conversation_id": "conv2",
            "conversation": [{"role": "system", "content": "System message"}],
            "model": "gpt-4",
        },
        {
            "conversation_id": "conv3",
            "conversation": [{"role": "user", "content": "Valid query"}],
            "model": "gpt-4",
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=3)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset")

        records = list(adapter)

        assert len(records) == 2
        import uuid

        assert uuid.UUID(records[0]["conversation_id"])
        assert records[0]["query_text"] == "Test"
        assert records[1]["conversation_id"] == "conv3"
        assert records[1]["query_text"] == "Valid query"


def test_hf_adapter_handles_invalid_json():
    """Test that HuggingFaceAdapter skips records with invalid JSON conversations."""
    mock_data = [
        {
            "conversation_id": "conv1",
            "conversation": "invalid json {{{",
            "model": "gpt-4",
        },
        {
            "conversation_id": "conv2",
            "conversation": [{"role": "user", "content": "Valid query"}],
            "model": "gpt-4",
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=2)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset")

        records = list(adapter)

        assert len(records) == 1
        assert records[0]["conversation_id"] == "conv2"


def test_hf_adapter_streaming_mode():
    """Test HuggingFaceAdapter in streaming mode."""
    mock_data = [
        {
            "conversation_id": f"conv{i}",
            "conversation": [{"role": "user", "content": f"Query {i}"}],
            "model": "gpt-4",
        }
        for i in range(5)
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(
            dataset_name="test/dataset",
            use_streaming=True,
            limit=3,
        )

        records = list(adapter)

        assert len(records) == 3
        assert records[0]["conversation_id"] == "conv0"
        assert records[2]["conversation_id"] == "conv2"

        assert adapter.__len__() is None


def test_hf_adapter_conforms_to_protocol():
    """Test that HuggingFaceAdapter conforms to DataSourceAdapter protocol."""
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter([]))
    mock_dataset.__len__ = Mock(return_value=0)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset")

        assert isinstance(adapter, DataSourceAdapter)


def test_hf_adapter_handles_simple_prompt_column():
    """Test HuggingFaceAdapter with simple prompt format (e.g., fka/awesome-chatgpt-prompts)."""
    mock_data = [
        {
            "act": "Linux Terminal",
            "prompt": "I want you to act as a linux terminal",
        },
        {
            "act": "Python Interpreter",
            "prompt": "Act as a Python interpreter",
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=2)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(
            dataset_name="fka/awesome-chatgpt-prompts",
            query_column="prompt",
            is_conversation_format=False,
        )

        records = list(adapter)

        assert len(records) == 2
        assert records[0]["query_text"] == "I want you to act as a linux terminal"
        assert records[1]["query_text"] == "Act as a Python interpreter"

        import uuid

        assert uuid.UUID(records[0]["conversation_id"])
        assert uuid.UUID(records[1]["conversation_id"])

        assert records[0]["extra_metadata"]["act"] == "Linux Terminal"
        assert records[1]["extra_metadata"]["act"] == "Python Interpreter"


def test_hf_adapter_uses_defaults():
    """Test that HuggingFaceAdapter uses correct defaults."""
    mock_data_conversation = [
        {
            "conversation_id": "conv1",
            "conversation": [{"role": "user", "content": "Test"}],
            "model": "gpt-4",
        }
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data_conversation))
    mock_dataset.__len__ = Mock(return_value=1)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(dataset_name="test/dataset")

        assert adapter.query_column == "conversation"
        assert adapter.is_conversation_format is True

        records = list(adapter)
        assert len(records) == 1

    mock_data_prompt = [{"prompt": "Test prompt"}]

    mock_dataset2 = Mock()
    mock_dataset2.__iter__ = Mock(return_value=iter(mock_data_prompt))
    mock_dataset2.__len__ = Mock(return_value=1)
    mock_dataset2.select = Mock(return_value=mock_dataset2)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset2):
        adapter2 = HuggingFaceAdapter(
            dataset_name="test/dataset",
            query_column="prompt",
            is_conversation_format=False,
        )

        assert adapter2.query_column == "prompt"
        assert adapter2.is_conversation_format is False


def test_hf_adapter_custom_column_mapping():
    """Test HuggingFaceAdapter with explicit column mapping."""
    mock_data = [
        {
            "custom_query": "This is my custom query field",
            "other_field": "ignored",
        }
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=1)
    mock_dataset.select = Mock(return_value=mock_dataset)
    mock_dataset.column_names = ["custom_query", "other_field"]

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(
            dataset_name="test/dataset",
            query_column="custom_query",
            is_conversation_format=False,
        )

        assert adapter.query_column == "custom_query"
        assert adapter.is_conversation_format is False

        records = list(adapter)

        assert len(records) == 1
        assert records[0]["query_text"] == "This is my custom query field"
        assert records[0]["extra_metadata"]["other_field"] == "ignored"


def test_hf_adapter_generates_uuid_for_missing_ids():
    """Test that adapter generates UUIDs when conversation_id is not in dataset."""
    mock_data = [
        {
            "prompt": "Query 1",
        },
        {
            "prompt": "Query 2",
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=2)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(
            dataset_name="test/dataset",
            query_column="prompt",
            is_conversation_format=False,
        )

        records = list(adapter)

        assert len(records) == 2

        import uuid

        conv_id_1 = records[0]["conversation_id"]
        conv_id_2 = records[1]["conversation_id"]

        assert uuid.UUID(conv_id_1)
        assert uuid.UUID(conv_id_2)

        assert conv_id_1 != conv_id_2


def test_hf_adapter_preserves_existing_conversation_ids():
    """Test that adapter preserves conversation_id when present in dataset."""
    mock_data = [
        {
            "conversation_id": "existing-id-123",
            "prompt": "Test query",
        },
        {
            "conversation_id": "another-id-456",
            "prompt": "Another query",
        },
    ]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=2)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(
            dataset_name="test/dataset",
            query_column="prompt",
            is_conversation_format=False,
        )

        records = list(adapter)

        assert len(records) == 2

        assert records[0]["conversation_id"] == "existing-id-123"
        assert records[1]["conversation_id"] == "another-id-456"


def test_hf_adapter_with_text_column():
    """Test adapter with generic 'text' column."""
    mock_data = [{"text": "Generic text content"}]

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=1)
    mock_dataset.select = Mock(return_value=mock_dataset)

    with patch("lmsys_query_analysis.db.adapters.load_dataset", return_value=mock_dataset):
        adapter = HuggingFaceAdapter(
            dataset_name="test/dataset",
            query_column="text",
            is_conversation_format=False,
        )

        assert adapter.query_column == "text"
        assert adapter.is_conversation_format is False

        records = list(adapter)
        assert len(records) == 1
        assert records[0]["query_text"] == "Generic text content"
