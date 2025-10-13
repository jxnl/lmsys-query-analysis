"""Tests for data source abstractions (BaseSource, HuggingFaceSource, etc.)."""

import json
from unittest.mock import Mock, patch

import pytest

from lmsys_query_analysis.db.sources import BaseSource, HuggingFaceSource


class TestBaseSource:
    """Test the BaseSource abstract class."""
    
    def test_cannot_instantiate_base_source(self):
        """BaseSource is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSource()
    
    def test_subclass_must_implement_methods(self):
        """Subclasses must implement all abstract methods."""
        class IncompleteSource(BaseSource):
            pass
        
        with pytest.raises(TypeError):
            IncompleteSource()


class TestHuggingFaceSource:
    """Test the HuggingFaceSource implementation."""
    
    def test_initialization(self):
        """Test basic initialization of HuggingFaceSource."""
        source = HuggingFaceSource(
            dataset_id="lmsys/lmsys-chat-1m",
            limit=100,
            streaming=True
        )
        
        assert source.dataset_id == "lmsys/lmsys-chat-1m"
        assert source.limit == 100
        assert source.streaming is True
    
    def test_initialization_defaults(self):
        """Test HuggingFaceSource with default parameters."""
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        assert source.dataset_id == "org/dataset"
        assert source.limit is None
        assert source.streaming is True
    
    def test_validate_source_empty_dataset_id(self):
        """Test validation fails with empty dataset_id."""
        source = HuggingFaceSource(dataset_id="")
        
        with pytest.raises(ValueError, match="dataset_id cannot be empty"):
            source.validate_source()
    
    def test_validate_source_invalid_format(self):
        """Test validation fails with invalid dataset_id format."""
        source = HuggingFaceSource(dataset_id="invalid-format")
        
        with pytest.raises(ValueError, match="Invalid dataset_id format"):
            source.validate_source()
    
    def test_validate_source_valid_dataset_id(self):
        """Test validation passes with valid dataset_id."""
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        # Should not raise
        source.validate_source()
    
    def test_get_source_label(self):
        """Test get_source_label returns correct format."""
        source = HuggingFaceSource(dataset_id="lmsys/lmsys-chat-1m")
        
        assert source.get_source_label() == "hf:lmsys/lmsys-chat-1m"
    
    def test_iter_records_streaming_mode(self):
        """Test iter_records with streaming mode enabled."""
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
                "timestamp": "2024-01-01T00:00:00Z",
                "redacted": False,
                "openai_moderation": None,
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(
            dataset_id="org/dataset",
            streaming=True,
            limit=None
        )
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset) as mock_load:
            records = list(source.iter_records())
        
        # Verify load_dataset was called with streaming=True
        mock_load.assert_called_once_with("org/dataset", split="train", streaming=True)
        
        # Verify records
        assert len(records) == 1
        assert records[0]["conversation_id"] == "conv1"
        assert records[0]["query_text"] == "What is AI?"
        assert records[0]["model"] == "gpt-4"
        assert records[0]["language"] == "English"
        assert records[0]["extra_metadata"]["turn_count"] == 2
        assert records[0]["extra_metadata"]["redacted"] is False
    
    def test_iter_records_non_streaming_mode(self):
        """Test iter_records with streaming disabled."""
        mock_data = [
            {
                "conversation_id": "conv1",
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
        mock_dataset.select = Mock(return_value=mock_dataset)
        
        source = HuggingFaceSource(
            dataset_id="org/dataset",
            streaming=False,
            limit=100
        )
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset) as mock_load:
            records = list(source.iter_records())
        
        # Verify load_dataset was called without streaming parameter (defaults to False)
        mock_load.assert_called_once_with("org/dataset", split="train")
        
        # Verify select was called with limit
        mock_dataset.select.assert_called_once()
        
        # Verify records
        assert len(records) == 1
        assert records[0]["query_text"] == "Hola mundo"
    
    def test_iter_records_with_limit_streaming(self):
        """Test iter_records respects limit in streaming mode."""
        mock_data = [
            {
                "conversation_id": f"conv{i}",
                "model": "gpt-4",
                "conversation": [{"role": "user", "content": f"Query {i}"}],
            }
            for i in range(5)
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(
            dataset_id="org/dataset",
            streaming=True,
            limit=3
        )
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        # Should only process 3 records
        assert len(records) == 3
        assert records[0]["conversation_id"] == "conv0"
        assert records[2]["conversation_id"] == "conv2"
    
    def test_iter_records_parses_json_conversation(self):
        """Test iter_records handles conversation as JSON string."""
        mock_data = [
            {
                "conversation_id": "conv1",
                "model": "gpt-4",
                "conversation": json.dumps([
                    {"role": "user", "content": "JSON query"},
                    {"role": "assistant", "content": "Response"},
                ]),
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        assert len(records) == 1
        assert records[0]["query_text"] == "JSON query"
        assert records[0]["extra_metadata"]["turn_count"] == 2
    
    def test_iter_records_skips_invalid_json_conversation(self):
        """Test iter_records skips records with invalid JSON conversation."""
        mock_data = [
            {
                "conversation_id": "conv1",
                "model": "gpt-4",
                "conversation": "invalid json {",
            },
            {
                "conversation_id": "conv2",
                "model": "gpt-4",
                "conversation": [{"role": "user", "content": "Valid query"}],
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        # Should skip invalid record, only process valid one
        assert len(records) == 1
        assert records[0]["conversation_id"] == "conv2"
    
    def test_iter_records_skips_missing_conversation_id(self):
        """Test iter_records skips records without conversation_id."""
        mock_data = [
            {
                "model": "gpt-4",
                "conversation": [{"role": "user", "content": "Query"}],
            },
            {
                "conversation_id": "conv2",
                "model": "gpt-4",
                "conversation": [{"role": "user", "content": "Valid query"}],
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        assert len(records) == 1
        assert records[0]["conversation_id"] == "conv2"
    
    def test_iter_records_skips_no_user_query(self):
        """Test iter_records skips records without user query in conversation."""
        mock_data = [
            {
                "conversation_id": "conv1",
                "model": "gpt-4",
                "conversation": [
                    {"role": "assistant", "content": "Hello"},
                ],
            },
            {
                "conversation_id": "conv2",
                "model": "gpt-4",
                "conversation": [{"role": "user", "content": "Valid query"}],
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        assert len(records) == 1
        assert records[0]["conversation_id"] == "conv2"
    
    def test_iter_records_defaults_model_to_unknown(self):
        """Test iter_records defaults model to 'unknown' if not provided."""
        mock_data = [
            {
                "conversation_id": "conv1",
                "conversation": [{"role": "user", "content": "Query"}],
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        assert len(records) == 1
        assert records[0]["model"] == "unknown"
    
    def test_iter_records_handles_none_language(self):
        """Test iter_records converts empty language to None."""
        mock_data = [
            {
                "conversation_id": "conv1",
                "model": "gpt-4",
                "conversation": [{"role": "user", "content": "Query"}],
                "language": "",
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        assert len(records) == 1
        assert records[0]["language"] is None
    
    def test_iter_records_includes_extra_metadata(self):
        """Test iter_records includes all expected metadata fields."""
        mock_data = [
            {
                "conversation_id": "conv1",
                "model": "gpt-4",
                "conversation": [
                    {"role": "user", "content": "Query"},
                    {"role": "assistant", "content": "Response"},
                    {"role": "user", "content": "Follow-up"},
                ],
                "redacted": True,
                "openai_moderation": {"flagged": False},
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        assert len(records) == 1
        metadata = records[0]["extra_metadata"]
        assert metadata["turn_count"] == 3
        assert metadata["redacted"] is True
        assert metadata["openai_moderation"] == {"flagged": False}
    
    def test_iter_records_normalized_format(self):
        """Test iter_records returns correctly normalized format."""
        mock_data = [
            {
                "conversation_id": "conv1",
                "model": "gpt-4",
                "language": "English",
                "conversation": [{"role": "user", "content": "Test query"}],
                "timestamp": "2024-01-01T12:00:00Z",
            },
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        
        source = HuggingFaceSource(dataset_id="org/dataset")
        
        with patch('lmsys_query_analysis.db.sources.load_dataset', return_value=mock_dataset):
            records = list(source.iter_records())
        
        assert len(records) == 1
        record = records[0]
        
        # Verify all required fields present
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
        assert isinstance(record["extra_metadata"], dict)

