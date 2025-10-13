"""Tests for data source abstractions (BaseSource, HuggingFaceSource, etc.)."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lmsys_query_analysis.db.sources import BaseSource, HuggingFaceSource, CSVSource


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


class TestCSVSource:
    """Test the CSVSource implementation."""
    
    @pytest.fixture
    def fixtures_dir(self):
        """Path to test fixtures directory."""
        return Path(__file__).parent.parent.parent / "fixtures"
    
    def test_initialization(self):
        """Test basic initialization of CSVSource."""
        source = CSVSource(file_path="/tmp/test.csv")
        assert source.file_path == Path("/tmp/test.csv")
    
    def test_validate_source_file_not_found(self):
        """Test validate_source raises FileNotFoundError for missing file."""
        source = CSVSource(file_path="/nonexistent/file.csv")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            source.validate_source()
        
        assert "CSV file not found" in str(exc_info.value)
    
    def test_validate_source_missing_conversation_id_column(self, fixtures_dir):
        """Test validate_source raises ValueError when conversation_id column missing."""
        # Create temp CSV with only query_text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("query_text,model\n")
            f.write("Test query,gpt-4\n")
            temp_path = f.name
        
        try:
            source = CSVSource(file_path=temp_path)
            
            with pytest.raises(ValueError) as exc_info:
                source.validate_source()
            
            assert "missing required columns" in str(exc_info.value)
            assert "conversation_id" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_source_missing_query_text_column(self, fixtures_dir):
        """Test validate_source raises ValueError when query_text column missing."""
        invalid_headers_csv = fixtures_dir / "invalid_headers.csv"
        source = CSVSource(file_path=str(invalid_headers_csv))
        
        with pytest.raises(ValueError) as exc_info:
            source.validate_source()
        
        assert "missing required columns" in str(exc_info.value)
        assert "query_text" in str(exc_info.value)
    
    def test_validate_source_empty_file(self):
        """Test validate_source raises ValueError for empty CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write nothing
            temp_path = f.name
        
        try:
            source = CSVSource(file_path=temp_path)
            
            with pytest.raises(ValueError) as exc_info:
                source.validate_source()
            
            assert "empty or has no headers" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_source_valid_csv(self, fixtures_dir):
        """Test validate_source succeeds for valid CSV."""
        valid_csv = fixtures_dir / "valid_queries.csv"
        source = CSVSource(file_path=str(valid_csv))
        
        # Should not raise
        source.validate_source()
    
    def test_get_source_label(self):
        """Test get_source_label returns correct format."""
        source = CSVSource(file_path="/path/to/data.csv")
        label = source.get_source_label()
        
        assert label == "csv:/path/to/data.csv"
    
    def test_iter_records_valid_csv(self, fixtures_dir):
        """Test iter_records yields correct records from valid CSV."""
        valid_csv = fixtures_dir / "valid_queries.csv"
        source = CSVSource(file_path=str(valid_csv))
        
        records = list(source.iter_records())
        
        assert len(records) == 5
        
        # Check first record
        record = records[0]
        assert record["conversation_id"] == "conv_1"
        assert record["query_text"] == "What is machine learning?"
        assert record["model"] == "gpt-4"
        assert record["language"] == "English"
        assert isinstance(record["timestamp"], datetime)
        assert record["extra_metadata"] is None
    
    def test_iter_records_skips_empty_conversation_id(self, fixtures_dir):
        """Test iter_records skips rows with empty conversation_id."""
        empty_fields_csv = fixtures_dir / "empty_fields.csv"
        source = CSVSource(file_path=str(empty_fields_csv))
        
        records = list(source.iter_records())
        
        # Should skip row with empty conversation_id (row 2)
        assert len(records) == 2  # Only conv_1 and conv_4
        assert records[0]["conversation_id"] == "conv_1"
        assert records[1]["conversation_id"] == "conv_4"
    
    def test_iter_records_skips_empty_query_text(self, fixtures_dir):
        """Test iter_records skips rows with empty query_text."""
        empty_fields_csv = fixtures_dir / "empty_fields.csv"
        source = CSVSource(file_path=str(empty_fields_csv))
        
        records = list(source.iter_records())
        
        # Should skip row with empty query_text (row 3: conv_3)
        assert len(records) == 2
        conv_ids = [r["conversation_id"] for r in records]
        assert "conv_3" not in conv_ids
    
    def test_iter_records_defaults_model_to_unknown(self):
        """Test iter_records defaults model to 'unknown' when missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("conversation_id,query_text\n")
            f.write("conv_1,Test query\n")
            temp_path = f.name
        
        try:
            source = CSVSource(file_path=temp_path)
            records = list(source.iter_records())
            
            assert len(records) == 1
            assert records[0]["model"] == "unknown"
        finally:
            Path(temp_path).unlink()
    
    def test_iter_records_handles_none_language(self):
        """Test iter_records sets language to None when missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("conversation_id,query_text,model\n")
            f.write("conv_1,Test query,gpt-4\n")
            temp_path = f.name
        
        try:
            source = CSVSource(file_path=temp_path)
            records = list(source.iter_records())
            
            assert len(records) == 1
            assert records[0]["language"] is None
        finally:
            Path(temp_path).unlink()
    
    def test_iter_records_handles_invalid_timestamp(self):
        """Test iter_records sets timestamp to None for invalid formats."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("conversation_id,query_text,model,timestamp\n")
            f.write("conv_1,Test query,gpt-4,invalid-timestamp\n")
            temp_path = f.name
        
        try:
            source = CSVSource(file_path=temp_path)
            records = list(source.iter_records())
            
            assert len(records) == 1
            assert records[0]["timestamp"] is None
        finally:
            Path(temp_path).unlink()
    
    def test_iter_records_parses_valid_timestamp(self, fixtures_dir):
        """Test iter_records correctly parses ISO-8601 timestamps."""
        valid_csv = fixtures_dir / "valid_queries.csv"
        source = CSVSource(file_path=str(valid_csv))
        
        records = list(source.iter_records())
        
        # Check timestamp parsing
        assert isinstance(records[0]["timestamp"], datetime)
        assert records[0]["timestamp"].year == 2024
        assert records[0]["timestamp"].month == 1
        assert records[0]["timestamp"].day == 1
    
    def test_iter_records_ignores_extra_columns(self):
        """Test iter_records ignores extra columns not in schema."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("conversation_id,query_text,model,extra_col1,extra_col2\n")
            f.write("conv_1,Test query,gpt-4,value1,value2\n")
            temp_path = f.name
        
        try:
            source = CSVSource(file_path=temp_path)
            records = list(source.iter_records())
            
            assert len(records) == 1
            # Should not include extra columns
            assert "extra_col1" not in records[0]
            assert "extra_col2" not in records[0]
        finally:
            Path(temp_path).unlink()
    
    def test_iter_records_normalized_format(self, fixtures_dir):
        """Test iter_records returns correctly normalized format."""
        valid_csv = fixtures_dir / "valid_queries.csv"
        source = CSVSource(file_path=str(valid_csv))
        
        records = list(source.iter_records())
        
        # Verify all required fields present
        for record in records:
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
    
    def test_iter_records_strips_whitespace(self):
        """Test iter_records strips whitespace from fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("conversation_id,query_text,model\n")
            f.write("  conv_1  ,  Test query  ,  gpt-4  \n")
            temp_path = f.name
        
        try:
            source = CSVSource(file_path=temp_path)
            records = list(source.iter_records())
            
            assert len(records) == 1
            assert records[0]["conversation_id"] == "conv_1"
            assert records[0]["query_text"] == "Test query"
            assert records[0]["model"] == "gpt-4"
        finally:
            Path(temp_path).unlink()

