"""Data source abstraction for loading queries from various sources.

This module provides:
- BaseSource: Abstract base class for all data sources
- HuggingFaceSource: Load queries from HuggingFace datasets
- CSVSource: Load queries from CSV files
- extract_first_query: Helper to extract first user query from conversation
"""

import csv
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from datasets import load_dataset
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

logger = logging.getLogger(__name__)


def extract_first_query(conversation: list[dict] | None) -> str | None:
    """Extract the first user query from a conversation.

    Args:
        conversation: List of conversation turns in OpenAI format

    Returns:
        The first user message content, or None if not found
    """
    if not conversation:
        return None

    for turn in conversation:
        if turn.get("role") == "user":
            return turn.get("content", "").strip()

    return None


class BaseSource(ABC):
    """Abstract base class for data sources that provide query records.
    
    All sources must implement:
    - validate_source(): Check that the source is accessible
    - iter_records(): Yield normalized record dictionaries
    - get_source_label(): Return a human-readable source identifier
    """

    @abstractmethod
    def validate_source(self) -> None:
        """Validate that the source is accessible and properly configured.
        
        Raises:
            ValueError: If source is invalid or inaccessible
            FileNotFoundError: If source file doesn't exist
            Exception: Other source-specific errors
        """
        pass

    @abstractmethod
    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Iterate over records from the source.
        
        Yields:
            Normalized record dictionaries with keys:
            - conversation_id (str, required): Unique conversation identifier
            - query_text (str, required): The user's query text
            - model (str, required): Model identifier (default "unknown")
            - language (Optional[str]): Language code
            - timestamp (Optional[datetime]): Record timestamp
            - extra_metadata (Optional[dict]): Additional metadata
        """
        pass

    @abstractmethod
    def get_source_label(self) -> str:
        """Get a human-readable label for this source.
        
        Returns:
            Label in format "type:identifier" (e.g., "hf:org/dataset")
        """
        pass


class HuggingFaceSource(BaseSource):
    """Load queries from HuggingFace datasets.
    
    Expects dataset to have records with:
    - conversation_id: Unique identifier
    - conversation: List of chat turns or JSON string of turns
    - model: Model identifier (optional, defaults to "unknown")
    - language: Language code (optional)
    - timestamp: Timestamp (optional)
    - Additional metadata fields (turn_count, redacted, openai_moderation)
    """

    def __init__(
        self,
        dataset_id: str,
        limit: Optional[int] = None,
        streaming: bool = True,
    ):
        """Initialize HuggingFace data source.
        
        Args:
            dataset_id: HuggingFace dataset identifier (e.g., "lmsys/lmsys-chat-1m")
            limit: Maximum number of records to load (None = all)
            streaming: Use streaming mode (True) or load full dataset (False)
        """
        self.dataset_id = dataset_id
        self.limit = limit
        self.streaming = streaming
        self._dataset = None

    def validate_source(self) -> None:
        """Validate that the HuggingFace dataset is accessible.
        
        Raises:
            ValueError: If dataset_id is invalid or dataset can't be loaded
        """
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        
        # Basic validation - actual loading is deferred to iter_records()
        if not isinstance(self.dataset_id, str) or "/" not in self.dataset_id:
            raise ValueError(
                f"Invalid dataset_id format: {self.dataset_id}. "
                "Expected format: 'org/dataset'"
            )

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Iterate over records from HuggingFace dataset.
        
        Downloads dataset (with progress indicator) and yields normalized records.
        Parses conversation field using extract_first_query() to get query text.
        
        Yields:
            Normalized record dictionaries
        """
        # Load dataset with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Downloading {self.dataset_id}...", total=None
            )
            
            if self.streaming:
                dataset = load_dataset(self.dataset_id, split="train", streaming=True)
            else:
                dataset = load_dataset(self.dataset_id, split="train")
            
            progress.update(
                task, completed=True, description="[green]Dataset downloaded"
            )

        # Apply limit if specified and not streaming
        if self.limit and not self.streaming:
            dataset = dataset.select(range(min(self.limit, len(dataset))))

        # Helper to limit streaming datasets
        def _limited_iter(it, limit):
            if limit is None:
                yield from it
            else:
                for i, x in enumerate(it):
                    if i >= limit:
                        break
                    yield x

        # Iterate and normalize records
        source_iter = _limited_iter(dataset, self.limit if self.streaming else None)
        
        for row in source_iter:
            conversation_id = row.get("conversation_id")
            if not conversation_id:
                continue

            # Parse conversation field (may be list or JSON string)
            conversation = row.get("conversation")
            if isinstance(conversation, str):
                try:
                    conversation = json.loads(conversation)
                except json.JSONDecodeError:
                    continue

            # Extract first user query from conversation
            query_text = extract_first_query(conversation)
            if query_text is None:
                continue

            # Build normalized record
            model = row.get("model", "unknown")
            language = row.get("language") or None
            timestamp = row.get("timestamp")

            # Build extra metadata
            extra_metadata = {
                "turn_count": len(conversation) if conversation else 0,
                "redacted": row.get("redacted", False),
                "openai_moderation": row.get("openai_moderation"),
            }

            yield {
                "conversation_id": conversation_id,
                "query_text": query_text,
                "model": model,
                "language": language,
                "timestamp": timestamp,
                "extra_metadata": extra_metadata,
            }

    def get_source_label(self) -> str:
        """Get label for this HuggingFace dataset source.
        
        Returns:
            Label in format "hf:dataset_id"
        """
        return f"hf:{self.dataset_id}"


class CSVSource(BaseSource):
    """Load queries from CSV files.
    
    Expects CSV to have columns:
    - conversation_id (required): Unique identifier
    - query_text (required): The user's query text
    - model (optional): Model identifier (defaults to "unknown")
    - language (optional): Language code
    - timestamp (optional): ISO-8601 timestamp
    
    CSV must be UTF-8 encoded with headers in the first row.
    Rows with empty conversation_id or query_text are skipped.
    """

    def __init__(self, file_path: str):
        """Initialize CSV data source.
        
        Args:
            file_path: Path to CSV file
        """
        self.file_path = Path(file_path)
        self._skipped_rows = 0

    def validate_source(self) -> None:
        """Validate that the CSV file exists and has required columns.
        
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        # Check headers
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            if headers is None:
                raise ValueError(f"CSV file is empty or has no headers: {self.file_path}")
            
            required_columns = {'conversation_id', 'query_text'}
            missing_columns = required_columns - set(headers)
            
            if missing_columns:
                raise ValueError(
                    f"CSV file missing required columns: {missing_columns}. "
                    f"Required: {required_columns}. Found: {set(headers)}"
                )

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Iterate over records from CSV file.
        
        Parses CSV and yields normalized records.
        Skips rows with empty conversation_id or query_text.
        
        Yields:
            Normalized record dictionaries
        """
        self._skipped_rows = 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                # Check required fields
                conversation_id = row.get('conversation_id', '').strip()
                query_text = row.get('query_text', '').strip()
                
                if not conversation_id or not query_text:
                    self._skipped_rows += 1
                    if not conversation_id:
                        logger.warning(
                            f"Row {row_num} in {self.file_path.name}: "
                            "Empty conversation_id, skipping"
                        )
                    if not query_text:
                        logger.warning(
                            f"Row {row_num} in {self.file_path.name}: "
                            "Empty query_text, skipping"
                        )
                    continue
                
                # Get optional fields
                model = row.get('model', '').strip() or 'unknown'
                language = row.get('language', '').strip() or None
                
                # Parse timestamp
                timestamp = None
                timestamp_str = row.get('timestamp', '').strip()
                if timestamp_str:
                    try:
                        # Replace 'Z' with '+00:00' for Python 3.10 compatibility
                        if timestamp_str.endswith('Z'):
                            timestamp_str = timestamp_str[:-1] + '+00:00'
                        timestamp = datetime.fromisoformat(timestamp_str)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Row {row_num} in {self.file_path.name}: "
                            f"Invalid timestamp '{timestamp_str}': {e}"
                        )
                
                # Build normalized record
                yield {
                    'conversation_id': conversation_id,
                    'query_text': query_text,
                    'model': model,
                    'language': language,
                    'timestamp': timestamp,
                    'extra_metadata': None,
                }

    def get_source_label(self) -> str:
        """Get label for this CSV source.
        
        Returns:
            Label in format "csv:filename"
        """
        return f"csv:{self.file_path}"

