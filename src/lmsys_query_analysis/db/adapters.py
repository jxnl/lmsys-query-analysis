"""Data source adapters for multi-source ingestion.

This module provides adapters that normalize different data sources
(HuggingFace datasets, CSV files, etc.) into a common format for ingestion.

The adapter pattern allows the loader to remain source-agnostic while
supporting multiple input formats.

Example usage:
    >>> from lmsys_query_analysis.db.adapters import HuggingFaceAdapter
    >>> adapter = HuggingFaceAdapter(limit=100)
    >>> for record in adapter:
    ...     print(record["query_text"])

Architecture:
    - DataSourceAdapter: Protocol defining the adapter interface
    - HuggingFaceAdapter: Adapter for HuggingFace datasets
    - CSVAdapter: (Future) Adapter for CSV files
"""

import json
import uuid
from typing import Protocol, Iterator, runtime_checkable, Optional, Any
from datasets import load_dataset, Dataset, IterableDataset


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Protocol defining interface for data source adapters.
    
    All adapters must yield normalized records with this schema:
    {
        "conversation_id": str,
        "query_text": str,
        "model": str,
        "language": str | None,
        "timestamp": datetime | None,
        "extra_metadata": dict,
    }
    """
    
    def __iter__(self) -> Iterator[dict]:
        """Yield normalized records with standard schema.
        
        Yields:
            dict: Normalized record containing conversation_id, query_text,
                  model, language, timestamp, and extra_metadata.
        """
        ...
    
    def __len__(self) -> int | None:
        """Return total count if known, None for streaming sources.
        
        Returns:
            int | None: Total number of records, or None if unknown/streaming.
        """
        ...


def extract_first_query(conversation: list[dict] | None) -> str | None:
    """Extract the first user query from a conversation.

    Args:
        conversation: List of conversation turns in OpenAI format
                     (each turn has "role" and "content" keys)

    Returns:
        The first user message content (stripped of whitespace),
        or None if not found
        
    Examples:
        >>> conv = [
        ...     {"role": "user", "content": "What is Python?"},
        ...     {"role": "assistant", "content": "Python is..."}
        ... ]
        >>> extract_first_query(conv)
        'What is Python?'
        
        >>> extract_first_query([])
        None
        
        >>> extract_first_query(None)
        None
    """
    if not conversation:
        return None

    for turn in conversation:
        if turn.get("role") == "user":
            return turn.get("content", "").strip()

    return None


# Optional fast JSON parsing
try:  # pragma: no cover - speed optimization only
    import orjson as _fastjson  # type: ignore

    def _json_loads(s: str) -> Any:
        return _fastjson.loads(s)

except Exception:  # pragma: no cover

    def _json_loads(s: str) -> Any:
        return json.loads(s)


class HuggingFaceAdapter:
    """Adapter for HuggingFace datasets with flexible schema mapping.
    
    Normalizes HuggingFace dataset records into a standard format for ingestion.
    Supports multiple dataset schemas:
    - LMSYS conversation format (conversation field with JSON structure)
    - Simple prompt format (direct text column, e.g., 'prompt')
    - Custom schemas (configurable field mappings)
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "lmsys/lmsys-chat-1m")
        split: Dataset split to load (default: "train")
        limit: Maximum number of records to yield (None for all records)
        use_streaming: Whether to use streaming mode (default: False)
        query_column: Column name for query text (default: "conversation")
        is_conversation_format: Whether to parse as conversation JSON (default: True)
    
    Example:
        >>> # Default LMSYS format
        >>> adapter = HuggingFaceAdapter("lmsys/lmsys-chat-1m", limit=100)
        >>> for record in adapter:
        ...     print(record["query_text"])
        
        >>> # Simple prompt format
        >>> adapter = HuggingFaceAdapter(
        ...     "fka/awesome-chatgpt-prompts",
        ...     query_column="prompt",
        ...     is_conversation_format=False
        ... )
    """
    
    def __init__(
        self,
        dataset_name: str = "lmsys/lmsys-chat-1m",
        split: str = "train",
        limit: Optional[int] = None,
        use_streaming: bool = False,
        query_column: str = "conversation",
        is_conversation_format: bool = True,
    ):
        """Initialize the HuggingFace adapter.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load
            limit: Maximum records to yield (None for all)
            use_streaming: Whether to use streaming mode
            query_column: Column name for query text (default: "conversation")
            is_conversation_format: Whether to parse as conversation JSON (default: True)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit
        self.use_streaming = use_streaming
        self.query_column = query_column
        self.is_conversation_format = is_conversation_format
        
        # Load the dataset
        if use_streaming:
            self._dataset: Dataset | IterableDataset = load_dataset(
                dataset_name, split=split, streaming=True
            )
        else:
            self._dataset = load_dataset(dataset_name, split=split)
            
            # Apply limit if specified (only for non-streaming)
            if limit is not None:
                dataset_len = len(self._dataset)
                self._dataset = self._dataset.select(range(min(limit, dataset_len)))
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate over dataset and yield normalized records.
        
        Yields:
            dict: Normalized record with standard schema:
                - conversation_id: str (from row or generated UUID)
                - query_text: str
                - model: str
                - language: str | None
                - timestamp: Any | None
                - extra_metadata: dict
        """
        count = 0
        
        for row in self._dataset:
            # Apply limit for streaming mode
            if self.use_streaming and self.limit is not None:
                if count >= self.limit:
                    break
                count += 1
            
            # Get or generate conversation_id
            conversation_id = self._get_or_generate_conversation_id(row)
            
            # Extract query text based on schema format
            if self.is_conversation_format:
                # LMSYS conversation format: parse JSON and extract first user query
                conversation = row.get(self.query_column)
                if isinstance(conversation, str):
                    try:
                        conversation = _json_loads(conversation)
                    except json.JSONDecodeError:
                        continue
                
                query_text = extract_first_query(conversation)
                if query_text is None:
                    continue
                
                # Build extra metadata for conversation format
                extra_metadata = {
                    "turn_count": len(conversation) if conversation else 0,
                    "redacted": row.get("redacted", False),
                }
                
                # Include openai_moderation if present
                if "openai_moderation" in row:
                    extra_metadata["openai_moderation"] = row.get("openai_moderation")
            else:
                # Simple prompt format: read directly from column
                query_text = row.get(self.query_column)
                if not query_text or not isinstance(query_text, str):
                    continue
                
                query_text = query_text.strip()
                if not query_text:
                    continue
                
                # Build extra metadata for prompt format
                extra_metadata = {}
                
                # Include any additional fields from row as metadata
                for key, value in row.items():
                    if key not in [self.query_column, "conversation_id", "model", "language", "timestamp"]:
                        extra_metadata[key] = value
            
            # Extract common metadata
            model = row.get("model", "unknown")
            language = row.get("language") or None
            timestamp = row.get("timestamp")
            
            # Yield normalized record
            yield {
                "conversation_id": conversation_id,
                "query_text": query_text,
                "model": model,
                "language": language,
                "timestamp": timestamp,
                "extra_metadata": extra_metadata,
            }
    
    def __len__(self) -> int | None:
        """Return total record count if known, None for streaming sources.
        
        Returns:
            int | None: Number of records, or None if streaming or unknown
        """
        if self.use_streaming:
            return None
        
        # For non-streaming with limit applied
        return len(self._dataset)
    
    @staticmethod
    def _get_or_generate_conversation_id(row: dict) -> str:
        """Get conversation_id from row or generate a new UUID.
        
        For datasets with conversation_id field, use the existing value.
        For datasets without conversation_id (e.g., simple prompt datasets),
        generate a random UUID to ensure uniqueness.
        
        Args:
            row: Dataset row (dict)
        
        Returns:
            Conversation ID string (from row or newly generated UUID)
        """
        # Check if row has conversation_id field
        if "conversation_id" in row and row["conversation_id"]:
            return str(row["conversation_id"])
        
        # Generate random UUID for datasets without conversation_id
        return str(uuid.uuid4())

