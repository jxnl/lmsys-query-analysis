"""Data source adapters for loading queries from different sources.

This module provides a uniform interface for loading query data from various
sources (Hugging Face datasets, CSV files, etc.) into the LMSYS query analysis
system.

The adapter pattern allows the loader to work with any data source that can
provide records in the normalized RecordDict format.
"""

import json
from typing import Protocol, TypedDict, Optional, Iterator
from datetime import datetime
from datasets import load_dataset

# Optional fast JSON
try:  # pragma: no cover - speed optimization only
    import orjson as _fastjson  # type: ignore

    def _json_loads(s: str):
        return _fastjson.loads(s)

except Exception:  # pragma: no cover

    def _json_loads(s: str):
        return json.loads(s)


class RecordDict(TypedDict, total=False):
    """Normalized record format expected by the loader.
    
    All adapters must yield records in this format. The loader will handle
    the conversion to database models.
    
    Required fields:
        conversation_id: Unique identifier for the conversation/query
        query_text: The actual query text (first user message)
        
    Optional fields:
        model: Model used (defaults to "unknown" if not provided)
        language: Language of the query (can be None)
        timestamp: When the query was created (can be None)
        extra_metadata: Additional metadata to store (can be empty dict)
    """
    conversation_id: str  # Required
    query_text: str  # Required
    model: str
    language: Optional[str]
    timestamp: Optional[datetime]
    extra_metadata: dict


class BaseAdapter(Protocol):
    """Protocol defining the interface for data source adapters.
    
    All data source adapters must implement this protocol to be compatible
    with the loader. The adapter is responsible for:
    
    1. Reading from the data source (HF dataset, CSV file, API, etc.)
    2. Normalizing the data into RecordDict format
    3. Handling data source-specific errors and validation
    4. Respecting the limit parameter if provided
    
    Example:
        ```python
        class MyAdapter:
            def __init__(self, source_path: str):
                self.source_path = source_path
                
            def iter_records(self, limit: Optional[int] = None) -> Iterator[RecordDict]:
                # Read from source and yield normalized records
                for i, raw_record in enumerate(self.read_source()):
                    if limit and i >= limit:
                        break
                    yield self.normalize_record(raw_record)
        ```
    """
    
    def iter_records(self, limit: Optional[int] = None) -> Iterator[RecordDict]:
        """Iterate over records from the data source.
        
        Args:
            limit: Maximum number of records to yield (None for all)
            
        Yields:
            RecordDict instances with normalized query data
            
        Raises:
            May raise source-specific exceptions for data access errors,
            validation errors, etc. These should include helpful error
            messages to guide the user.
        """
        ...


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


class HuggingFaceAdapter:
    """Adapter for loading data from Hugging Face datasets.
    
    This adapter loads data from any Hugging Face dataset that contains
    conversational data in the LMSYS-1M format (conversation turns with
    role and content fields).
    
    Example:
        ```python
        # Load from default LMSYS dataset
        adapter = HuggingFaceAdapter()
        
        # Load from a custom dataset
        adapter = HuggingFaceAdapter(dataset_name="username/my-dataset")
        
        # Iterate over records
        for record in adapter.iter_records(limit=100):
            print(record["query_text"])
        ```
    """
    
    def __init__(
        self,
        dataset_name: str = "lmsys/lmsys-chat-1m",
        split: str = "train",
        use_streaming: bool = False,
    ):
        """Initialize the Hugging Face adapter.
        
        Args:
            dataset_name: Name of the HF dataset (format: "owner/dataset")
            split: Dataset split to use (default: "train")
            use_streaming: Whether to use streaming mode for large datasets
        """
        self.dataset_name = dataset_name
        self.split = split
        self.use_streaming = use_streaming
        self._dataset = None
    
    def _load_dataset(self):
        """Lazy load the dataset on first access."""
        if self._dataset is None:
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.use_streaming
            )
        return self._dataset
    
    def iter_records(self, limit: Optional[int] = None) -> Iterator[RecordDict]:
        """Iterate over records from the Hugging Face dataset.
        
        Args:
            limit: Maximum number of records to yield (None for all)
            
        Yields:
            RecordDict instances with normalized query data
            
        Raises:
            ValueError: If dataset cannot be loaded or has invalid format
        """
        dataset = self._load_dataset()
        
        # Apply limit for non-streaming datasets
        if limit and not self.use_streaming:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        count = 0
        for row in dataset:
            # Check limit for streaming datasets
            if limit is not None and count >= limit:
                break
            
            # Extract required fields
            conversation_id = row.get("conversation_id")
            if not conversation_id:
                # Skip records without conversation_id
                continue
            
            # Parse conversation
            conversation = row.get("conversation")
            if isinstance(conversation, str):
                try:
                    conversation = _json_loads(conversation)
                except json.JSONDecodeError:
                    # Skip malformed conversations
                    continue
            
            # Extract first user query
            query_text = extract_first_query(conversation)
            if query_text is None:
                # Skip conversations with no user query
                continue
            
            # Extract optional fields
            model = row.get("model", "unknown")
            language = row.get("language") or None
            timestamp = row.get("timestamp")
            
            # Build extra metadata
            extra_metadata = {
                "turn_count": len(conversation) if conversation else 0,
                "redacted": row.get("redacted", False),
            }
            
            # Add OpenAI moderation if present
            openai_moderation = row.get("openai_moderation")
            if openai_moderation:
                extra_metadata["openai_moderation"] = openai_moderation
            
            # Yield normalized record
            yield RecordDict(
                conversation_id=conversation_id,
                query_text=query_text,
                model=model,
                language=language,
                timestamp=timestamp,
                extra_metadata=extra_metadata,
            )
            
            count += 1

