"""Data source adapters for multi-source ingestion.

This module provides adapters that normalize different data sources
(HuggingFace datasets, CSV files, etc.) into a common format for ingestion.

The adapter pattern allows the loader to remain source-agnostic while
supporting multiple input formats.

Example usage:
    >>> from lmsys_query_analysis.db.adapters import DataSourceAdapter
    >>> # Future: adapter = HuggingFaceAdapter(limit=100)
    >>> # for record in adapter:
    >>> #     print(record["query_text"])

Architecture:
    - DataSourceAdapter: Protocol defining the adapter interface
    - HuggingFaceAdapter: (Phase 1b) Adapter for HuggingFace datasets
    - CSVAdapter: (Future) Adapter for CSV files
"""

from typing import Protocol, Iterator, runtime_checkable


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

