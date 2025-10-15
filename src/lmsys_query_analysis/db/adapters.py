"""Data source adapters for loading queries from different sources.

This module provides a uniform interface for loading query data from various
sources (Hugging Face datasets, CSV files, etc.) into the LMSYS query analysis
system.

The adapter pattern allows the loader to work with any data source that can
provide records in the normalized RecordDict format.
"""

from typing import Protocol, TypedDict, Optional, Iterator
from datetime import datetime


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

