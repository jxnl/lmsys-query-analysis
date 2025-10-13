"""Data source abstraction for loading queries from various sources.

This module provides:
- BaseSource: Abstract base class for all data sources
- HuggingFaceSource: Load queries from HuggingFace datasets
- extract_first_query: Helper to extract first user query from conversation
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

from datasets import load_dataset
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)


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

