"""Data loader for LMSYS-1M dataset."""
import json
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from sqlmodel import Session, select
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from .models import Query
from .connection import Database
from .chroma import ChromaManager


def extract_first_query(conversation: list[dict]) -> str | None:
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


def load_lmsys_dataset(
    db: Database,
    limit: int | None = None,
    skip_existing: bool = True,
    chroma: Optional[ChromaManager] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """Load LMSYS-1M dataset into the database.

    Args:
        db: Database instance
        limit: Maximum number of records to load (None for all)
        skip_existing: Skip conversations that already exist in DB
        chroma: Optional ChromaDB manager for vector storage
        embedding_model: Model for generating embeddings (if chroma is provided)

    Returns:
        Dictionary with loading statistics
    """
    stats = {
        "total_processed": 0,
        "loaded": 0,
        "skipped": 0,
        "errors": 0,
    }

    # Create tables if they don't exist
    db.create_tables()

    # Load dataset from HuggingFace
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Downloading LMSYS-1M dataset...", total=None)
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
        progress.update(task, completed=True, description="[green]Dataset downloaded")

        # Apply limit if specified
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        load_task = progress.add_task("[cyan]Loading queries...", total=len(dataset))

        session = db.get_session()

        try:
            for idx, row in enumerate(dataset):
                stats["total_processed"] += 1

                conversation_id = row.get("conversation_id")
                if not conversation_id:
                    stats["errors"] += 1
                    continue

                # Skip if already exists
                if skip_existing:
                    statement = select(Query).where(Query.conversation_id == conversation_id)
                    existing = session.exec(statement).first()
                    if existing:
                        stats["skipped"] += 1
                        progress.update(load_task, advance=1)
                        continue

                # Parse conversation
                conversation = row.get("conversation")
                if isinstance(conversation, str):
                    try:
                        conversation = json.loads(conversation)
                    except json.JSONDecodeError:
                        stats["errors"] += 1
                        progress.update(load_task, advance=1)
                        continue

                # Extract first query
                query_text = extract_first_query(conversation)
                if not query_text:
                    stats["errors"] += 1
                    progress.update(load_task, advance=1)
                    continue

                # Create query record
                query = Query(
                    conversation_id=conversation_id,
                    model=row.get("model", "unknown"),
                    query_text=query_text,
                    language=row.get("language"),
                    timestamp=row.get("timestamp"),
                    extra_metadata={
                        "turn_count": len(conversation) if conversation else 0,
                        "redacted": row.get("redacted", False),
                        "openai_moderation": row.get("openai_moderation"),
                    }
                )

                session.add(query)
                stats["loaded"] += 1

                # Commit in batches
                if stats["loaded"] % 1000 == 0:
                    session.commit()

                progress.update(load_task, advance=1)

            # Final commit
            session.commit()

            # If ChromaDB is enabled, write embeddings after SQLite commit
            if chroma and stats["loaded"] > 0:
                progress.add_task("[cyan]Writing to ChromaDB...", total=None)

                # Get all newly loaded queries with their assigned IDs
                statement = select(Query)
                all_queries = session.exec(statement).all()

                if all_queries:
                    from ..clustering.embeddings import EmbeddingGenerator

                    # Generate embeddings
                    embedding_gen = EmbeddingGenerator(model_name=embedding_model)
                    query_texts = [q.query_text for q in all_queries]
                    embeddings = embedding_gen.generate_embeddings(
                        query_texts,
                        batch_size=32,
                        show_progress=False,
                    )

                    # Write to ChromaDB in batches
                    batch_size = 1000
                    for i in range(0, len(all_queries), batch_size):
                        batch_queries = all_queries[i:i + batch_size]
                        batch_embeddings = embeddings[i:i + batch_size]

                        query_ids = [q.id for q in batch_queries]
                        texts = [q.query_text for q in batch_queries]
                        metadata = [
                            {
                                "model": q.model,
                                "language": q.language or "unknown",
                                "conversation_id": q.conversation_id,
                            }
                            for q in batch_queries
                        ]

                        chroma.add_queries_batch(
                            query_ids=query_ids,
                            texts=texts,
                            embeddings=batch_embeddings,
                            metadata=metadata,
                        )

        finally:
            session.close()

    return stats
