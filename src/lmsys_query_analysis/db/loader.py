"""Data loader for LMSYS-1M dataset.

Performance optimizations:
- Batch inserts with pre-check for existing conversation_ids per batch
- SQLite PRAGMA tuning during load (WAL, synchronous=OFF, temp_store=MEMORY)
- Deduplicate within input to avoid redundant work
- Only embed newly-inserted queries into ChromaDB
"""

from collections.abc import Iterable

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from sqlalchemy import func, text
from sqlmodel import select

from .adapters import HuggingFaceAdapter
from .chroma import ChromaManager
from .connection import Database
from .models import Dataset, Query

# Note: extract_first_query moved to adapters.py


def load_dataset(
    db: Database,
    limit: int | None = None,
    skip_existing: bool = True,
    chroma: ChromaManager | None = None,
    embedding_model: str = "embed-v4.0",
    embedding_provider: str = "cohere",
    batch_size: int = 5000,
    use_streaming: bool = False,
    apply_pragmas: bool = True,
    dataset_name: str = "lmsys/lmsys-chat-1m",
    dataset_label: str | None = None,
    dataset_description: str | None = None,
    text_column: str = "conversation",
    is_conversation_format: bool = True,
    model_column: str = "model",
    language_column: str = "language",
    timestamp_column: str = "timestamp",
    conversation_id_column: str = "conversation_id",
    model_default: str = "unknown",
) -> dict:
    """Load dataset from HuggingFace (or other sources in future).

    Args:
        db: Database instance
        limit: Maximum number of records to load (None for all)
        skip_existing: Skip conversations that already exist in DB
        chroma: Optional ChromaDB manager for vector storage
        embedding_model: Model for generating embeddings (if chroma is provided)
        embedding_provider: Provider for embeddings (cohere, openai, etc.)
        batch_size: Number of records per batch for DB inserts
        use_streaming: Use streaming dataset iteration
        apply_pragmas: Apply SQLite PRAGMA speedups during load
        dataset_name: HuggingFace dataset identifier (default: lmsys/lmsys-chat-1m)
        dataset_label: User-friendly label for dataset (default: derived from dataset_name)
        dataset_description: Optional description of the dataset
        text_column: Column name for query text (default: "conversation")
        is_conversation_format: Whether to parse as conversation JSON (default: True)
        model_column: Column name for model field (default: "model")
        language_column: Column name for language field (default: "language")
        timestamp_column: Column name for timestamp field (default: "timestamp")
        conversation_id_column: Column name for conversation_id field (default: "conversation_id")
        model_default: Default value for model if column missing (default: "unknown")

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

    # Get or create dataset record
    session = db.get_session()
    try:
        # Derive label from dataset_name if not provided
        if dataset_label is None:
            dataset_label = dataset_name.split("/")[-1]  # e.g., "WildChat-1M" from "allenai/WildChat-1M"

        # Check if dataset already exists
        existing_dataset = session.exec(
            select(Dataset).where(Dataset.name == dataset_label)
        ).first()

        if existing_dataset:
            dataset_id = existing_dataset.id
            print(f"Using existing dataset: {dataset_label} (ID: {dataset_id})")
        else:
            # Create new dataset record
            new_dataset = Dataset(
                name=dataset_label,
                source=dataset_name,
                description=dataset_description or f"Dataset loaded from {dataset_name}",
                query_count=0,
                column_mapping={
                    "text_column": text_column,
                    "is_conversation_format": is_conversation_format,
                    "model_column": model_column,
                    "language_column": language_column,
                    "timestamp_column": timestamp_column,
                    "conversation_id_column": conversation_id_column,
                    "model_default": model_default,
                },
            )
            session.add(new_dataset)
            session.commit()
            session.refresh(new_dataset)
            dataset_id = new_dataset.id
            print(f"Created new dataset: {dataset_label} (ID: {dataset_id})")
    finally:
        session.close()

    # Create adapter for HuggingFace dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]Downloading dataset {dataset_name}...", total=None)

        # Create the adapter (it will handle dataset loading internally)
        adapter = HuggingFaceAdapter(
            dataset_name=dataset_name,
            split="train",
            limit=limit,
            use_streaming=use_streaming,
            query_column=text_column,
            is_conversation_format=is_conversation_format,
            model_column=model_column,
            language_column=language_column,
            timestamp_column=timestamp_column,
            conversation_id_column=conversation_id_column,
            model_default=model_default,
        )

        progress.update(task, completed=True, description="[green]Dataset downloaded")

        # Setup progress for loading
        adapter_len = len(adapter)
        if adapter_len is None:
            load_task = progress.add_task("[cyan]Loading queries...", total=None)
        else:
            load_task = progress.add_task("[cyan]Loading queries...", total=adapter_len)

        session = db.get_session()

        try:
            # Speed up bulk inserts with SQLite PRAGMAs during this session
            if apply_pragmas:
                session.exec(text("PRAGMA journal_mode=WAL"))
                session.exec(text("PRAGMA synchronous=OFF"))
                session.exec(text("PRAGMA temp_store=MEMORY"))
                # Set a moderate in-memory page cache (~64MB)
                session.exec(text("PRAGMA cache_size=-65536"))

            # Detect if table is empty to skip existence checks in the common first-load case
            total_existing = session.exec(select(func.count(Query.id))).one()
            table_empty = (total_existing or 0) == 0

            # Internal helpers
            def chunk_iter(it: Iterable, size: int):
                buf = []
                for x in it:
                    buf.append(x)
                    if len(buf) >= size:
                        yield buf
                        buf = []
                if buf:
                    yield buf

            BATCH_SIZE = max(500, int(batch_size))

            # Accumulate info for Chroma only for newly inserted rows
            new_queries_meta: list[tuple[int, str, str, str, str]] = []
            # tuple: (id, query_text, model, language or 'unknown', conversation_id)

            seen_conv_ids: set[str] = set()

            # Iterate adapter (normalized records) and build rows in batches
            for batch in chunk_iter(adapter, BATCH_SIZE):
                to_insert_rows: list[dict] = []
                batch_conv_ids: list[str] = []
                raw_texts_by_cid: dict[str, tuple[str, str, str]] = {}
                # tuple: (query_text, model, language)

                for normalized_record in batch:
                    stats["total_processed"] += 1

                    # Extract fields from normalized record
                    conversation_id = normalized_record["conversation_id"]
                    query_text = normalized_record["query_text"]
                    model = normalized_record["model"]
                    language = normalized_record["language"]
                    timestamp = normalized_record["timestamp"]
                    extra_metadata = normalized_record["extra_metadata"]

                    # Check for duplicates within this load
                    if conversation_id in seen_conv_ids:
                        stats["skipped"] += 1
                        continue

                    # Track for stats and future Chroma
                    batch_conv_ids.append(conversation_id)
                    raw_texts_by_cid[conversation_id] = (
                        query_text,
                        model,
                        language or "unknown",
                    )

                    seen_conv_ids.add(conversation_id)

                    to_insert_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "conversation_id": conversation_id,
                            "model": model,
                            "query_text": query_text,
                            "language": language,
                            "timestamp": timestamp,
                            "extra_metadata": extra_metadata,
                        }
                    )

                # If nothing valid in this batch, just advance progress and continue
                progress.update(load_task, advance=len(batch))
                if not to_insert_rows:
                    continue

                # If skipping existing and table not empty, remove those already present
                if skip_existing and not table_empty:
                    # SQLite has parameter limits; chunk this IN query
                    EXIST_CHUNK = 900
                    existing_ids: set[str] = set()
                    for c in chunk_iter(batch_conv_ids, EXIST_CHUNK):
                        stmt = select(Query.conversation_id).where(Query.conversation_id.in_(c))
                        # Use scalars() to get flat list of conversation_id
                        rows = session.exec(stmt).all()
                        existing_ids.update(rows)

                    if existing_ids:
                        filtered_rows = [
                            r for r in to_insert_rows if r["conversation_id"] not in existing_ids
                        ]
                        stats["skipped"] += len(to_insert_rows) - len(filtered_rows)
                        to_insert_rows = filtered_rows

                # Bulk insert remaining rows using executemany
                if to_insert_rows:
                    table = Query.__table__
                    session.execute(table.insert(), to_insert_rows)
                    session.commit()

                    # Retrieve IDs for inserted conv_ids
                    # Note: Using IN on the conv_ids for just-inserted rows
                    INSERT_CHUNK = 900
                    inserted_count = 0
                    for c in chunk_iter(
                        [r["conversation_id"] for r in to_insert_rows], INSERT_CHUNK
                    ):
                        stmt = select(Query.id, Query.conversation_id).where(
                            Query.conversation_id.in_(c)
                        )
                        for qid, cid in session.exec(stmt).all():
                            qt, mdl, lang = raw_texts_by_cid[cid]
                            new_queries_meta.append((qid, qt, mdl, lang, cid))
                            inserted_count += 1

                    stats["loaded"] += inserted_count

                # After first successful insert, table is not empty anymore
                table_empty = False

        finally:
            session.close()

    # If ChromaDB is enabled, write embeddings only for newly inserted queries
    if chroma and stats["loaded"] > 0 and new_queries_meta:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            from ..clustering.embeddings import EmbeddingGenerator

            # Generate embeddings with richer progress
            emb_task = progress.add_task(
                f"[cyan]Embedding {len(new_queries_meta)} new queries...",
                total=len(new_queries_meta),
            )

            embedding_gen = EmbeddingGenerator(
                model_name=embedding_model,
                provider=embedding_provider,
            )
            query_texts = [q[1] for q in new_queries_meta]
            # Let EmbeddingGenerator show its own progress; we update our task to done after
            embeddings = embedding_gen.generate_embeddings(
                query_texts,
                batch_size=32,
                show_progress=True,
            )
            progress.update(emb_task, completed=len(new_queries_meta))

            # Write to ChromaDB in visible batches
            written = 0
            chroma_task = progress.add_task(
                f"[cyan]Writing to ChromaDB (0/{len(new_queries_meta)})...",
                total=len(new_queries_meta),
            )
            batch_size_chroma = 1000
            for i in range(0, len(new_queries_meta), batch_size_chroma):
                batch = new_queries_meta[i : i + batch_size_chroma]
                batch_embeddings = embeddings[i : i + batch_size_chroma]

                query_ids = [q[0] for q in batch]
                texts = [q[1] for q in batch]
                metadata = [
                    {
                        "model": q[2],
                        "language": q[3],
                        "conversation_id": q[4],
                    }
                    for q in batch
                ]

                chroma.add_queries_batch(
                    query_ids=query_ids,
                    texts=texts,
                    embeddings=batch_embeddings,
                    metadata=metadata,
                )
                written += len(batch)
                progress.update(
                    chroma_task,
                    advance=len(batch),
                    description=f"[cyan]Writing to ChromaDB ({written}/{len(new_queries_meta)})...",
                )

            progress.update(
                chroma_task,
                completed=len(new_queries_meta),
                description="[green]ChromaDB updated",
            )

    # Update dataset query count
    session = db.get_session()
    try:
        dataset = session.exec(select(Dataset).where(Dataset.id == dataset_id)).first()
        if dataset:
            dataset.query_count = session.exec(
                select(func.count(Query.id)).where(Query.dataset_id == dataset_id)
            ).one()
            session.add(dataset)
            session.commit()
    finally:
        session.close()

    return stats
