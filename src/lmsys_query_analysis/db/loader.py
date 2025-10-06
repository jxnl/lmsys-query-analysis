"""Data loader for LMSYS-1M dataset.

Performance optimizations:
- Batch inserts with pre-check for existing conversation_ids per batch
- SQLite PRAGMA tuning during load (WAL, synchronous=OFF, temp_store=MEMORY)
- Deduplicate within input to avoid redundant work
- Only embed newly-inserted queries into ChromaDB
"""

import json
from typing import Optional, Iterable
from datasets import load_dataset
from sqlmodel import Session, select
from sqlalchemy import func, text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from .models import Query
from .connection import Database
from .chroma import ChromaManager

# Optional fast JSON
try:  # pragma: no cover - speed optimization only
    import orjson as _fastjson  # type: ignore

    def _json_loads(s: str):
        return _fastjson.loads(s)

except Exception:  # pragma: no cover

    def _json_loads(s: str):
        return json.loads(s)


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


def load_lmsys_dataset(
    db: Database,
    limit: int | None = None,
    skip_existing: bool = True,
    chroma: Optional[ChromaManager] = None,
    embedding_model: str = "embed-v4.0",
    embedding_provider: str = "cohere",
    batch_size: int = 5000,
    use_streaming: bool = False,
    apply_pragmas: bool = True,
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
        if use_streaming:
            dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        else:
            dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
        progress.update(task, completed=True, description="[green]Dataset downloaded")

        # Apply limit if specified
        if limit and not use_streaming:
            dataset = dataset.select(range(min(limit, len(dataset))))

        if use_streaming:
            load_task = progress.add_task("[cyan]Loading queries...", total=None)
        else:
            load_task = progress.add_task(
                "[cyan]Loading queries...", total=len(dataset)
            )

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

            # Iterate dataset and build rows in batches
            # Create an iterator that respects limit when streaming
            def _limited_iter(it: Iterable, n: Optional[int]):
                if n is None:
                    for x in it:
                        yield x
                else:
                    count = 0
                    for x in it:
                        if count >= n:
                            break
                        yield x
                        count += 1

            source_iter = dataset if not use_streaming else dataset
            for batch in chunk_iter(
                _limited_iter(source_iter, limit if use_streaming else None), BATCH_SIZE
            ):
                to_insert_rows: list[dict] = []
                batch_conv_ids: list[str] = []
                raw_texts_by_cid: dict[str, tuple[str, str, str]] = {}
                # tuple: (query_text, model, language)

                for row in batch:
                    stats["total_processed"] += 1
                    conversation_id = row.get("conversation_id")
                    if not conversation_id:
                        stats["errors"] += 1
                        continue

                    if conversation_id in seen_conv_ids:
                        stats["skipped"] += 1
                        continue

                    conversation = row.get("conversation")
                    if isinstance(conversation, str):
                        try:
                            conversation = _json_loads(conversation)
                        except json.JSONDecodeError:
                            stats["errors"] += 1
                            continue

                    query_text = extract_first_query(conversation)
                    if query_text is None:
                        stats["errors"] += 1
                        continue

                    model = row.get("model", "unknown")
                    language = row.get("language") or None

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
                            "conversation_id": conversation_id,
                            "model": model,
                            "query_text": query_text,
                            "language": language,
                            "timestamp": row.get("timestamp"),
                            "extra_metadata": {
                                "turn_count": len(conversation) if conversation else 0,
                                "redacted": row.get("redacted", False),
                                "openai_moderation": row.get("openai_moderation"),
                            },
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
                        stmt = select(Query.conversation_id).where(
                            Query.conversation_id.in_(c)
                        )
                        # Use scalars() to get flat list of conversation_id
                        rows = session.exec(stmt).all()
                        existing_ids.update(rows)

                    if existing_ids:
                        filtered_rows = [
                            r
                            for r in to_insert_rows
                            if r["conversation_id"] not in existing_ids
                        ]
                        stats["skipped"] += len(to_insert_rows) - len(filtered_rows)
                        to_insert_rows = filtered_rows

                # Bulk insert remaining rows using executemany
                if to_insert_rows:
                    table = Query.__table__
                    result = session.execute(table.insert(), to_insert_rows)
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

    return stats
