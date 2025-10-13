"""Data loader for query datasets.

Performance optimizations:
- Batch inserts with pre-check for existing conversation_ids per batch
- SQLite PRAGMA tuning during load (WAL, synchronous=OFF, temp_store=MEMORY)
- Deduplicate within input to avoid redundant work
- Only embed newly-inserted queries into ChromaDB
"""

from typing import Optional, Iterable
from sqlmodel import select
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
from .sources import BaseSource

def load_queries(
    db: Database,
    source: BaseSource,
    chroma: Optional[ChromaManager] = None,
    embedding_model: str = "embed-v4.0",
    embedding_provider: str = "cohere",
    batch_size: int = 5000,
    skip_existing: bool = True,
    apply_pragmas: bool = True,
) -> dict:
    """Load queries from any data source into the database.

    Args:
        db: Database instance
        source: Data source implementing BaseSource interface
        chroma: Optional ChromaDB manager for vector storage
        embedding_model: Model for generating embeddings (if chroma is provided)
        embedding_provider: Provider for generating embeddings (if chroma is provided)
        batch_size: Number of records to process in each batch
        skip_existing: Skip conversations that already exist in DB
        apply_pragmas: Apply SQLite performance optimizations

    Returns:
        Dictionary with loading statistics including:
        - source: Source label
        - total_processed: Total records processed
        - loaded: Records successfully inserted
        - skipped: Records skipped (duplicates)
        - errors: Records with errors
    """
    stats = {
        "source": source.get_source_label(),
        "total_processed": 0,
        "loaded": 0,
        "skipped": 0,
        "errors": 0,
    }

    # Create tables if they don't exist
    db.create_tables()

    # Iterate records from source with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        load_task = progress.add_task("[cyan]Loading queries...", total=None)

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

            # Iterate records from source in batches
            for batch in chunk_iter(source.iter_records(), BATCH_SIZE):
                to_insert_rows: list[dict] = []
                batch_conv_ids: list[str] = []
                raw_texts_by_cid: dict[str, tuple[str, str, str]] = {}
                # tuple: (query_text, model, language)

                for record in batch:
                    stats["total_processed"] += 1
                    
                    # Records from source are already normalized
                    conversation_id = record.get("conversation_id")
                    if not conversation_id:
                        stats["errors"] += 1
                        continue

                    if conversation_id in seen_conv_ids:
                        stats["skipped"] += 1
                        continue

                    query_text = record.get("query_text")
                    if not query_text:
                        stats["errors"] += 1
                        continue

                    model = record.get("model", "unknown")
                    language = record.get("language") or None

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
                            "timestamp": record.get("timestamp"),
                            "extra_metadata": record.get("extra_metadata"),
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

    return stats
