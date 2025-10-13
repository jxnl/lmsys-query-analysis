"""Data loading and management commands."""

import shutil
import time
import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sqlmodel import select
from sqlalchemy import func

from ..common import with_error_handling, db_path_option, chroma_path_option, embedding_model_option
from ..formatters import tables
from ..helpers.client_factory import parse_embedding_model, create_chroma_client, create_embedding_generator
from ...db.connection import get_db
from ...db.chroma import get_chroma
from ...db.loader import load_queries, load_queries_from_multiple
from ...db.sources import HuggingFaceSource, CSVSource
from ...db.models import Query

console = Console()


@with_error_handling
def load(
    csv: list[str] = typer.Option(None, "--csv", help="CSV file path(s) to load"),
    hf_dataset: str = typer.Option(None, "--hf-dataset", help="HuggingFace dataset ID (e.g., 'lmsys/lmsys-chat-1m')"),
    limit: int = typer.Option(None, help="Limit number of records to load (HF datasets only)"),
    db_path: str = db_path_option,
    use_chroma: bool = typer.Option(False, help="Enable ChromaDB for semantic search"),
    chroma_path: str = chroma_path_option,
    embedding_model: str = embedding_model_option,
    db_batch_size: int = typer.Option(5000, help="DB insert batch size"),
    streaming: bool = typer.Option(False, help="Use streaming dataset iteration (HF datasets only)"),
    no_pragmas: bool = typer.Option(
        False, help="Disable SQLite PRAGMA speedups during load"
    ),
    force_reload: bool = typer.Option(
        False, "--force-reload", help="Reload existing queries (skip duplicate check)"
    ),
):
    """Load query data from CSV files or HuggingFace datasets into SQLite.
    
    Examples:
        Load from CSV:
            lmsys load --csv data.csv
            
        Load from multiple CSVs:
            lmsys load --csv data1.csv --csv data2.csv
            
        Load from HuggingFace dataset:
            lmsys load --hf-dataset lmsys/lmsys-chat-1m --limit 1000
            
        Load LMSYS-1M (default):
            lmsys load --limit 1000
    """
    model, provider = parse_embedding_model(embedding_model)
    
    # Validate: cannot specify both CSV and HF dataset
    if csv and hf_dataset:
        console.print("[red]Error: Cannot specify both --csv and --hf-dataset[/red]")
        raise typer.Exit(1)
    
    db = get_db(db_path)
    chroma = create_chroma_client(chroma_path, model, provider) if use_chroma else None
    
    # Build source list
    sources = []
    
    if csv:
        # CSV sources
        for csv_path in csv:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                console.print(f"[red]Error: CSV file not found: {csv_path}[/red]")
                raise typer.Exit(1)
            sources.append(CSVSource(file_path=csv_path))
        
        console.print(
            f"[cyan]Loading from {len(sources)} CSV file(s): "
            f"use_chroma={use_chroma}, force_reload={force_reload}[/cyan]"
        )
    elif hf_dataset:
        # Specific HuggingFace dataset
        sources.append(HuggingFaceSource(
            dataset_id=hf_dataset,
            limit=limit,
            streaming=streaming,
        ))
        console.print(
            f"[cyan]Loading from HuggingFace dataset '{hf_dataset}': "
            f"limit={limit}, use_chroma={use_chroma}, force_reload={force_reload}[/cyan]"
        )
    else:
        # Default: LMSYS-1M
        sources.append(HuggingFaceSource(
            dataset_id="lmsys/lmsys-chat-1m",
            limit=limit,
            streaming=streaming,
        ))
        console.print(
            f"[cyan]Loading LMSYS-1M dataset: "
            f"limit={limit}, use_chroma={use_chroma}, force_reload={force_reload}[/cyan]"
        )
    
    # Validate sources
    for source in sources:
        try:
            source.validate_source()
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[red]Error validating source: {e}[/red]")
            raise typer.Exit(1)
    
    # Load data
    if len(sources) == 1:
        # Single source - use load_queries
        stats = load_queries(
            db,
            source=sources[0],
            skip_existing=not force_reload,
            chroma=chroma,
            embedding_model=model,
            embedding_provider=provider,
            batch_size=db_batch_size,
            apply_pragmas=not no_pragmas,
        )
        
        # Display results
        table = tables.format_loading_stats_table(stats)
        console.print(table)
    else:
        # Multiple sources - use load_queries_from_multiple
        stats_list = load_queries_from_multiple(
            db,
            sources=sources,
            skip_existing=not force_reload,
            chroma=chroma,
            embedding_model=model,
            embedding_provider=provider,
            batch_size=db_batch_size,
            apply_pragmas=not no_pragmas,
        )
        
        # Display results
        table = tables.format_multi_source_stats_table(stats_list)
        console.print(table)
    
    console.print(f"[cyan]Database path: {db.db_path}[/cyan]")
    if use_chroma and chroma:
        console.print(f"[cyan]Chroma path: {chroma.persist_directory}[/cyan]")
        console.print(f"[cyan]Chroma queries total: {chroma.count_queries()}[/cyan]")


@with_error_handling
def clear(
    db_path: str = db_path_option,
    chroma_path: str = chroma_path_option,
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Clear all data from SQLite database and ChromaDB."""
    db = get_db(db_path)
    # Use defaults for chroma - this just selects the directory, not specific collections
    chroma = get_chroma(chroma_path)
    
    # Show what will be deleted
    console.print("[yellow]This will delete:[/yellow]")
    console.print(f"  - SQLite database: {db.db_path}")
    console.print(f"  - ChromaDB data: {chroma.persist_directory}")
    
    # Confirm unless --yes flag is used
    if not confirm:
        response = typer.confirm("Are you sure you want to continue?")
        if not response:
            console.print("[green]Cancelled[/green]")
            raise typer.Exit(0)
    
    # Delete SQLite database
    db_file = Path(db.db_path)
    if db_file.exists():
        db_file.unlink()
        console.print(f"[green]Deleted SQLite database: {db.db_path}[/green]")
    else:
        console.print(f"[yellow]SQLite database not found: {db.db_path}[/yellow]")
    
    # Delete ChromaDB directory
    chroma_dir = Path(chroma.persist_directory)
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        console.print(
            f"[green]Deleted ChromaDB data: {chroma.persist_directory}[/green]"
        )
    else:
        console.print(
            f"[yellow]ChromaDB directory not found: {chroma.persist_directory}[/yellow]"
        )
    
    console.print("[green]Database cleared successfully[/green]")


@with_error_handling
def backfill_chroma(
    db_path: str = db_path_option,
    chroma_path: str = chroma_path_option,
    embedding_model: str = typer.Option(
        "openai/text-embedding-3-small", help="Embedding model (provider/model)"
    ),
    chunk_size: int = typer.Option(5000, help="DB iteration chunk size"),
    embed_batch_size: int = typer.Option(64, help="Embedding batch size"),
):
    """Backfill embeddings into ChromaDB for SQLite queries missing vectors."""
    model, provider = parse_embedding_model(embedding_model)
    
    db = get_db(db_path)
    chroma = create_chroma_client(chroma_path, model, provider)
    
    with db.get_session() as session:
        # Count total queries for progress
        count_result = session.exec(select(func.count()).select_from(Query)).one()
        total_queries = int(
            count_result[0] if isinstance(count_result, tuple) else count_result
        )
        if total_queries == 0:
            console.print("[yellow]No queries found in database[/yellow]")
            return
        
        console.print(
            f"[green]Backfilling embeddings for up to {total_queries} queries[/green]"
        )
        
        def iter_query_chunks():
            offset = 0
            while offset < total_queries:
                rows = session.exec(
                    select(Query).offset(offset).limit(chunk_size)
                ).all()
                if not rows:
                    break
                yield rows
                offset += len(rows)
        
        eg = create_embedding_generator(model, provider)
        
        backfilled = 0
        scanned = 0
        start = time.perf_counter()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            scan_task = progress.add_task(
                "[cyan]Scanning and backfilling...", total=total_queries
            )
            
            for chunk in iter_query_chunks():
                ids = [q.id for q in chunk]
                texts = [q.query_text for q in chunk]
                
                # Check which embeddings exist in Chroma
                present = chroma.get_query_embeddings_map(ids)
                missing_ids = [qid for qid in ids if qid not in present]
                if missing_ids:
                    missing_idx = [ids.index(mid) for mid in missing_ids]
                    to_embed_texts = [texts[i] for i in missing_idx]
                    
                    # Generate embeddings with visible progress handled by generator
                    emb = eg.generate_embeddings(
                        to_embed_texts,
                        batch_size=embed_batch_size,
                        show_progress=True,
                    )
                    
                    metadata = [
                        {
                            "model": chunk[i].model,
                            "language": chunk[i].language or "unknown",
                            "conversation_id": chunk[i].conversation_id,
                        }
                        for i in missing_idx
                    ]
                    
                    chroma.add_queries_batch(
                        query_ids=missing_ids,
                        texts=to_embed_texts,
                        embeddings=emb,
                        metadata=metadata,
                    )
                    backfilled += len(missing_ids)
                
                scanned += len(chunk)
                progress.update(
                    scan_task,
                    advance=len(chunk),
                    description=f"[cyan]Scanning and backfilling ({scanned}/{total_queries})...",
                )
        
        elapsed = time.perf_counter() - start
        already_present = max(0, scanned - backfilled)
        rate = (scanned / elapsed) if elapsed > 0 else float("inf")
        
        # Summary table
        summary = tables.format_backfill_summary_table(
            scanned, backfilled, already_present, elapsed, rate
        )
        console.print(summary)

