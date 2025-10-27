"""CLI command for LLM-based dataset summarization."""

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sqlalchemy import text
from sqlmodel import select

from ...clustering.embeddings import EmbeddingGenerator
from ...clustering.row_summarizer import RowSummarizer
from ...db.chroma import ChromaManager
from ...db.connection import Database
from ...db.models import ClusteringRun, Dataset, Prompt, Query, QueryCluster, compute_prompt_hash

app = typer.Typer()
console = Console()
logger = logging.getLogger("lmsys")


@app.command()
def summarize(
    source_dataset: str = typer.Argument(..., help="Source dataset name"),
    output: str = typer.Option(..., "--output", help="Output dataset name"),
    prompt: str = typer.Option(
        ..., "--prompt", help="Prompt template with {query} and optional {examples}"
    ),
    model: str = typer.Option("openai/gpt-4o-mini", "--model", help="LLM model (provider/model)"),
    limit: int | None = typer.Option(None, "--limit", help="Max queries to summarize (first N)"),
    where: str | None = typer.Option(None, "--where", help="SQL WHERE clause for filtering"),
    run_id: str | None = typer.Option(None, "--run-id", help="Filter by clustering run ID"),
    cluster_ids: str | None = typer.Option(
        None, "--cluster-ids", help="Comma-separated cluster IDs (requires --run-id)"
    ),
    examples_file: str | None = typer.Option(
        None, "--examples", help="JSONL file with few-shot examples"
    ),
    example: list[str] | None = typer.Option(
        None, "--example", help="Inline example 'query_id:output'"
    ),
    concurrency: int = typer.Option(100, "--concurrency", help="Concurrent LLM requests"),
    use_chroma: bool = typer.Option(
        False, "--use-chroma", help="Generate embeddings for summaries"
    ),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model"),
    embedding_provider: str = typer.Option("openai", "--embedding-provider"),
    db_path: str = typer.Option("~/.lmsys-query-analysis/queries.db", "--db-path"),
    chroma_path: str = typer.Option("~/.lmsys-query-analysis/chroma", "--chroma-path"),
):
    """Summarize queries using LLM to create derived dataset.

    The {query} placeholder in the prompt is replaced with the full Query record serialized as XML,
    including fields like query_text, model, language, timestamp, and extra_metadata.

    Examples:
        # Basic summarization
        lmsys summarize-dataset "lmsys-1m" \\
          --output "lmsys-1m-intent" \\
          --prompt "Extract user intent from <query_text>: {query}" \\
          --limit 10000 \\
          --use-chroma

        # Filter by specific clusters (after clustering analysis)
        lmsys summarize-dataset "WildChat-1M" \\
          --output "wildchat-security-analysis" \\
          --run-id hdbscan-10-20251027-045256 \\
          --cluster-ids 4,7,11 \\
          --prompt "Detailed security analysis: {query}"

        # With few-shot examples from file
        lmsys summarize-dataset "lmsys-1m" \\
          --output "lmsys-1m-classified" \\
          --prompt "Examples:\\n{examples}\\n\\nClassify this query: {query}" \\
          --examples examples.jsonl

        # With inline examples
        lmsys summarize-dataset "lmsys-1m" \\
          --output "lmsys-1m-intent" \\
          --prompt "Examples: {examples}\\nIntent: {query}" \\
          --example "12345:Write Python code" \\
          --example "67890:Debug JavaScript error"

        # Metadata-aware prompt
        lmsys summarize-dataset "lmsys-1m" \\
          --output "lmsys-1m-normalized" \\
          --prompt "If <language> is not English, translate <query_text> to English. Otherwise extract intent: {query}"
    """
    # Expand paths
    db_path = Path(db_path).expanduser()
    chroma_path = Path(chroma_path).expanduser()

    # Initialize database
    db = Database(db_path)
    session = db.get_session()

    try:
        # 1. Find source dataset
        source_ds = session.exec(select(Dataset).where(Dataset.name == source_dataset)).first()
        if not source_ds:
            console.print(f"[red]Error: Source dataset '{source_dataset}' not found[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Source dataset: {source_dataset} (ID: {source_ds.id})[/cyan]")

        # 1b. Validate cluster filtering flags
        if (run_id and not cluster_ids) or (cluster_ids and not run_id):
            console.print(
                "[red]Error: --run-id and --cluster-ids must be used together[/red]"
            )
            raise typer.Exit(1)

        if (run_id or cluster_ids) and where:
            console.print(
                "[red]Error: Cannot use both cluster filtering (--run-id/--cluster-ids) "
                "and custom --where clause[/red]"
            )
            raise typer.Exit(1)

        if run_id:
            # Validate run exists
            clustering_run = session.exec(
                select(ClusteringRun).where(ClusteringRun.run_id == run_id)
            ).first()
            if not clustering_run:
                console.print(f"[red]Error: Clustering run '{run_id}' not found[/red]")
                raise typer.Exit(1)

            # Parse cluster IDs
            try:
                cluster_id_list = [int(x.strip()) for x in cluster_ids.split(",")]
            except ValueError:
                console.print(
                    f"[red]Error: Invalid cluster IDs '{cluster_ids}' - must be comma-separated integers[/red]"
                )
                raise typer.Exit(1)

            console.print(
                f"[cyan]Filtering by run '{run_id}' "
                f"clusters {cluster_id_list}[/cyan]"
            )

        # 2. Check if output dataset already exists
        existing_output = session.exec(select(Dataset).where(Dataset.name == output)).first()
        if existing_output:
            console.print(
                f"[red]Error: Output dataset '{output}' already exists (ID: {existing_output.id})[/red]"
            )
            raise typer.Exit(1)

        # 3. Load few-shot examples
        examples_list: list[tuple[Query, str]] = []

        # Load from file if provided
        if examples_file:
            examples_path = Path(examples_file).expanduser()
            if not examples_path.exists():
                console.print(f"[red]Error: Examples file not found: {examples_file}[/red]")
                raise typer.Exit(1)

            with examples_path.open() as f:
                for line in f:
                    entry = json.loads(line.strip())
                    query_id = entry["query_id"]
                    expected_output = entry["output"]

                    # Fetch query from database
                    query_obj = session.exec(select(Query).where(Query.id == query_id)).first()
                    if not query_obj:
                        console.print(
                            f"[yellow]Warning: Query ID {query_id} not found, skipping example[/yellow]"
                        )
                        continue

                    examples_list.append((query_obj, expected_output))

            console.print(
                f"[green]Loaded {len(examples_list)} examples from {examples_file}[/green]"
            )

        # Load inline examples if provided
        if example:
            for ex in example:
                if ":" not in ex:
                    console.print(
                        f"[yellow]Warning: Invalid example format '{ex}', expected 'query_id:output'[/yellow]"
                    )
                    continue

                query_id_str, expected_output = ex.split(":", 1)
                try:
                    query_id = int(query_id_str)
                except ValueError:
                    console.print(
                        f"[yellow]Warning: Invalid query_id '{query_id_str}', skipping[/yellow]"
                    )
                    continue

                # Fetch query from database
                query_obj = session.exec(select(Query).where(Query.id == query_id)).first()
                if not query_obj:
                    console.print(
                        f"[yellow]Warning: Query ID {query_id} not found, skipping example[/yellow]"
                    )
                    continue

                examples_list.append((query_obj, expected_output))

            console.print(f"[green]Loaded {len(examples_list)} inline examples[/green]")

        # 4. Build query for source queries
        if run_id:
            # Filter by cluster: requires JOIN with query_clusters
            query_stmt = (
                select(Query)
                .join(QueryCluster, Query.id == QueryCluster.query_id)
                .where(Query.dataset_id == source_ds.id)
                .where(QueryCluster.run_id == run_id)
                .where(QueryCluster.cluster_id.in_(cluster_id_list))
            )
        else:
            # Standard query (no cluster filtering)
            query_stmt = select(Query).where(Query.dataset_id == source_ds.id)

            # Apply WHERE clause if provided
            if where:
                # Wrap in text() for raw SQL
                query_stmt = query_stmt.where(text(where))

        # Apply limit if provided
        if limit:
            query_stmt = query_stmt.limit(limit)

        # Fetch queries
        console.print("[cyan]Fetching source queries...[/cyan]")
        source_queries = session.exec(query_stmt).all()

        if not source_queries:
            console.print("[red]Error: No queries found matching criteria[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Found {len(source_queries)} queries to summarize[/green]")

        # 5. Store prompt in Prompt table
        prompt_hash = compute_prompt_hash(prompt)
        existing_prompt = session.exec(
            select(Prompt).where(Prompt.prompt_hash == prompt_hash)
        ).first()

        if not existing_prompt:
            new_prompt = Prompt(
                prompt_hash=prompt_hash,
                prompt_text=prompt,
                usage_count=1,
            )
            session.add(new_prompt)
            session.commit()
            logger.info(f"Created new prompt (hash: {prompt_hash[:8]}...)")
        else:
            existing_prompt.usage_count += 1
            session.add(existing_prompt)
            session.commit()
            logger.info(f"Reusing existing prompt (hash: {prompt_hash[:8]}...)")

        # 6. Create output dataset
        # Determine root dataset (traverse lineage if source is also derived)
        root_ds_id = source_ds.root_dataset_id if source_ds.root_dataset_id else source_ds.id

        output_ds = Dataset(
            name=output,
            source=f"llm-derived from {source_dataset}",
            description=f"LLM-summarized dataset from {source_dataset} using {model}",
            source_dataset_id=source_ds.id,
            root_dataset_id=root_ds_id,
            prompt_hash=prompt_hash,
            summarization_model=model,
            query_count=0,
        )
        session.add(output_ds)
        session.commit()
        session.refresh(output_ds)
        console.print(f"[green]Created output dataset: {output} (ID: {output_ds.id})[/green]")

        # 7. Initialize RowSummarizer
        console.print(f"[cyan]Initializing RowSummarizer with model {model}...[/cyan]")
        summarizer = RowSummarizer(
            model=model,
            prompt_template=prompt,
            examples=examples_list if examples_list else None,
            concurrency=concurrency,
        )

        # 8. Run summarization with progress
        console.print(f"[cyan]Summarizing {len(source_queries)} queries...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Summarizing with {model}...",
                total=len(source_queries),
            )

            # Run batch summarization
            summaries = summarizer.summarize_batch(source_queries)

            progress.update(task, completed=len(source_queries))

        console.print(f"[green]Summarized {len(summaries)} queries[/green]")

        # 9. Create new Query records with summaries
        console.print("[cyan]Creating new Query records...[/cyan]")
        new_queries = []

        for source_query in source_queries:
            if source_query.id not in summaries:
                logger.warning(f"No summary found for query {source_query.id}, skipping")
                continue

            summary_data = summaries[source_query.id]
            summary_text = summary_data["summary"]
            properties = summary_data["properties"]

            # Merge properties into extra_metadata
            merged_metadata = {
                "source_query_id": source_query.id,
                "summarization_model": model,
                **(source_query.extra_metadata or {}),
                **properties,
            }

            new_query = Query(
                dataset_id=output_ds.id,
                conversation_id=source_query.conversation_id,
                model=source_query.model,
                query_text=summary_text,
                language=source_query.language,
                timestamp=source_query.timestamp,
                extra_metadata=merged_metadata,
            )
            new_queries.append(new_query)

        # Bulk insert
        if new_queries:
            session.bulk_save_objects(new_queries)
            session.commit()
            console.print(f"[green]Created {len(new_queries)} new Query records[/green]")

            # Update dataset query count
            output_ds.query_count = len(new_queries)
            session.add(output_ds)
            session.commit()

        # 10. Generate embeddings if --use-chroma
        if use_chroma and new_queries:
            console.print("[cyan]Generating embeddings for ChromaDB...[/cyan]")

            # Refresh to get IDs
            session.expire_all()
            new_query_records = session.exec(
                select(Query).where(Query.dataset_id == output_ds.id)
            ).all()

            # Initialize ChromaDB
            chroma = ChromaManager(persist_directory=str(chroma_path))

            # Generate embeddings
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                emb_task = progress.add_task(
                    f"[cyan]Embedding {len(new_query_records)} queries...",
                    total=len(new_query_records),
                )

                embedding_gen = EmbeddingGenerator(
                    model_name=embedding_model,
                    provider=embedding_provider,
                )
                query_texts = [q.query_text for q in new_query_records]
                embeddings = embedding_gen.generate_embeddings(
                    query_texts,
                    batch_size=32,
                    show_progress=True,
                )
                progress.update(emb_task, completed=len(new_query_records))

                # Write to ChromaDB
                chroma_task = progress.add_task(
                    "[cyan]Writing to ChromaDB...",
                    total=len(new_query_records),
                )

                batch_size_chroma = 1000
                for i in range(0, len(new_query_records), batch_size_chroma):
                    batch = new_query_records[i : i + batch_size_chroma]
                    batch_embeddings = embeddings[i : i + batch_size_chroma]

                    query_ids = [q.id for q in batch]
                    texts = [q.query_text for q in batch]
                    metadata = [
                        {
                            "model": q.model,
                            "language": q.language or "unknown",
                            "conversation_id": q.conversation_id,
                        }
                        for q in batch
                    ]

                    chroma.add_queries_batch(
                        query_ids=query_ids,
                        texts=texts,
                        embeddings=batch_embeddings,
                        metadata=metadata,
                    )
                    progress.update(chroma_task, advance=len(batch))

                progress.update(chroma_task, completed=len(new_query_records))

            console.print("[green]ChromaDB updated successfully[/green]")

        console.print("\n[green]✓ Dataset summarization complete![/green]")
        console.print(f"  Source: {source_dataset} ({len(source_queries)} queries)")
        console.print(f"  Output: {output} ({len(new_queries)} queries)")
        console.print(f"  Model: {model}")
        console.print(f"  Prompt hash: {prompt_hash[:16]}...")

    finally:
        session.close()
