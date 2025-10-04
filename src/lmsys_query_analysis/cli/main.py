"""Main CLI entry point for LMSYS query analysis."""

import logging
import typer
from rich.console import Console
from rich.table import Table
from ..db.connection import get_db
from ..db.chroma import get_chroma
from ..db.loader import load_lmsys_dataset
from ..utils.logging import setup_logging

app = typer.Typer(help="LMSYS Query Analysis CLI")
cluster_app = typer.Typer(help="Clustering commands")
console = Console()
logger = logging.getLogger("lmsys")


@app.callback()
def _configure(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging (DEBUG)."
    ),
):
    """Configure logging before executing a subcommand."""
    setup_logging(verbose=verbose)


@app.command()
def load(
    limit: int = typer.Option(None, help="Limit number of records to load"),
    db_path: str = typer.Option(
        None, help="Database path (default: ~/.lmsys-query-analysis/queries.db)"
    ),
    use_chroma: bool = typer.Option(False, help="Enable ChromaDB for semantic search"),
    chroma_path: str = typer.Option(
        None, help="ChromaDB path (default: ~/.lmsys-query-analysis/chroma)"
    ),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", help="Embedding model for ChromaDB"
    ),
    embedding_provider: str = typer.Option(
        "openai", help="Embedding provider: 'openai' or 'sentence-transformers'"
    ),
    db_batch_size: int = typer.Option(5000, help="DB insert batch size"),
    streaming: bool = typer.Option(False, help="Use streaming dataset iteration"),
    no_pragmas: bool = typer.Option(
        False, help="Disable SQLite PRAGMA speedups during load"
    ),
    force_reload: bool = typer.Option(
        False, "--force-reload", help="Reload existing queries (skip duplicate check)"
    ),
):
    """Download and load LMSYS-1M dataset into SQLite."""
    try:
        logger.info(
            "Starting data load: limit=%s, use_chroma=%s, force_reload=%s",
            limit,
            use_chroma,
            force_reload,
        )
        db = get_db(db_path)
        chroma = get_chroma(chroma_path) if use_chroma else None

        stats = load_lmsys_dataset(
            db,
            limit=limit,
            skip_existing=not force_reload,
            chroma=chroma,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            batch_size=db_batch_size,
            use_streaming=streaming,
            apply_pragmas=not no_pragmas,
        )

        # Display results
        table = Table(title="Loading Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Total Processed", str(stats["total_processed"]))
        table.add_row("Loaded", str(stats["loaded"]))
        table.add_row("Skipped", str(stats["skipped"]))
        table.add_row("Errors", str(stats["errors"]))

        console.print(table)
        logger.info("Database path: %s", db.db_path)
        if use_chroma and chroma:
            logger.info("Chroma path: %s", chroma.persist_directory)
            logger.info("Chroma queries total: %s", chroma.count_queries())

    except Exception as e:
        logger.exception("Load failed: %s", e)
        raise typer.Exit(1)


@cluster_app.command("kmeans")
def cluster_kmeans(
    n_clusters: int = typer.Option(200, help="Number of clusters"),
    description: str = typer.Option("", help="Description of this clustering run"),
    db_path: str = typer.Option(None, help="Database path"),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", help="Embedding model"
    ),
    embedding_provider: str = typer.Option(
        "openai", help="'sentence-transformers' or 'openai'"
    ),
    embed_batch_size: int = typer.Option(32, help="Embedding encode batch size"),
    chunk_size: int = typer.Option(5000, help="DB iteration chunk size"),
    mb_batch_size: int = typer.Option(4096, help="MiniBatchKMeans batch_size"),
    use_chroma: bool = typer.Option(False, help="Enable ChromaDB"),
    chroma_path: str = typer.Option(None, help="ChromaDB path"),
):
    """Run MiniBatchKMeans clustering."""
    try:
        from ..clustering.kmeans import run_kmeans_clustering

        db = get_db(db_path)
        chroma = get_chroma(chroma_path) if use_chroma else None

        logger.info(
            "Running clustering: algo=kmeans, n_clusters=%s, model=%s, provider=%s, use_chroma=%s",
            n_clusters,
            embedding_model,
            embedding_provider,
            use_chroma,
        )
        run_id = run_kmeans_clustering(
            db=db,
            n_clusters=n_clusters,
            description=description,
            embedding_model=embedding_model,
            embed_batch_size=embed_batch_size,
            chunk_size=chunk_size,
            mb_batch_size=mb_batch_size,
            embedding_provider=embedding_provider,
            chroma=chroma,
        )

        if run_id:
            logger.info("Clustering completed: run_id=%s", run_id)
            console.print(
                f"[cyan]Use 'lmsys inspect {run_id} <cluster_id>' to explore clusters[/cyan]"
            )
            if use_chroma and chroma:
                console.print(
                    f"[cyan]Use 'lmsys search --run-id {run_id} <query>' to search cluster summaries[/cyan]"
                )
    except Exception as e:
        logger.exception("KMeans clustering failed: %s", e)
        raise typer.Exit(1)


@cluster_app.command("hdbscan")
def cluster_hdbscan(
    description: str = typer.Option("", help="Description of this clustering run"),
    db_path: str = typer.Option(None, help="Database path"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model"),
    embedding_provider: str = typer.Option(
        "sentence-transformers", help="'sentence-transformers' or 'openai'"
    ),
    embed_batch_size: int = typer.Option(32, help="Embedding encode batch size"),
    chunk_size: int = typer.Option(5000, help="DB iteration chunk size"),
    min_cluster_size: int = typer.Option(25, help="HDBSCAN minimum cluster size"),
    min_samples: int = typer.Option(
        None, help="HDBSCAN min_samples (default: min_cluster_size)"
    ),
    epsilon: float = typer.Option(0.0, help="HDBSCAN cluster_selection_epsilon"),
    metric: str = typer.Option(
        "euclidean", help="Distance metric: euclidean or cosine"
    ),
    use_chroma: bool = typer.Option(False, help="Enable ChromaDB"),
    chroma_path: str = typer.Option(None, help="ChromaDB path"),
):
    """Run HDBSCAN clustering."""
    try:
        from ..clustering.hdbscan_clustering import run_hdbscan_clustering

        db = get_db(db_path)
        chroma = get_chroma(chroma_path) if use_chroma else None

        logger.info(
            "Running clustering: algo=hdbscan, model=%s, provider=%s, use_chroma=%s",
            embedding_model,
            embedding_provider,
            use_chroma,
        )
        run_id = run_hdbscan_clustering(
            db=db,
            description=description,
            embedding_model=embedding_model,
            embed_batch_size=embed_batch_size,
            chunk_size=chunk_size,
            embedding_provider=embedding_provider,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon,
            metric=metric,
            chroma=chroma,
        )

        if run_id:
            logger.info("Clustering completed: run_id=%s", run_id)
            console.print(
                f"[cyan]Use 'lmsys inspect {run_id} <cluster_id>' to explore clusters[/cyan]"
            )
            if use_chroma and chroma:
                console.print(
                    f"[cyan]Use 'lmsys search --run-id {run_id} <query>' to search cluster summaries[/cyan]"
                )
    except Exception as e:
        logger.exception("HDBSCAN clustering failed: %s", e)
        raise typer.Exit(1)


app.add_typer(cluster_app, name="cluster")


@app.command()
def list(
    run_id: str = typer.Option(None, help="Filter by run ID"),
    cluster_id: int = typer.Option(None, help="Filter by cluster ID"),
    model: str = typer.Option(None, help="Filter by model name"),
    limit: int = typer.Option(50, help="Number of queries to display"),
    db_path: str = typer.Option(None, help="Database path"),
):
    """List queries with optional filtering by run_id and cluster_id."""
    try:
        from ..db.models import Query, QueryCluster
        from sqlmodel import select

        db = get_db(db_path)
        session = db.get_session()

        try:
            # Build query with filters
            if run_id and cluster_id:
                # Filter by run and cluster
                statement = (
                    select(Query)
                    .join(QueryCluster, Query.id == QueryCluster.query_id)
                    .where(QueryCluster.run_id == run_id)
                    .where(QueryCluster.cluster_id == cluster_id)
                    .limit(limit)
                )
            elif run_id:
                # Filter by run only
                statement = (
                    select(Query)
                    .join(QueryCluster, Query.id == QueryCluster.query_id)
                    .where(QueryCluster.run_id == run_id)
                    .limit(limit)
                )
            else:
                # No run filter, just list queries
                statement = select(Query).limit(limit)
                if model:
                    statement = statement.where(Query.model == model)

            queries = session.exec(statement).all()

            if not queries:
                logger.warning("No queries found for given filters")
                return

            table = Table(title=f"Queries ({len(queries)} shown)")
            table.add_column("ID", style="cyan", width=8)
            table.add_column("Model", style="yellow", width=20)
            table.add_column("Query", style="white", width=80)
            table.add_column("Language", style="green", width=10)

            for query in queries:
                table.add_row(
                    str(query.id),
                    query.model[:20] if query.model else "unknown",
                    (query.query_text[:77] + "...")
                    if len(query.query_text) > 80
                    else query.query_text,
                    query.language or "?",
                )

            console.print(table)

        finally:
            session.close()

    except Exception as e:
        logger.exception("List failed: %s", e)
        raise typer.Exit(1)


@app.command()
def runs(
    db_path: str = typer.Option(None, help="Database path"),
    latest: bool = typer.Option(
        False, "--latest", help="Show only the most recent run"
    ),
):
    """List all clustering runs."""
    try:
        from ..db.models import ClusteringRun
        from sqlmodel import select

        db = get_db(db_path)
        session = db.get_session()

        try:
            statement = select(ClusteringRun).order_by(ClusteringRun.created_at.desc())

            if latest:
                statement = statement.limit(1)

            runs = session.exec(statement).all()

            if not runs:
                logger.warning("No clustering runs found")
                return

            title = "Latest Clustering Run" if latest else "Clustering Runs"
            table = Table(title=title)
            table.add_column("Run ID", style="cyan", no_wrap=True)
            table.add_column("Algorithm", style="yellow")
            table.add_column("Clusters", style="green")
            table.add_column("Created", style="magenta")
            table.add_column("Description", style="white")

            for run in runs:
                table.add_row(
                    run.run_id,
                    run.algorithm,
                    str(run.num_clusters) if run.num_clusters else "?",
                    run.created_at.strftime("%Y-%m-%d %H:%M")
                    if run.created_at
                    else "?",
                    run.description[:50] if run.description else "",
                )

            console.print(table)

        finally:
            session.close()

    except Exception as e:
        logger.exception("Runs failed: %s", e)
        raise typer.Exit(1)


@app.command()
def list_clusters(
    run_id: str = typer.Argument(..., help="Run ID to list clusters for"),
    db_path: str = typer.Option(None, help="Database path"),
    summary_run_id: str = typer.Option(None, help="Filter by specific summary run ID"),
    alias: str = typer.Option(None, help="Filter by summary alias (e.g., 'claude-v1')"),
    limit: int = typer.Option(None, help="Limit number of clusters to show"),
    show_examples: int = typer.Option(
        0, help="Show up to N example queries per cluster"
    ),
    example_width: int = typer.Option(80, help="Max characters per example query"),
):
    """List all clusters for a run with their titles and descriptions."""
    try:
        from ..db.models import ClusterSummary
        from sqlmodel import select

        db = get_db(db_path)
        session = db.get_session()

        try:
            statement = (
                select(ClusterSummary)
                .where(ClusterSummary.run_id == run_id)
                .order_by(
                    ClusterSummary.num_queries.desc(), ClusterSummary.cluster_id.asc()
                )
            )

            # Filter by summary_run_id or alias if provided
            if summary_run_id:
                statement = statement.where(ClusterSummary.summary_run_id == summary_run_id)
            elif alias:
                statement = statement.where(ClusterSummary.alias == alias)
            else:
                # If no summary_run_id or alias specified, get the most recent one
                from sqlalchemy import func
                latest_stmt = (
                    select(ClusterSummary.summary_run_id)
                    .where(ClusterSummary.run_id == run_id)
                    .order_by(ClusterSummary.generated_at.desc())
                    .limit(1)
                )
                latest_summary_run = session.exec(latest_stmt).first()
                if latest_summary_run:
                    statement = statement.where(ClusterSummary.summary_run_id == latest_summary_run)

            if limit:
                statement = statement.limit(limit)

            summaries = session.exec(statement).all()

            if not summaries:
                logger.warning("No summaries found for run %s", run_id)
                console.print(
                    f"[cyan]Run 'lmsys summarize {run_id}' to generate summaries[/cyan]"
                )
                return

            table = Table(title=f"Clusters for Run: {run_id}")
            table.add_column("Cluster", style="cyan", width=8)
            table.add_column("Title", style="yellow", width=40)
            table.add_column("Queries", style="green", width=8)
            table.add_column("Description", style="white", width=60)
            if show_examples and show_examples > 0:
                table.add_column("Examples", style="white", width=example_width + 6)

            for summary in summaries:
                row = [
                    str(summary.cluster_id),
                    summary.title or "No title",
                    str(summary.num_queries) if summary.num_queries else "?",
                    (summary.description[:57] + "...")
                    if summary.description and len(summary.description) > 60
                    else (summary.description or "No description"),
                ]

                if show_examples and show_examples > 0:
                    reps = summary.representative_queries or []
                    examples = reps[:show_examples]
                    # Trim each example and keep first line only for compactness
                    formatted = []
                    for ex in examples:
                        ex = ex.splitlines()[0].strip()
                        if len(ex) > example_width:
                            ex = ex[: example_width - 3] + "..."
                        formatted.append("- " + ex)
                    row.append("\n".join(formatted) if formatted else "")

                table.add_row(*row)

            console.print(table)
            # Ensure examples are visible even if the table drops/wraps columns in narrow terminals
            if show_examples and show_examples > 0:
                console.print("\n[bold cyan]Examples per cluster[/bold cyan]")
                for summary in summaries:
                    reps = summary.representative_queries or []
                    if not reps:
                        continue
                    console.print(
                        f"[yellow]Cluster {summary.cluster_id}[/yellow] — {summary.title or ''}"
                    )
                    for ex in reps[:show_examples]:
                        ex_line = ex.splitlines()[0].strip()
                        if len(ex_line) > example_width:
                            ex_line = ex_line[: example_width - 3] + "..."
                        console.print(f"  - {ex_line}")

            console.print(f"\n[cyan]Total: {len(summaries)} clusters[/cyan]")

        finally:
            session.close()

    except Exception as e:
        logger.exception("List-clusters failed: %s", e)
        raise typer.Exit(1)


@app.command()
def summarize(
    run_id: str = typer.Argument(..., help="Run ID to summarize"),
    cluster_id: int = typer.Option(None, help="Specific cluster to summarize"),
    model: str = typer.Option(
        "openai/gpt-4o-mini", help="LLM model (provider/model)"
    ),
    max_queries: int = typer.Option(100, help="Max queries to send to LLM per cluster"),
    concurrency: int = typer.Option(30, help="Parallel LLM calls for summarization"),
    rpm: int = typer.Option(None, help="Optional requests-per-minute rate limit"),
    use_chroma: bool = typer.Option(False, help="Update ChromaDB with new summaries"),
    contrast_neighbors: int = typer.Option(
        2, help="Number of nearest neighbor clusters to include for contrast"
    ),
    contrast_examples: int = typer.Option(
        2, help="Examples per neighbor cluster to include for contrast"
    ),
    contrast_mode: str = typer.Option(
        "neighbors", help="Contrast mode: 'neighbors' (examples) or 'keywords'"
    ),
    summary_run_id: str = typer.Option(
        None, help="Custom summary run ID (auto-generated if not provided)"
    ),
    alias: str = typer.Option(
        None, help="Friendly alias for this summary run (e.g., 'claude-v1', 'gpt4-best')"
    ),
    db_path: str = typer.Option(None, help="Database path"),
    chroma_path: str = typer.Option(None, help="ChromaDB path"),
):
    """Generate LLM-powered titles and descriptions for clusters."""
    try:
        from datetime import datetime
        from ..clustering.summarizer import ClusterSummarizer
        from ..db.models import ClusterSummary, QueryCluster, Query, ClusteringRun
        from sqlmodel import select

        db = get_db(db_path)

        console.print(f"[cyan]Using LLM: {model}[/cyan]")

        session = db.get_session()

        try:
            # Get the clustering run to check it exists
            run_statement = select(ClusteringRun).where(ClusteringRun.run_id == run_id)
            run = session.exec(run_statement).first()
            if not run:
                console.print(f"[red]Run {run_id} not found[/red]")
                raise typer.Exit(1)

            # Generate summary_run_id if not provided
            if not summary_run_id:
                # Format: summary-{model_short}-{timestamp}
                model_short = model.split("/")[-1][:20]  # Take last part of model name
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                summary_run_id = f"summary-{model_short}-{timestamp}"

            console.print(f"[cyan]Summary run ID: {summary_run_id}[/cyan]")
            if alias:
                console.print(f"[cyan]Alias: {alias}[/cyan]")
            logger.info("Starting summarization run: %s (alias: %s)", summary_run_id, alias or "none")

            # Initialize summarizer (no embedding model needed)
            summarizer = ClusterSummarizer(
                model=model,
                concurrency=concurrency,
                rpm=rpm,
            )

            # Get clusters to summarize
            if cluster_id is not None:
                cluster_ids = [cluster_id]
            else:
                # Get all clusters for this run
                statement = (
                    select(QueryCluster.cluster_id)
                    .where(QueryCluster.run_id == run_id)
                    .distinct()
                )
                cluster_ids = [row for row in session.exec(statement)]

            logger.info("Summarizing %s clusters for run %s", len(cluster_ids), run_id)

            # Prepare clusters data
            clusters_data = []
            for cid in cluster_ids:
                # Get all queries in this cluster
                statement = (
                    select(Query)
                    .join(QueryCluster, Query.id == QueryCluster.query_id)
                    .where(QueryCluster.run_id == run_id)
                    .where(QueryCluster.cluster_id == cid)
                )
                queries = session.exec(statement).all()
                query_texts = [q.query_text for q in queries]
                clusters_data.append((cid, query_texts))

            # Generate summaries with LLM
            results = summarizer.generate_batch_summaries(
                clusters_data=clusters_data,
                max_queries=max_queries,
                concurrency=concurrency,
                rpm=rpm,
                contrast_neighbors=contrast_neighbors,
                contrast_examples=contrast_examples,
                contrast_mode=contrast_mode,
            )

            # Store in SQLite
            logger.info("Storing summaries in SQLite...")
            sizes_map = {cid: len(qs) for cid, qs in clusters_data}

            # Prepare summarization parameters for storage
            summary_params = {
                "max_queries": max_queries,
                "concurrency": concurrency,
                "rpm": rpm,
                "contrast_neighbors": contrast_neighbors,
                "contrast_examples": contrast_examples,
                "contrast_mode": contrast_mode,
            }

            for cid, summary_data in results.items():
                # Check if summary already exists for this summary_run_id
                statement = select(ClusterSummary).where(
                    ClusterSummary.run_id == run_id,
                    ClusterSummary.cluster_id == cid,
                    ClusterSummary.summary_run_id == summary_run_id,
                )
                existing = session.exec(statement).first()

                if existing:
                    # Update existing
                    existing.title = summary_data["title"]
                    existing.description = summary_data["description"]
                    existing.summary = (
                        f"{summary_data['title']}\n\n{summary_data['description']}"
                    )
                    existing.representative_queries = [
                        q[:200] for q in summary_data["sample_queries"]
                    ]
                    existing.model = model
                    existing.parameters = summary_params
                    existing.alias = alias
                else:
                    # Create new
                    new_summary = ClusterSummary(
                        run_id=run_id,
                        cluster_id=cid,
                        summary_run_id=summary_run_id,
                        alias=alias,
                        title=summary_data["title"],
                        description=summary_data["description"],
                        summary=f"{summary_data['title']}\n\n{summary_data['description']}",
                        num_queries=sizes_map.get(cid, 0),
                        representative_queries=[
                            q for q in summary_data["sample_queries"]
                        ],
                        model=model,
                        parameters=summary_params,
                    )
                    session.add(new_summary)

            session.commit()

            # Display generated summaries
            logger.info("Generated summaries for %s clusters", len(results))
            console.print(f"\n[bold green]✓ Generated {len(results)} cluster summaries[/bold green]\n")

            for cid in sorted(results.keys()):
                summary_data = results[cid]
                console.print(f"[bold cyan]Cluster {cid}[/bold cyan]: {summary_data['title']}")
                console.print(f"  {summary_data['description']}")
                console.print(f"  [dim]({sizes_map.get(cid, 0)} queries)[/dim]\n")

            # Store in ChromaDB if requested
            if use_chroma:
                from ..db.chroma import get_chroma
                from ..clustering.embeddings import EmbeddingGenerator

                logger.info("Storing summaries in ChromaDB...")
                console.print("[cyan]Generating embeddings for summaries...[/cyan]")

                chroma = get_chroma(chroma_path)

                # Initialize embedding generator (use same defaults as load command)
                embedding_gen = EmbeddingGenerator(
                    model_name="text-embedding-3-small",
                    provider="openai",
                )

                # Prepare data for batch storage
                cluster_ids_list = []
                summaries_list = []
                titles_list = []
                descriptions_list = []
                metadata_list = []

                for cid, summary_data in results.items():
                    cluster_ids_list.append(cid)
                    # Combine title + description for the document text
                    summary_text = f"{summary_data['title']}\n\n{summary_data['description']}"
                    summaries_list.append(summary_text)
                    titles_list.append(summary_data["title"])
                    descriptions_list.append(summary_data["description"])
                    metadata_list.append({
                        "num_queries": sizes_map.get(cid, 0),
                        "model": model,
                    })

                # Generate embeddings for summaries
                embeddings = embedding_gen.generate_embeddings(
                    summaries_list,
                    batch_size=32,
                    show_progress=True,
                )

                # Store in ChromaDB
                chroma.add_cluster_summaries_batch(
                    run_id=run_id,
                    cluster_ids=cluster_ids_list,
                    summaries=summaries_list,
                    embeddings=embeddings,
                    metadata_list=metadata_list,
                    titles=titles_list,
                    descriptions=descriptions_list,
                )

                console.print(
                    f"[green]Stored {len(cluster_ids_list)} summaries in ChromaDB[/green]"
                )

            console.print(
                f"\n[cyan]Use 'lmsys list-clusters {run_id}' to view all cluster titles[/cyan]"
            )

        finally:
            session.close()

    except Exception as e:
        logger.exception("Summarize failed: %s", e)
        import traceback

        # For ExceptionGroup, show all sub-exceptions
        if hasattr(e, 'exceptions'):
            console.print("[red]Multiple errors occurred:[/red]")
            for sub_exc in e.exceptions:
                console.print(f"[yellow]{type(sub_exc).__name__}: {sub_exc}[/yellow]")
                traceback.print_exception(type(sub_exc), sub_exc, sub_exc.__traceback__)
        else:
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def inspect(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    cluster_id: int = typer.Argument(..., help="Cluster ID to inspect"),
    db_path: str = typer.Option(None, help="Database path"),
    show_queries: int = typer.Option(10, help="Number of queries to show"),
):
    """Inspect specific cluster in detail."""
    try:
        from ..db.models import Query, QueryCluster, ClusterSummary
        from sqlmodel import select

        db = get_db(db_path)
        session = db.get_session()

        try:
            # Get cluster summary if exists
            statement = select(ClusterSummary).where(
                ClusterSummary.run_id == run_id, ClusterSummary.cluster_id == cluster_id
            )
            summary = session.exec(statement).first()

            # Get all queries in this cluster
            statement = (
                select(Query)
                .join(QueryCluster, Query.id == QueryCluster.query_id)
                .where(QueryCluster.run_id == run_id)
                .where(QueryCluster.cluster_id == cluster_id)
            )
            queries = session.exec(statement).all()

            if not queries:
                logger.warning("No queries found in cluster %s", cluster_id)
                return

            # Display cluster info
            console.print(f"\n[cyan]{'=' * 80}[/cyan]")
            console.print(
                f"[bold cyan]Cluster {cluster_id} from run {run_id}[/bold cyan]"
            )
            console.print(f"[cyan]{'=' * 80}[/cyan]\n")

            if summary:
                console.print(
                    f"[bold yellow]Title:[/bold yellow] {summary.title or 'N/A'}"
                )
                console.print(f"\n[bold yellow]Description:[/bold yellow]")
                console.print(f"{summary.description or 'N/A'}\n")

            console.print(f"[bold green]Total Queries:[/bold green] {len(queries)}\n")

            # Show sample queries
            console.print(
                f"[bold yellow]Sample Queries (showing {min(show_queries, len(queries))}):[/bold yellow]\n"
            )

            for i, query in enumerate(queries[:show_queries], 1):
                console.print(f"[cyan]{i}.[/cyan] {query.query_text}")
                console.print(
                    f"   [dim]Model: {query.model} | Language: {query.language or 'unknown'}[/dim]\n"
                )

            if len(queries) > show_queries:
                console.print(
                    f"[dim]... and {len(queries) - show_queries} more queries[/dim]\n"
                )

            console.print(
                f"[cyan]Use 'lmsys list --run-id {run_id} --cluster-id {cluster_id}' to see all queries[/cyan]"
            )

        finally:
            session.close()

    except Exception as e:
        logger.exception("Inspect failed: %s", e)
        raise typer.Exit(1)


@app.command()
def export(
    run_id: str = typer.Argument(..., help="Run ID to export"),
    output: str = typer.Option("export.csv", help="Output file path"),
    format: str = typer.Option("csv", help="Export format: csv or json"),
    db_path: str = typer.Option(None, help="Database path"),
):
    """Export cluster results to file."""
    try:
        import csv
        import json
        from ..db.models import Query, QueryCluster, ClusterSummary
        from sqlmodel import select

        db = get_db(db_path)
        session = db.get_session()

        try:
            # Get all query-cluster mappings
            statement = (
                select(Query, QueryCluster, ClusterSummary)
                .join(QueryCluster, Query.id == QueryCluster.query_id)
                .outerjoin(
                    ClusterSummary,
                    (ClusterSummary.run_id == QueryCluster.run_id)
                    & (ClusterSummary.cluster_id == QueryCluster.cluster_id),
                )
                .where(QueryCluster.run_id == run_id)
            )
            results = session.exec(statement).all()

            if not results:
                logger.warning("No data found for run %s", run_id)
                return

            console.print(f"[cyan]Exporting {len(results)} queries...[/cyan]")

            if format == "csv":
                with open(output, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "query_id",
                            "cluster_id",
                            "query_text",
                            "model",
                            "language",
                            "cluster_title",
                            "cluster_description",
                        ]
                    )

                    for query, qc, summary in results:
                        writer.writerow(
                            [
                                query.id,
                                qc.cluster_id,
                                query.query_text,
                                query.model,
                                query.language or "",
                                summary.title if summary else "",
                                summary.description if summary else "",
                            ]
                        )

            elif format == "json":
                data = []
                for query, qc, summary in results:
                    data.append(
                        {
                            "query_id": query.id,
                            "cluster_id": qc.cluster_id,
                            "query_text": query.query_text,
                            "model": query.model,
                            "language": query.language,
                            "cluster_title": summary.title if summary else None,
                            "cluster_description": summary.description
                            if summary
                            else None,
                        }
                    )

                with open(output, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            else:
                console.print(f"[red]Unsupported format: {format}[/red]")
                raise typer.Exit(1)

            logger.info("Exported to %s", output)

        finally:
            session.close()

    except Exception as e:
        logger.exception("Export failed: %s", e)
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    search_type: str = typer.Option(
        "queries", help="Search type: 'queries' or 'clusters'"
    ),
    run_id: str = typer.Option(None, help="Filter cluster search by run_id"),
    n_results: int = typer.Option(10, help="Number of results to return"),
    chroma_path: str = typer.Option(
        None, help="ChromaDB path (default: ~/.lmsys-query-analysis/chroma)"
    ),
    embedding_model: str = typer.Option(
        "all-MiniLM-L6-v2", help="Embedding model to embed the query for search"
    ),
):
    """Semantic search across queries or cluster summaries using ChromaDB."""
    try:
        chroma = get_chroma(chroma_path)

        if search_type == "queries":
            logger.info("Searching queries: n_results=%s", n_results)
            # Explicitly embed query to ensure consistency with stored vectors
            from ..clustering.embeddings import EmbeddingGenerator

            eg = EmbeddingGenerator(model_name=embedding_model)
            eg.load_model()
            q_emb = eg.model.encode(
                [query], batch_size=1, show_progress_bar=False, convert_to_numpy=True
            )[0]
            results = chroma.search_queries(
                query, n_results=n_results, query_embedding=q_emb
            )

            if results and results["ids"] and len(results["ids"][0]) > 0:
                table = Table(title=f"Top {len(results['ids'][0])} Similar Queries")
                table.add_column("Rank", style="cyan", width=6)
                table.add_column("Query ID", style="yellow", width=12)
                table.add_column("Query Text", style="white", width=60)
                table.add_column("Model", style="green", width=15)
                table.add_column("Distance", style="magenta", width=10)

                for rank, (qid, doc, meta, dist) in enumerate(
                    zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    ),
                    1,
                ):
                    # Extract query_id from "query_123" format
                    query_id = qid.split("_")[1] if "_" in qid else qid
                    table.add_row(
                        str(rank),
                        query_id,
                        doc[:60] + "..." if len(doc) > 60 else doc,
                        meta.get("model", "unknown"),
                        f"{dist:.4f}",
                    )

                console.print(table)
            else:
                console.print("[yellow]No results found[/yellow]")

        elif search_type == "clusters":
            logger.info(
                "Searching cluster summaries: run_id=%s, n_results=%s",
                run_id,
                n_results,
            )
            # Use the run's embedding model if available
            from ..db.models import ClusteringRun
            from sqlmodel import select

            search_model = embedding_model
            if run_id:
                db = get_db(None)
                with db.get_session() as s:
                    run = s.exec(
                        select(ClusteringRun).where(ClusteringRun.run_id == run_id)
                    ).first()
                    if run and run.parameters and run.parameters.get("embedding_model"):
                        search_model = run.parameters["embedding_model"]

            from ..clustering.embeddings import EmbeddingGenerator

            eg = EmbeddingGenerator(model_name=search_model)
            eg.load_model()
            q_emb = eg.model.encode(
                [query], batch_size=1, show_progress_bar=False, convert_to_numpy=True
            )[0]
            results = chroma.search_cluster_summaries(
                query, run_id=run_id, n_results=n_results, query_embedding=q_emb
            )

            if results and results["ids"] and len(results["ids"][0]) > 0:
                table = Table(title=f"Top {len(results['ids'][0])} Similar Clusters")
                table.add_column("Rank", style="cyan", width=6)
                table.add_column("Run ID", style="yellow", width=25)
                table.add_column("Cluster", style="green", width=10)
                table.add_column("Summary", style="white", width=50)
                table.add_column("Distance", style="magenta", width=10)

                for rank, (cid, doc, meta, dist) in enumerate(
                    zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    ),
                    1,
                ):
                    table.add_row(
                        str(rank),
                        meta.get("run_id", "unknown"),
                        str(meta.get("cluster_id", "?")),
                        doc[:50] + "..." if len(doc) > 50 else doc,
                        f"{dist:.4f}",
                    )

                console.print(table)
            else:
                console.print("[yellow]No results found[/yellow]")

        else:
            logger.error("Invalid search_type: %s", search_type)
            raise typer.Exit(1)

    except Exception as e:
        logger.exception("Search failed: %s", e)
        console.print(
            "[yellow]Make sure ChromaDB is initialized by running 'lmsys load --use-chroma' first[/yellow]"
        )
        raise typer.Exit(1)


@app.command()
def clear(
    db_path: str = typer.Option(None, help="Database path"),
    chroma_path: str = typer.Option(None, help="ChromaDB path"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Clear all data from SQLite database and ChromaDB."""
    import shutil
    from pathlib import Path

    try:
        db = get_db(db_path)
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

    except Exception as e:
        logger.exception("Clear failed: %s", e)
        raise typer.Exit(1)


@app.command("backfill-chroma")
def backfill_chroma(
    db_path: str = typer.Option(None, help="Database path"),
    chroma_path: str = typer.Option(None, help="ChromaDB path"),
    embedding_provider: str = typer.Option(
        "openai", help="Embedding provider: 'openai' or 'sentence-transformers'"
    ),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", help="Embedding model to use"
    ),
    chunk_size: int = typer.Option(5000, help="DB iteration chunk size"),
    embed_batch_size: int = typer.Option(64, help="Embedding batch size"),
):
    """Backfill embeddings into ChromaDB for SQLite queries missing vectors."""
    try:
        import time
        from ..db.models import Query
        from sqlmodel import select
        from sqlalchemy import func
        from ..clustering.embeddings import EmbeddingGenerator
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
        )

        db = get_db(db_path)
        chroma = get_chroma(chroma_path)

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

            eg = EmbeddingGenerator(
                model_name=embedding_model, provider=embedding_provider
            )
            # Load local model early if sentence-transformers
            if embedding_provider == "sentence-transformers":
                eg.load_model()

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
            summary = Table(title="Backfill Summary")
            summary.add_column("Metric", style="cyan")
            summary.add_column("Count", style="green")
            summary.add_row("Scanned", str(scanned))
            summary.add_row("Backfilled", str(backfilled))
            summary.add_row("Already Present", str(already_present))
            summary.add_row("Elapsed (s)", f"{elapsed:.2f}")
            summary.add_row("Avg Rate (/s)", f"{rate:.1f}")
            console.print(summary)

    except Exception as e:
        logger.exception("Backfill failed: %s", e)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
