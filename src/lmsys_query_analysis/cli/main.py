"""Main CLI entry point for LMSYS query analysis."""
import typer
from rich.console import Console
from rich.table import Table
from ..db.connection import get_db
from ..db.chroma import get_chroma
from ..db.loader import load_lmsys_dataset

app = typer.Typer(help="LMSYS Query Analysis CLI")
console = Console()


@app.command()
def load(
    limit: int = typer.Option(None, help="Limit number of records to load"),
    db_path: str = typer.Option(None, help="Database path (default: ~/.lmsys-query-analysis/queries.db)"),
    use_chroma: bool = typer.Option(False, help="Enable ChromaDB for semantic search"),
    chroma_path: str = typer.Option(None, help="ChromaDB path (default: ~/.lmsys-query-analysis/chroma)"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model for ChromaDB"),
):
    """Download and load LMSYS-1M dataset into SQLite."""
    try:
        db = get_db(db_path)
        chroma = get_chroma(chroma_path) if use_chroma else None

        stats = load_lmsys_dataset(
            db,
            limit=limit,
            chroma=chroma,
            embedding_model=embedding_model,
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
        console.print(f"\n[green]Database: {db.db_path}[/green]")
        if use_chroma and chroma:
            console.print(f"[green]ChromaDB: {chroma.persist_directory}[/green]")
            console.print(f"[green]Total vectors in ChromaDB: {chroma.count_queries()}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def cluster(
    algorithm: str = typer.Option("kmeans", help="Clustering algorithm (kmeans, hdbscan)"),
    n_clusters: int = typer.Option(200, help="Number of clusters for kmeans"),
    description: str = typer.Option("", help="Description of this clustering run"),
    db_path: str = typer.Option(None, help="Database path (default: ~/.lmsys-query-analysis/queries.db)"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", help="Sentence transformer model"),
    use_chroma: bool = typer.Option(False, help="Enable ChromaDB for cluster summaries"),
    chroma_path: str = typer.Option(None, help="ChromaDB path (default: ~/.lmsys-query-analysis/chroma)"),
):
    """Run clustering analysis on queries."""
    if algorithm.lower() != "kmeans":
        console.print(f"[red]Only 'kmeans' algorithm is currently supported[/red]")
        raise typer.Exit(1)

    try:
        from ..clustering.kmeans import run_kmeans_clustering

        db = get_db(db_path)
        chroma = get_chroma(chroma_path) if use_chroma else None

        run_id = run_kmeans_clustering(
            db=db,
            n_clusters=n_clusters,
            description=description,
            embedding_model=embedding_model,
            chroma=chroma,
        )

        if run_id:
            console.print(f"\n[green]Run ID: {run_id}[/green]")
            console.print(f"[cyan]Use 'lmsys inspect {run_id} <cluster_id>' to explore clusters[/cyan]")
            if use_chroma and chroma:
                console.print(f"[cyan]Use 'lmsys search --run-id {run_id} <query>' to search cluster summaries[/cyan]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    run_id: str = typer.Option(None, help="Filter by run ID"),
    topic_id: int = typer.Option(None, help="Filter by topic/cluster ID"),
    limit: int = typer.Option(50, help="Number of queries to display"),
):
    """List queries with optional filtering by run_id and topic_id."""
    console.print("[yellow]Listing queries...[/yellow]")
    # TODO: Implement query listing
    console.print(f"[green]Showing {limit} queries[/green]")


@app.command()
def runs(
    db_path: str = typer.Option(None, help="Database path"),
):
    """List all clustering runs."""
    try:
        from ..db.models import ClusteringRun
        from sqlmodel import select

        db = get_db(db_path)
        session = db.get_session()

        try:
            statement = select(ClusteringRun)
            runs = session.exec(statement).all()

            if not runs:
                console.print("[yellow]No clustering runs found[/yellow]")
                return

            table = Table(title="Clustering Runs")
            table.add_column("Run ID", style="cyan")
            table.add_column("Algorithm", style="yellow")
            table.add_column("Clusters", style="green")
            table.add_column("Created", style="magenta")
            table.add_column("Description", style="white")

            for run in runs:
                table.add_row(
                    run.run_id,
                    run.algorithm,
                    str(run.num_clusters) if run.num_clusters else "?",
                    run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "?",
                    run.description[:50] if run.description else "",
                )

            console.print(table)

        finally:
            session.close()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_clusters(
    run_id: str = typer.Argument(..., help="Run ID to list clusters for"),
    db_path: str = typer.Option(None, help="Database path"),
    limit: int = typer.Option(None, help="Limit number of clusters to show"),
):
    """List all clusters for a run with their titles and descriptions."""
    try:
        from ..db.models import ClusterSummary
        from sqlmodel import select

        db = get_db(db_path)
        session = db.get_session()

        try:
            statement = select(ClusterSummary).where(ClusterSummary.run_id == run_id)
            if limit:
                statement = statement.limit(limit)

            summaries = session.exec(statement).all()

            if not summaries:
                console.print(f"[yellow]No summaries found for run {run_id}[/yellow]")
                console.print(f"[cyan]Run 'lmsys summarize {run_id} --use-chroma' to generate summaries[/cyan]")
                return

            table = Table(title=f"Clusters for Run: {run_id}")
            table.add_column("Cluster", style="cyan", width=8)
            table.add_column("Title", style="yellow", width=40)
            table.add_column("Queries", style="green", width=8)
            table.add_column("Description", style="white", width=60)

            for summary in summaries:
                table.add_row(
                    str(summary.cluster_id),
                    summary.title or "No title",
                    str(summary.num_queries) if summary.num_queries else "?",
                    (summary.description[:57] + "...") if summary.description and len(summary.description) > 60 else (summary.description or "No description"),
                )

            console.print(table)
            console.print(f"\n[cyan]Total: {len(summaries)} clusters[/cyan]")

        finally:
            session.close()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def summarize(
    run_id: str = typer.Argument(..., help="Run ID to summarize"),
    cluster_id: int = typer.Option(None, help="Specific cluster to summarize"),
    model: str = typer.Option("anthropic/claude-3-haiku-20240307", help="LLM model (provider/model)"),
    max_queries: int = typer.Option(50, help="Max queries to send to LLM per cluster"),
    use_chroma: bool = typer.Option(False, help="Update ChromaDB with new summaries"),
    db_path: str = typer.Option(None, help="Database path"),
    chroma_path: str = typer.Option(None, help="ChromaDB path"),
):
    """Generate LLM-powered titles and descriptions for clusters."""
    try:
        from ..clustering.summarizer import ClusterSummarizer
        from ..clustering.embeddings import EmbeddingGenerator
        from ..db.models import ClusterSummary, QueryCluster, Query
        from sqlmodel import select

        db = get_db(db_path)
        chroma = get_chroma(chroma_path) if use_chroma else None

        # Initialize summarizer
        summarizer = ClusterSummarizer(model=model)
        console.print(f"[cyan]Using LLM: {model}[/cyan]")

        session = db.get_session()

        try:
            # Get clusters to summarize
            if cluster_id is not None:
                cluster_ids = [cluster_id]
            else:
                # Get all clusters for this run
                statement = select(QueryCluster.cluster_id).where(
                    QueryCluster.run_id == run_id
                ).distinct()
                cluster_ids = [row for row in session.exec(statement)]

            console.print(f"[cyan]Summarizing {len(cluster_ids)} clusters for run {run_id}...[/cyan]")

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
            )

            # Store in SQLite
            console.print("[yellow]Storing summaries in SQLite...[/yellow]")
            for cid, summary_data in results.items():
                # Check if summary already exists
                statement = select(ClusterSummary).where(
                    ClusterSummary.run_id == run_id,
                    ClusterSummary.cluster_id == cid
                )
                existing = session.exec(statement).first()

                if existing:
                    # Update existing
                    existing.title = summary_data["title"]
                    existing.description = summary_data["description"]
                    existing.summary = f"{summary_data['title']}\n\n{summary_data['description']}"
                    existing.representative_queries = [q[:200] for q in summary_data["sample_queries"]]
                else:
                    # Create new
                    new_summary = ClusterSummary(
                        run_id=run_id,
                        cluster_id=cid,
                        title=summary_data["title"],
                        description=summary_data["description"],
                        summary=f"{summary_data['title']}\n\n{summary_data['description']}",
                        num_queries=len([q for c, q in clusters_data if c == cid][0]),
                        representative_queries=[q[:200] for q in summary_data["sample_queries"]],
                    )
                    session.add(new_summary)

            session.commit()

            # Update ChromaDB if requested
            if use_chroma and chroma:
                console.print("[yellow]Updating ChromaDB with new summaries...[/yellow]")

                # Generate embeddings for title + description
                embedding_gen = EmbeddingGenerator()
                texts = [f"{results[cid]['title']}\n\n{results[cid]['description']}" for cid in cluster_ids]
                embeddings = embedding_gen.generate_embeddings(texts, show_progress=False)

                # Update ChromaDB (will overwrite existing)
                titles = [results[cid]['title'] for cid in cluster_ids]
                descriptions = [results[cid]['description'] for cid in cluster_ids]
                summaries = texts
                metadata_list = [{"num_queries": len([q for c, q in clusters_data if c == cid][0])} for cid in cluster_ids]

                chroma.add_cluster_summaries_batch(
                    run_id=run_id,
                    cluster_ids=cluster_ids,
                    summaries=summaries,
                    embeddings=embeddings,
                    metadata_list=metadata_list,
                    titles=titles,
                    descriptions=descriptions,
                )

                console.print(f"[green]Updated {len(cluster_ids)} summaries in ChromaDB[/green]")

            console.print(f"\n[green]Generated summaries for {len(results)} clusters![/green]")
            console.print(f"[cyan]Use 'lmsys list-clusters {run_id}' to view titles[/cyan]")

        finally:
            session.close()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def inspect(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    cluster_id: int = typer.Argument(..., help="Cluster ID to inspect"),
):
    """Inspect specific cluster in detail."""
    console.print(f"[yellow]Inspecting cluster {cluster_id} from run {run_id}...[/yellow]")
    # TODO: Implement cluster inspection
    console.print("[green]Done![/green]")


@app.command()
def export(
    run_id: str = typer.Argument(..., help="Run ID to export"),
    output: str = typer.Option("export.csv", help="Output file path"),
):
    """Export cluster results to file."""
    console.print(f"[yellow]Exporting run {run_id} to {output}...[/yellow]")
    # TODO: Implement export
    console.print("[green]Export complete![/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    search_type: str = typer.Option("queries", help="Search type: 'queries' or 'clusters'"),
    run_id: str = typer.Option(None, help="Filter cluster search by run_id"),
    n_results: int = typer.Option(10, help="Number of results to return"),
    chroma_path: str = typer.Option(None, help="ChromaDB path (default: ~/.lmsys-query-analysis/chroma)"),
):
    """Semantic search across queries or cluster summaries using ChromaDB."""
    try:
        chroma = get_chroma(chroma_path)

        if search_type == "queries":
            console.print(f"[cyan]Searching for similar queries...[/cyan]")
            results = chroma.search_queries(query, n_results=n_results)

            if results and results["ids"] and len(results["ids"][0]) > 0:
                table = Table(title=f"Top {len(results['ids'][0])} Similar Queries")
                table.add_column("Rank", style="cyan", width=6)
                table.add_column("Query ID", style="yellow", width=12)
                table.add_column("Query Text", style="white", width=60)
                table.add_column("Model", style="green", width=15)
                table.add_column("Distance", style="magenta", width=10)

                for rank, (qid, doc, meta, dist) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ), 1):
                    # Extract query_id from "query_123" format
                    query_id = qid.split("_")[1] if "_" in qid else qid
                    table.add_row(
                        str(rank),
                        query_id,
                        doc[:60] + "..." if len(doc) > 60 else doc,
                        meta.get("model", "unknown"),
                        f"{dist:.4f}"
                    )

                console.print(table)
            else:
                console.print("[yellow]No results found[/yellow]")

        elif search_type == "clusters":
            console.print(f"[cyan]Searching cluster summaries{f' for run {run_id}' if run_id else ''}...[/cyan]")
            results = chroma.search_cluster_summaries(query, run_id=run_id, n_results=n_results)

            if results and results["ids"] and len(results["ids"][0]) > 0:
                table = Table(title=f"Top {len(results['ids'][0])} Similar Clusters")
                table.add_column("Rank", style="cyan", width=6)
                table.add_column("Run ID", style="yellow", width=25)
                table.add_column("Cluster", style="green", width=10)
                table.add_column("Summary", style="white", width=50)
                table.add_column("Distance", style="magenta", width=10)

                for rank, (cid, doc, meta, dist) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ), 1):
                    table.add_row(
                        str(rank),
                        meta.get("run_id", "unknown"),
                        str(meta.get("cluster_id", "?")),
                        doc[:50] + "..." if len(doc) > 50 else doc,
                        f"{dist:.4f}"
                    )

                console.print(table)
            else:
                console.print("[yellow]No results found[/yellow]")

        else:
            console.print(f"[red]Invalid search_type: {search_type}. Use 'queries' or 'clusters'[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure ChromaDB is initialized by running 'lmsys load --use-chroma' first[/yellow]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
