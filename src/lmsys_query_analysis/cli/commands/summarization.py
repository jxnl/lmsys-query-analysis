"""Cluster summarization commands."""

from datetime import datetime

import typer
from rich.console import Console

from ...clustering.summarizer import ClusterData, ClusterSummarizer
from ...db.connection import get_db
from ...db.models import ClusterSummary
from ...services import cluster_service, run_service
from ..common import chroma_path_option, db_path_option, with_error_handling
from ..helpers.client_factory import (
    create_chroma_client,
    create_embedding_generator,
)

console = Console()


@with_error_handling
def summarize(
    run_id: str = typer.Argument(..., help="Run ID to summarize"),
    cluster_id: int = typer.Option(None, help="Specific cluster to summarize"),
    model: str = typer.Option("openai/gpt-4o-mini", help="LLM model (provider/model)"),
    max_queries: int = typer.Option(100, help="Max queries to send to LLM per cluster"),
    concurrency: int = typer.Option(50, help="Parallel LLM calls for summarization"),
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
    db_path: str = db_path_option,
    chroma_path: str = chroma_path_option,
):
    """Generate LLM-powered titles and descriptions for clusters."""
    db = get_db(db_path)

    console.print(f"[cyan]Using LLM: {model}[/cyan]")

    run = run_service.get_run(db, run_id)
    if not run:
        console.print(f"[red]Run {run_id} not found[/red]")
        raise typer.Exit(1)

    if not summary_run_id:
        model_short = model.split("/")[-1][:20]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_run_id = f"summary-{model_short}-{timestamp}"

    console.print(f"[cyan]Summary run ID: {summary_run_id}[/cyan]")
    if alias:
        console.print(f"[cyan]Alias: {alias}[/cyan]")

    summarizer = ClusterSummarizer(
        model=model,
        concurrency=concurrency,
        rpm=rpm,
    )

    if cluster_id is not None:
        cluster_ids = [cluster_id]
    else:
        cluster_ids = cluster_service.get_cluster_ids_for_run(db, run_id)

    console.print(f"[cyan]Summarizing {len(cluster_ids)} clusters for run {run_id}[/cyan]")

    clusters_data = []
    for cid in cluster_ids:
        _, query_texts = cluster_service.get_cluster_queries_with_texts(db, run_id, cid)
        clusters_data.append(ClusterData(cluster_id=cid, queries=query_texts))

    results = summarizer.generate_batch_summaries(
        clusters_data=clusters_data,
        max_queries=max_queries,
        concurrency=concurrency,
        rpm=rpm,
        contrast_neighbors=contrast_neighbors,
        contrast_examples=contrast_examples,
        contrast_mode=contrast_mode,
    )

    console.print("[cyan]Storing summaries in SQLite...[/cyan]")
    sizes_map = {cid: len(qs) for cid, qs in clusters_data}

    summary_params = {
        "max_queries": max_queries,
        "concurrency": concurrency,
        "rpm": rpm,
        "contrast_neighbors": contrast_neighbors,
        "contrast_examples": contrast_examples,
        "contrast_mode": contrast_mode,
    }

    with db.get_session() as session:
        for cid, summary_data in results.items():
            from sqlmodel import select

            statement = select(ClusterSummary).where(
                ClusterSummary.run_id == run_id,
                ClusterSummary.cluster_id == cid,
                ClusterSummary.summary_run_id == summary_run_id,
            )
            existing = session.exec(statement).first()

            if existing:
                existing.title = summary_data["title"]
                existing.description = summary_data["description"]
                existing.summary = f"{summary_data['title']}\n\n{summary_data['description']}"
                existing.representative_queries = [q[:200] for q in summary_data["sample_queries"]]
                existing.model = model
                existing.parameters = summary_params
                existing.alias = alias
            else:
                new_summary = ClusterSummary(
                    run_id=run_id,
                    cluster_id=cid,
                    summary_run_id=summary_run_id,
                    alias=alias,
                    title=summary_data["title"],
                    description=summary_data["description"],
                    summary=f"{summary_data['title']}\n\n{summary_data['description']}",
                    num_queries=sizes_map.get(cid, 0),
                    representative_queries=list(summary_data["sample_queries"]),
                    model=model,
                    parameters=summary_params,
                )
                session.add(new_summary)

        session.commit()

    console.print(f"\n[bold green]✓ Generated {len(results)} cluster summaries[/bold green]\n")

    for cid in sorted(results.keys()):
        summary_data = results[cid]
        console.print(f"[bold cyan]Cluster {cid}[/bold cyan]: {summary_data['title']}")
        console.print(f"  {summary_data['description']}")
        console.print(f"  [dim]({sizes_map.get(cid, 0)} queries)[/dim]\n")

    if use_chroma:
        console.print("[cyan]Generating embeddings for summaries...[/cyan]")

        embed_model = (
            run.parameters.get("embedding_model")
            if (run and run.parameters)
            else "text-embedding-3-small"
        )
        embed_provider = (
            run.parameters.get("embedding_provider") if (run and run.parameters) else "openai"
        )

        chroma = create_chroma_client(chroma_path, embed_model, embed_provider)
        embedding_gen = create_embedding_generator(embed_model, embed_provider)

        cluster_ids_list = []
        summaries_list = []
        titles_list = []
        descriptions_list = []
        metadata_list = []

        for cid, summary_data in results.items():
            cluster_ids_list.append(cid)
            summary_text = f"{summary_data['title']}\n\n{summary_data['description']}"
            summaries_list.append(summary_text)
            titles_list.append(summary_data["title"])
            descriptions_list.append(summary_data["description"])
            metadata_list.append(
                {
                    "num_queries": sizes_map.get(cid, 0),
                    "model": model,
                    "summary_run_id": summary_run_id,
                    "alias": alias,
                }
            )

        embeddings = embedding_gen.generate_embeddings(
            summaries_list,
            batch_size=32,
            show_progress=True,
        )

        chroma.add_cluster_summaries_batch(
            run_id=run_id,
            cluster_ids=cluster_ids_list,
            summaries=summaries_list,
            embeddings=embeddings,
            metadata_list=metadata_list,
            titles=titles_list,
            descriptions=descriptions_list,
        )

        console.print(f"[green]Stored {len(cluster_ids_list)} summaries in ChromaDB[/green]")

    console.print(f"\n[cyan]Use 'lmsys list-clusters {run_id}' to view all cluster titles[/cyan]")
