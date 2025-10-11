"""Cluster summarization commands."""

import typer
from datetime import datetime
from rich.console import Console

from ..common import with_error_handling, db_path_option, chroma_path_option
from ..helpers.client_factory import parse_embedding_model, create_chroma_client, create_embedding_generator
from ...db.connection import get_db
from ...services import cluster_service, run_service, summary_service
from ...clustering.summarizer import ClusterSummarizer
from ...db.models import ClusterSummary

console = Console()


@with_error_handling
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
    db_path: str = db_path_option,
    chroma_path: str = chroma_path_option,
):
    """Generate LLM-powered titles and descriptions for clusters."""
    db = get_db(db_path)
    
    console.print(f"[cyan]Using LLM: {model}[/cyan]")
    
    # Get the clustering run
    run = run_service.get_run(db, run_id)
    if not run:
        console.print(f"[red]Run {run_id} not found[/red]")
        raise typer.Exit(1)
    
    # Generate summary_run_id if not provided
    if not summary_run_id:
        model_short = model.split("/")[-1][:20]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_run_id = f"summary-{model_short}-{timestamp}"
    
    console.print(f"[cyan]Summary run ID: {summary_run_id}[/cyan]")
    if alias:
        console.print(f"[cyan]Alias: {alias}[/cyan]")
    
    # Parse LLM provider and model
    llm_provider, llm_model = model.split("/", 1)
    
    # Get clusters to summarize
    if cluster_id is not None:
        cluster_ids = [cluster_id]
    else:
        cluster_ids = cluster_service.get_cluster_ids_for_run(db, run_id)
    
    console.print(f"[cyan]Summarizing {len(cluster_ids)} clusters for run {run_id}[/cyan]")
    
    # Use summary service to create summaries with metadata persistence
    async def run_create_summaries():
        return await summary_service.create_summaries(
            db=db,
            run_id=run_id,
            summary_run_id=summary_run_id,
            llm_provider=llm_provider,
            llm_model=llm_model,
            max_queries=max_queries,
            concurrency=concurrency,
            rpm=rpm,
            contrast_neighbors=contrast_neighbors,
            contrast_examples=contrast_examples,
            contrast_mode=contrast_mode,
            alias=alias,
            cluster_ids=cluster_ids,
        )
    
    import asyncio
    summary_run_id, results = asyncio.run(run_create_summaries())
    
    # Summaries are now stored by the service
    
    # Display generated summaries
    console.print(f"\n[bold green]✓ Generated {len(results)} cluster summaries[/bold green]\n")
    
    # Results is now a list of summary data dictionaries
    for i, summary_data in enumerate(results):
        console.print(f"[bold cyan]Cluster {i}[/bold cyan]: {summary_data['title']}")
        console.print(f"  {summary_data['description']}")
        console.print(f"  [dim]({summary_data.get('num_queries', 'unknown')} queries)[/dim]\n")
    
    # Store in ChromaDB if requested
    if use_chroma:
        console.print("[cyan]Generating embeddings for summaries...[/cyan]")
        
        # Use the same embedding space as the clustering run if available
        embed_model = run.parameters.get("embedding_model") if (run and run.parameters) else "text-embedding-3-small"
        embed_provider = run.parameters.get("embedding_provider") if (run and run.parameters) else "openai"
        
        chroma = create_chroma_client(chroma_path, embed_model, embed_provider)
        embedding_gen = create_embedding_generator(embed_model, embed_provider)
        
        # Prepare data for batch storage
        cluster_ids_list = []
        summaries_list = []
        titles_list = []
        descriptions_list = []
        metadata_list = []
        
        for i, summary_data in enumerate(results):
            cluster_ids_list.append(i)
            summary_text = f"{summary_data['title']}\n\n{summary_data['description']}"
            summaries_list.append(summary_text)
            titles_list.append(summary_data["title"])
            descriptions_list.append(summary_data["description"])
            metadata_list.append({
                "num_queries": summary_data.get('num_queries', 0),
                "model": model,
                "summary_run_id": summary_run_id,
                "alias": alias,
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

