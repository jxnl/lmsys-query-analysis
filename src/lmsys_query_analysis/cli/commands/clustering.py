"""Clustering algorithm commands."""

import typer
from rich.console import Console

from ..common import with_error_handling, db_path_option, chroma_path_option, embedding_model_option
from ..helpers.client_factory import parse_embedding_model, create_chroma_client
from ...db.connection import get_db
from ...clustering.kmeans import run_kmeans_clustering
from ...clustering.hdbscan_clustering import run_hdbscan_clustering

console = Console()
app = typer.Typer(help="Clustering commands")


@app.command("kmeans")
@with_error_handling
def cluster_kmeans(
    n_clusters: int = typer.Option(200, help="Number of clusters"),
    description: str = typer.Option("", help="Description of this clustering run"),
    db_path: str = db_path_option,
    embedding_model: str = embedding_model_option,
    embed_batch_size: int = typer.Option(50, help="Embedding encode batch size"),
    chunk_size: int = typer.Option(5000, help="DB iteration chunk size"),
    mb_batch_size: int = typer.Option(4096, help="MiniBatchKMeans batch_size"),
    use_chroma: bool = typer.Option(False, help="Enable ChromaDB"),
    chroma_path: str = chroma_path_option,
    limit: int = typer.Option(None, help="Limit number of queries to cluster"),
):
    """Run MiniBatchKMeans clustering."""
    model, provider = parse_embedding_model(embedding_model)
    db = get_db(db_path)
    chroma = create_chroma_client(chroma_path, model, provider) if use_chroma else None
    
    console.print(
        f"[cyan]Running clustering: algo=kmeans, n_clusters={n_clusters}, model={model}, provider={provider}, use_chroma={use_chroma}, limit={limit}[/cyan]"
    )
    
    run_id = run_kmeans_clustering(
        db=db,
        n_clusters=n_clusters,
        description=description,
        embedding_model=model,
        embed_batch_size=embed_batch_size,
        chunk_size=chunk_size,
        mb_batch_size=mb_batch_size,
        embedding_provider=provider,
        chroma=chroma,
        max_queries=limit,
    )
    
    if run_id:
        console.print(f"[green]Clustering completed: run_id={run_id}[/green]")
        console.print(
            f"[cyan]Use 'lmsys inspect {run_id} <cluster_id>' to explore clusters[/cyan]"
        )
        if use_chroma and chroma:
            console.print(
                f"[cyan]Use 'lmsys search --run-id {run_id} <query>' to search cluster summaries[/cyan]"
            )


@app.command("hdbscan")
@with_error_handling
def cluster_hdbscan(
    description: str = typer.Option("", help="Description of this clustering run"),
    db_path: str = db_path_option,
    embedding_model: str = embedding_model_option,
    embed_batch_size: int = typer.Option(50, help="Embedding encode batch size"),
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
    chroma_path: str = chroma_path_option,
    limit: int = typer.Option(None, help="Limit number of queries to cluster"),
):
    """Run HDBSCAN clustering."""
    model, provider = parse_embedding_model(embedding_model)
    db = get_db(db_path)
    chroma = create_chroma_client(chroma_path, model, provider) if use_chroma else None
    
    console.print(
        f"[cyan]Running clustering: algo=hdbscan, model={model}, provider={provider}, use_chroma={use_chroma}, limit={limit}[/cyan]"
    )
    
    run_id = run_hdbscan_clustering(
        db=db,
        description=description,
        embedding_model=model,
        embed_batch_size=embed_batch_size,
        chunk_size=chunk_size,
        embedding_provider=provider,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric=metric,
        chroma=chroma,
        max_queries=limit,
    )
    
    if run_id:
        console.print(f"[green]Clustering completed: run_id={run_id}[/green]")
        console.print(
            f"[cyan]Use 'lmsys inspect {run_id} <cluster_id>' to explore clusters[/cyan]"
        )
        if use_chroma and chroma:
            console.print(
                f"[cyan]Use 'lmsys search --run-id {run_id} <query>' to search cluster summaries[/cyan]"
            )

