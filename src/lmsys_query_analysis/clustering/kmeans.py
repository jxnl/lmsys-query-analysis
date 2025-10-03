"""KMeans clustering implementation."""
import numpy as np
from datetime import datetime
from typing import Optional
from sklearn.cluster import KMeans
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from sqlmodel import Session, select

from ..db.connection import Database
from ..db.models import Query, ClusteringRun, QueryCluster
from ..db.chroma import ChromaManager
from .embeddings import EmbeddingGenerator

console = Console()


def run_kmeans_clustering(
    db: Database,
    n_clusters: int = 200,
    description: str = "",
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    random_state: int = 42,
    chroma: Optional[ChromaManager] = None,
) -> str:
    """Run KMeans clustering on all queries in the database.

    Args:
        db: Database instance
        n_clusters: Number of clusters (default: 200 for fine-grained clustering)
        description: Optional description of this clustering run
        embedding_model: Sentence transformer model name
        batch_size: Batch size for embedding generation
        random_state: Random seed for reproducibility
        chroma: Optional ChromaDB manager for storing cluster summaries

    Returns:
        run_id: Unique identifier for this clustering run
    """
    session = db.get_session()

    try:
        # Load all queries
        console.print("[yellow]Loading queries from database...[/yellow]")
        statement = select(Query)
        queries = session.exec(statement).all()

        if len(queries) == 0:
            console.print("[red]No queries found in database. Run 'lmsys load' first.[/red]")
            return None

        console.print(f"[green]Loaded {len(queries)} queries[/green]")

        # Extract query texts and IDs
        query_texts = [q.query_text for q in queries]
        query_ids = [q.id for q in queries]

        # Generate embeddings
        console.print(f"[yellow]Generating embeddings using {embedding_model}...[/yellow]")
        embedding_gen = EmbeddingGenerator(model_name=embedding_model)
        embeddings = embedding_gen.generate_embeddings(
            query_texts,
            batch_size=batch_size,
            show_progress=True,
        )

        console.print(f"[green]Generated embeddings: {embeddings.shape}[/green]")

        # Run KMeans
        console.print(f"[yellow]Running KMeans with {n_clusters} clusters...[/yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("[cyan]Clustering...", total=None)

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300,
            )
            cluster_labels = kmeans.fit_predict(embeddings)

            progress.update(task, completed=True)

        console.print(f"[green]Clustering complete![/green]")

        # Create clustering run record
        run_id = f"kmeans-{n_clusters}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        clustering_run = ClusteringRun(
            run_id=run_id,
            algorithm="kmeans",
            parameters={
                "n_clusters": n_clusters,
                "embedding_model": embedding_model,
                "random_state": random_state,
                "n_init": 10,
                "max_iter": 300,
            },
            description=description,
            num_clusters=n_clusters,
        )
        session.add(clustering_run)

        # Store cluster assignments
        console.print("[yellow]Storing cluster assignments...[/yellow]")
        cluster_assignments = []
        for query_id, cluster_id in zip(query_ids, cluster_labels):
            assignment = QueryCluster(
                run_id=run_id,
                query_id=query_id,
                cluster_id=int(cluster_id),
            )
            cluster_assignments.append(assignment)

        session.add_all(cluster_assignments)
        session.commit()

        # Calculate cluster statistics
        unique_clusters = np.unique(cluster_labels)
        cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]

        console.print(f"\n[green]Clustering Run: {run_id}[/green]")
        console.print(f"  Clusters: {len(unique_clusters)}")
        console.print(f"  Min cluster size: {min(cluster_sizes)}")
        console.print(f"  Max cluster size: {max(cluster_sizes)}")
        console.print(f"  Avg cluster size: {np.mean(cluster_sizes):.1f}")
        console.print(f"  Median cluster size: {np.median(cluster_sizes):.1f}")

        # If ChromaDB enabled, store cluster centroids and summaries
        if chroma:
            console.print("[yellow]Writing cluster centroids to ChromaDB...[/yellow]")

            # Compute cluster centroids
            cluster_centroids = []
            cluster_summaries = []
            cluster_metadata = []

            for cluster_id in unique_clusters:
                # Get indices for this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[cluster_mask]
                cluster_queries = [queries[i] for i in range(len(queries)) if cluster_mask[i]]

                # Compute centroid
                centroid = np.mean(cluster_embeddings, axis=0)
                cluster_centroids.append(centroid)

                # Create summary text with sample queries
                sample_queries = [q.query_text[:100] for q in cluster_queries[:5]]
                summary_text = f"Cluster {cluster_id} ({len(cluster_queries)} queries)\nSamples:\n" + "\n".join(f"- {q}" for q in sample_queries)
                cluster_summaries.append(summary_text)

                # Metadata
                cluster_metadata.append({
                    "num_queries": len(cluster_queries),
                    "sample_count": min(5, len(cluster_queries)),
                })

            # Write to ChromaDB in batch
            chroma.add_cluster_summaries_batch(
                run_id=run_id,
                cluster_ids=unique_clusters.tolist(),
                summaries=cluster_summaries,
                embeddings=np.array(cluster_centroids),
                metadata_list=cluster_metadata,
            )

            console.print(f"[green]Wrote {len(unique_clusters)} cluster summaries to ChromaDB[/green]")

        return run_id

    finally:
        session.close()


def get_cluster_info(db: Database, run_id: str, cluster_id: int) -> dict:
    """Get information about a specific cluster.

    Args:
        db: Database instance
        run_id: Clustering run ID
        cluster_id: Cluster ID to inspect

    Returns:
        Dictionary with cluster information
    """
    session = db.get_session()

    try:
        # Get all queries in this cluster
        statement = (
            select(Query, QueryCluster)
            .join(QueryCluster, Query.id == QueryCluster.query_id)
            .where(QueryCluster.run_id == run_id)
            .where(QueryCluster.cluster_id == cluster_id)
        )
        results = session.exec(statement).all()

        queries = [{"id": q.id, "text": q.query_text, "model": q.model} for q, _ in results]

        return {
            "run_id": run_id,
            "cluster_id": cluster_id,
            "size": len(queries),
            "queries": queries,
        }

    finally:
        session.close()
