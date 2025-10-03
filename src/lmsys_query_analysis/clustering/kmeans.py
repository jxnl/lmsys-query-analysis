"""Mini-batch KMeans clustering implementation with streaming embeddings."""
import numpy as np
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List
from sklearn.cluster import MiniBatchKMeans
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from sqlmodel import select
from sqlalchemy import func

from ..db.connection import Database
from ..db.models import Query, ClusteringRun, QueryCluster
from ..db.chroma import ChromaManager
from .embeddings import EmbeddingGenerator

console = Console()
logger = logging.getLogger("lmsys")


def run_kmeans_clustering(
    db: Database,
    n_clusters: int = 200,
    description: str = "",
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    random_state: int = 42,
    chroma: Optional[ChromaManager] = None,
) -> str:
    """Run MiniBatchKMeans clustering on all queries using streaming embeddings.

    Args:
        db: Database instance
        n_clusters: Number of clusters (default: 200 for fine-grained clustering)
        description: Optional description of this clustering run
        embedding_model: Sentence transformer model name
        batch_size: Batch size for embedding generation (encode batch)
        random_state: Random seed for reproducibility
        chroma: Optional ChromaDB manager for storing cluster summaries

    Returns:
        run_id: Unique identifier for this clustering run
    """
    session = db.get_session()

    try:
        console.print("[yellow]Counting queries in database...[/yellow]")
        total_queries = session.exec(select(func.count(Query.id))).scalar_one()

        if total_queries == 0:
            console.print("[red]No queries found in database. Run 'lmsys load' first.[/red]")
            return None

        console.print(f"[green]Found {total_queries} queries[/green]
")

        # Create clustering run record (saved early to link assignments)
        run_id = f"kmeans-{n_clusters}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        clustering_run = ClusteringRun(
            run_id=run_id,
            algorithm="kmeans-minibatch",
            parameters={
                "n_clusters": n_clusters,
                "embedding_model": embedding_model,
                "random_state": random_state,
                "kmeans": "MiniBatchKMeans",
                "encode_batch_size": batch_size,
            },
            description=description,
            num_clusters=n_clusters,
        )
        session.add(clustering_run)
        session.commit()

        # Initialize embedding model once
        embedding_gen = EmbeddingGenerator(model_name=embedding_model)
        embedding_gen.load_model()

        # Initialize MiniBatchKMeans
        mbk = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=max(1000, batch_size * 8),
            n_init=10,
            max_iter=100,
            verbose=0,
        )

        # Helper: iterate queries in chunks to limit memory
        def iter_query_chunks(chunk_size: int = 5000):
            offset = 0
            while offset < total_queries:
                stmt = select(Query).offset(offset).limit(chunk_size)
                rows = session.exec(stmt).all()
                if not rows:
                    break
                yield rows
                offset += len(rows)

        # First pass: partial_fit on streaming embeddings
        console.print(f"[yellow]Training MiniBatchKMeans with {n_clusters} clusters...[/yellow]")
        fit_start = time.perf_counter()
        total_fit = 0
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("[cyan]Fitting clusters (pass 1)...", total=None)
            for chunk in iter_query_chunks():
                texts = [q.query_text for q in chunk]
                embeddings = embedding_gen.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                mbk.partial_fit(embeddings)
                total_fit += len(texts)
            progress.update(task, completed=True)

        console.print("[green]Mini-batch training complete.[/green]")
        fit_elapsed = time.perf_counter() - fit_start
        if total_fit:
            logger.info(
                "MiniBatch fit: %s vectors in %.2fs (%.1f/s)",
                total_fit, fit_elapsed, (total_fit/fit_elapsed if fit_elapsed else float("inf")),
            )

        # Second pass: predict labels and write assignments incrementally
        console.print("[yellow]Assigning clusters and writing to DB...[/yellow]")
        cluster_counts: Dict[int, int] = {i: 0 for i in range(n_clusters)}
        sample_texts: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}

        batch_assignments: List[QueryCluster] = []
        commit_every = 5000

        predict_start = time.perf_counter()
        total_pred = 0
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("[cyan]Predicting (pass 2)...", total=None)
            for chunk in iter_query_chunks():
                texts = [q.query_text for q in chunk]
                ids = [q.id for q in chunk]
                embeddings = embedding_gen.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                labels = mbk.predict(embeddings)
                total_pred += len(texts)

                for qid, qtext, label in zip(ids, texts, labels):
                    cid = int(label)
                    batch_assignments.append(
                        QueryCluster(run_id=run_id, query_id=qid, cluster_id=cid)
                    )
                    # stats and samples
                    cluster_counts[cid] += 1
                    if len(sample_texts[cid]) < 5:
                        sample_texts[cid].append(qtext[:100])

                if len(batch_assignments) >= commit_every:
                    session.add_all(batch_assignments)
                    session.commit()
                    batch_assignments.clear()
            # final commit
            if batch_assignments:
                session.add_all(batch_assignments)
                session.commit()
            progress.update(task, completed=True)
        pred_elapsed = time.perf_counter() - predict_start
        if total_pred:
            logger.info(
                "Predict+assign: %s vectors in %.2fs (%.1f/s)",
                total_pred, pred_elapsed, (total_pred/pred_elapsed if pred_elapsed else float("inf")),
            )

        # Cluster statistics
        non_empty = [c for c, n in cluster_counts.items() if n > 0]
        sizes = [cluster_counts[c] for c in non_empty] if non_empty else []

        console.print(f"\n[green]Clustering Run: {run_id}[/green]")
        console.print(f"  Clusters: {len(non_empty)} of {n_clusters} used")
        if sizes:
            console.print(f"  Min cluster size: {min(sizes)}")
            console.print(f"  Max cluster size: {max(sizes)}")
            console.print(f"  Avg cluster size: {np.mean(sizes):.1f}")
            console.print(f"  Median cluster size: {np.median(sizes):.1f}")

        # If ChromaDB enabled, store cluster centroids and summaries
        if chroma:
            console.print("[yellow]Writing cluster centroids to ChromaDB...[/yellow]")

            used_cluster_ids = [cid for cid in range(n_clusters) if cluster_counts[cid] > 0]
            centroids = mbk.cluster_centers_[used_cluster_ids]
            summaries = [
                f"Cluster {cid} ({cluster_counts[cid]} queries)\nSamples:\n" + "\n".join(f"- {q}" for q in sample_texts[cid])
                for cid in used_cluster_ids
            ]
            metadata = [
                {"num_queries": cluster_counts[cid], "sample_count": len(sample_texts[cid])}
                for cid in used_cluster_ids
            ]

            chroma.add_cluster_summaries_batch(
                run_id=run_id,
                cluster_ids=used_cluster_ids,
                summaries=summaries,
                embeddings=np.array(centroids),
                metadata_list=metadata,
            )

            logger.info(
                "Chroma summaries written: clusters=%s", len(used_cluster_ids)
            )

            console.print(f"[green]Wrote {len(used_cluster_ids)} cluster summaries to ChromaDB[/green]
")

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
