"""Mini-batch KMeans clustering implementation with streaming embeddings."""

import logging
import time
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.cluster import MiniBatchKMeans
from sqlalchemy import func
from sqlmodel import select

from ..db.chroma import ChromaManager
from ..db.connection import Database
from ..db.models import ClusteringRun, Query, QueryCluster
from .embeddings import EmbeddingGenerator

console = Console()
logger = logging.getLogger("lmsys")


def run_kmeans_clustering(
    db: Database,
    n_clusters: int = 200,
    description: str = "",
    embedding_model: str = "all-MiniLM-L6-v2",
    embed_batch_size: int = 32,
    chunk_size: int = 5000,
    mb_batch_size: int = 4096,
    embedding_provider: str = "sentence-transformers",
    random_state: int = 42,
    chroma: ChromaManager | None = None,
    max_queries: int | None = None,
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
        console.print("Counting queries in database...")
        count_result = session.exec(select(func.count()).select_from(Query)).one()
        total_queries = int(count_result[0] if isinstance(count_result, tuple) else count_result)

        if total_queries == 0:
            console.print("[red]No queries found in database. Run 'lmsys load' first.[/red]")
            return None

        effective_total = (
            total_queries if max_queries is None else min(total_queries, int(max_queries))
        )
        console.print(f"[green]Found {total_queries} queries[/green]")
        if max_queries is not None and effective_total < total_queries:
            console.print(f"[yellow]Limiting to first {effective_total} queries[/yellow]")

        # Create clustering run record (saved early to link assignments)
        run_id = f"kmeans-{n_clusters}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        clustering_run = ClusteringRun(
            run_id=run_id,
            algorithm="kmeans-minibatch",
            parameters={
                "n_clusters": n_clusters,
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
                "random_state": random_state,
                "kmeans": "MiniBatchKMeans",
                "encode_batch_size": embed_batch_size,
                "mb_batch_size": mb_batch_size,
                **({"limit": int(max_queries)} if max_queries is not None else {}),
            },
            description=description,
            num_clusters=n_clusters,
        )
        session.add(clustering_run)
        session.commit()

        # Initialize embedding generator lazily (only if needed)
        embedding_gen = None

        # Initialize MiniBatchKMeans
        mbk = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=max(1024, mb_batch_size),
            n_init=20,
            max_iter=200,
            verbose=0,
        )

        # Helper: iterate queries in chunks to limit memory
        def iter_query_chunks(chunk_size: int = chunk_size):
            offset = 0
            target = effective_total
            while offset < target:
                remaining = target - offset
                stmt = select(Query).offset(offset).limit(min(chunk_size, remaining))
                rows = session.exec(stmt).all()
                if not rows:
                    break
                yield rows
                offset += len(rows)

        # First pass: partial_fit on streaming embeddings
        console.print(f"[yellow]Training MiniBatchKMeans with {n_clusters} clusters...[/yellow]")
        fit_start = time.perf_counter()
        total_fit = 0
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task("[cyan]Fitting clusters (pass 1)...", total=None)
            for chunk in iter_query_chunks():
                texts = [q.query_text for q in chunk]
                ids = [q.id for q in chunk]

                # Try to reuse embeddings from Chroma if available
                chunk_embeddings = None
                if chroma is not None:
                    emb_map = chroma.get_query_embeddings_map(ids)
                    if len(emb_map) == len(ids):
                        chunk_embeddings = np.stack([emb_map[qid] for qid in ids])
                    else:
                        # Compute missing and optionally backfill
                        missing_ids = [qid for qid in ids if qid not in emb_map]
                        if missing_ids:
                            if embedding_gen is None:
                                embedding_gen = EmbeddingGenerator(
                                    model_name=embedding_model,
                                    provider=embedding_provider,
                                )
                            # Compute all embeddings then compose full array
                            all_emb = embedding_gen.generate_embeddings(
                                texts,
                                batch_size=embed_batch_size,
                                show_progress=False,
                            )
                            # Backfill map and Chroma for missing
                            for qid, emb in zip(ids, all_emb, strict=False):
                                if qid not in emb_map:
                                    emb_map[qid] = emb
                            # Optionally write missing to Chroma (with metadata)
                            try:
                                if chroma is not None and len(missing_ids) > 0:
                                    missing_idx = [ids.index(mid) for mid in missing_ids]
                                    meta = [
                                        {
                                            "model": chunk[i].model,
                                            "language": chunk[i].language or "unknown",
                                            "conversation_id": chunk[i].conversation_id,
                                        }
                                        for i in missing_idx
                                    ]
                                    chroma.add_queries_batch(
                                        query_ids=missing_ids,
                                        texts=[texts[i] for i in missing_idx],
                                        embeddings=np.array([all_emb[i] for i in missing_idx]),
                                        metadata=meta,
                                    )
                            except Exception:
                                pass
                            chunk_embeddings = np.stack([emb_map[qid] for qid in ids])
                else:
                    # No Chroma: compute embeddings
                    if embedding_gen is None:
                        embedding_gen = EmbeddingGenerator(
                            model_name=embedding_model, provider=embedding_provider
                        )
                    chunk_embeddings = embedding_gen.generate_embeddings(
                        texts,
                        batch_size=embed_batch_size,
                        show_progress=False,
                    )

                # Ensure sklearn receives float64 arrays
                if chunk_embeddings.dtype != np.float64:
                    chunk_embeddings = chunk_embeddings.astype(np.float64, copy=False)
                mbk.partial_fit(chunk_embeddings)
                total_fit += len(texts)
            progress.update(task, completed=True)

        console.print("[green]Mini-batch training complete.[/green]")
        fit_elapsed = time.perf_counter() - fit_start
        if total_fit:
            logger.info(
                "MiniBatch fit: %s vectors in %.2fs (%.1f/s)",
                total_fit,
                fit_elapsed,
                (total_fit / fit_elapsed if fit_elapsed else float("inf")),
            )

        # Second pass: predict labels and write assignments incrementally
        console.print("[yellow]Assigning clusters and writing to DB...[/yellow]")
        cluster_counts: dict[int, int] = dict.fromkeys(range(n_clusters), 0)
        sample_texts: dict[int, list[str]] = {i: [] for i in range(n_clusters)}

        batch_assignments: list[QueryCluster] = []
        commit_every = 5000

        predict_start = time.perf_counter()
        total_pred = 0
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task("[cyan]Predicting (pass 2)...", total=None)
            for chunk in iter_query_chunks():
                texts = [q.query_text for q in chunk]
                ids = [q.id for q in chunk]

                # Reuse embeddings from Chroma if possible; otherwise compute
                if chroma is not None:
                    emb_map = chroma.get_query_embeddings_map(ids)
                    if len(emb_map) == len(ids):
                        chunk_embeddings = np.stack([emb_map[qid] for qid in ids])
                    else:
                        missing_ids = [qid for qid in ids if qid not in emb_map]
                        if embedding_gen is None:
                            embedding_gen = EmbeddingGenerator(
                                model_name=embedding_model, provider=embedding_provider
                            )
                        all_emb = embedding_gen.generate_embeddings(
                            texts,
                            batch_size=embed_batch_size,
                            show_progress=False,
                        )
                        for qid, emb in zip(ids, all_emb, strict=False):
                            if qid not in emb_map:
                                emb_map[qid] = emb
                        try:
                            if chroma is not None and len(missing_ids) > 0:
                                missing_idx = [ids.index(mid) for mid in missing_ids]
                                meta = [
                                    {
                                        "model": chunk[i].model,
                                        "language": chunk[i].language or "unknown",
                                        "conversation_id": chunk[i].conversation_id,
                                    }
                                    for i in missing_idx
                                ]
                                chroma.add_queries_batch(
                                    query_ids=missing_ids,
                                    texts=[texts[i] for i in missing_idx],
                                    embeddings=np.array([all_emb[i] for i in missing_idx]),
                                    metadata=meta,
                                )
                        except Exception:
                            pass
                        chunk_embeddings = np.stack([emb_map[qid] for qid in ids])
                else:
                    if embedding_gen is None:
                        embedding_gen = EmbeddingGenerator(
                            model_name=embedding_model, provider=embedding_provider
                        )
                    chunk_embeddings = embedding_gen.generate_embeddings(
                        texts,
                        batch_size=embed_batch_size,
                        show_progress=False,
                    )

                # Ensure sklearn receives float64 arrays
                if chunk_embeddings.dtype != np.float64:
                    chunk_embeddings = chunk_embeddings.astype(np.float64, copy=False)
                labels = mbk.predict(chunk_embeddings)
                total_pred += len(texts)

                for qid, qtext, label in zip(ids, texts, labels, strict=False):
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
                total_pred,
                pred_elapsed,
                (total_pred / pred_elapsed if pred_elapsed else float("inf")),
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
                f"Cluster {cid} ({cluster_counts[cid]} queries)\nSamples:\n"
                + "\n".join(f"- {q}" for q in sample_texts[cid])
                for cid in used_cluster_ids
            ]
            metadata = [
                {
                    "num_queries": cluster_counts[cid],
                    "sample_count": len(sample_texts[cid]),
                }
                for cid in used_cluster_ids
            ]

            chroma.add_cluster_summaries_batch(
                run_id=run_id,
                cluster_ids=used_cluster_ids,
                summaries=summaries,
                embeddings=np.array(centroids),
                metadata_list=metadata,
            )

            logger.info("Chroma summaries written: clusters=%s", len(used_cluster_ids))

            console.print(
                f"[green]Wrote {len(used_cluster_ids)} cluster summaries to ChromaDB[/green]"
            )

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
