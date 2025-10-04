"""HDBSCAN clustering for query analysis."""

from typing import List, Optional, Dict
import numpy as np
import hdbscan
from rich.console import Console
from sqlmodel import select

from ..db.connection import Database
from ..db.models import Query, ClusteringRun, QueryCluster
from ..db.chroma import ChromaManager
from .embeddings import EmbeddingGenerator

console = Console()


class HDBSCANClustering:
    """HDBSCAN-based clustering for finding natural query groups."""

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
    ):
        """Initialize HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum queries per cluster (default: 15)
            min_samples: Minimum samples in neighborhood (default: same as min_cluster_size)
            cluster_selection_epsilon: Distance threshold for merging clusters (default: 0.0)
            metric: Distance metric (default: euclidean)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.clusterer = None

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN and predict cluster labels.

        Args:
            embeddings: Array of shape (n_samples, n_features)

        Returns:
            Array of cluster labels (shape: n_samples)
            Note: Label -1 indicates noise/outliers
        """
        console.print(f"[cyan]Running HDBSCAN clustering...[/cyan]")
        console.print(f"  Min cluster size: {self.min_cluster_size}")
        console.print(f"  Min samples: {self.min_samples}")
        console.print(f"  Cluster selection epsilon: {self.cluster_selection_epsilon}")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            prediction_data=True,
            core_dist_n_jobs=-1,  # Use all cores
        )

        labels = self.clusterer.fit_predict(embeddings)

        # Report statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        n_total = len(labels)

        console.print(f"[green]✓ Found {n_clusters} clusters[/green]")
        console.print(f"  Noise points: {n_noise} ({n_noise / n_total * 100:.1f}%)")
        console.print(
            f"  Clustered points: {n_total - n_noise} ({(n_total - n_noise) / n_total * 100:.1f}%)"
        )

        return labels

    def get_cluster_probabilities(self) -> np.ndarray:
        """Get cluster membership probabilities.

        Returns:
            Array of probabilities (shape: n_samples)
        """
        if self.clusterer is None:
            raise ValueError("Must call fit_predict first")
        return self.clusterer.probabilities_

    def get_cluster_persistence(self) -> dict:
        """Get cluster persistence values (stability measure).

        Returns:
            Dict mapping cluster_id to persistence value
        """
        if self.clusterer is None:
            raise ValueError("Must call fit_predict first")

        persistence = {}
        for cluster_id in set(self.clusterer.labels_):
            if cluster_id != -1:
                persistence[cluster_id] = self.clusterer.cluster_persistence_[
                    cluster_id
                ]

        return persistence

    def compute_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Compute centroid for each cluster.

        Args:
            embeddings: Original embeddings
            labels: Cluster labels from fit_predict

        Returns:
            Dict mapping cluster_id to centroid vector
        """
        centroids = {}
        for cluster_id in set(labels):
            if cluster_id != -1:  # Skip noise
                mask = labels == cluster_id
                cluster_embeddings = embeddings[mask]
                centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)

        return centroids


def run_hdbscan_clustering(
    db: Database,
    description: str = "",
    embedding_model: str = "all-MiniLM-L6-v2",
    embed_batch_size: int = 32,
    chunk_size: int = 5000,
    embedding_provider: str = "sentence-transformers",
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    chroma: Optional[ChromaManager] = None,
) -> str:
    """Run HDBSCAN clustering on all queries in the database.

    Stores assignments (excluding noise) and optional centroids to Chroma.
    """
    session = db.get_session()
    try:
        total = session.exec(select(Query.id)).all()
        if not total:
            console.print(
                "[red]No queries found in database. Run 'lmsys load' first.[/red]"
            )
            return None

        # Load texts and ids in chunks and embed
        def iter_query_chunks() -> List[Query]:
            offset = 0
            while True:
                rows = session.exec(
                    select(Query).offset(offset).limit(chunk_size)
                ).all()
                if not rows:
                    break
                yield rows
                offset += len(rows)

        eg = EmbeddingGenerator(model_name=embedding_model, provider=embedding_provider)
        eg.load_model()
        all_ids: List[int] = []
        all_embs: List[np.ndarray] = []
        for chunk in iter_query_chunks():
            texts = [q.query_text for q in chunk]
            ids = [q.id for q in chunk]
            embs = eg.model.encode(
                texts,
                batch_size=embed_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_ids.extend(ids)
            all_embs.append(embs)

        embeddings = np.vstack(all_embs)
        clusterer = HDBSCANClustering(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
        )
        labels = clusterer.fit_predict(embeddings)

        # Persist run
        from datetime import datetime

        run_id = (
            f"hdbscan-{min_cluster_size}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        )
        run = ClusteringRun(
            run_id=run_id,
            algorithm="hdbscan",
            parameters={
                "embedding_model": embedding_model,
                "provider": embedding_provider,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples or min_cluster_size,
                "cluster_selection_epsilon": cluster_selection_epsilon,
                "metric": metric,
            },
            description=description,
            num_clusters=int(len(set(labels)) - (1 if -1 in labels else 0)),
        )
        session.add(run)
        session.commit()

        # Store assignments (exclude noise)
        assignments = []
        for qid, label in zip(all_ids, labels):
            if label != -1:
                assignments.append(
                    QueryCluster(run_id=run_id, query_id=qid, cluster_id=int(label))
                )
        session.add_all(assignments)
        session.commit()

        # Chroma: store centroids
        if chroma:
            centroids = clusterer.compute_centroids(embeddings, labels)
            used_ids = sorted(centroids.keys())
            centroid_arr = np.array([centroids[cid] for cid in used_ids])
            summaries = [f"HDBSCAN Cluster {cid}" for cid in used_ids]
            meta = [{"num_queries": int(int(np.sum(labels == cid)))} for cid in used_ids]
            chroma.add_cluster_summaries_batch(
                run_id=run_id,
                cluster_ids=used_ids,
                summaries=summaries,
                embeddings=centroid_arr,
                metadata_list=meta,
            )

        return run_id
    finally:
        session.close()
