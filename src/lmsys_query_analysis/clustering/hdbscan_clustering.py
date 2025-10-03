"""HDBSCAN clustering for query analysis."""
from typing import List, Optional
import numpy as np
import hdbscan
from rich.console import Console

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
        console.print(f"  Noise points: {n_noise} ({n_noise/n_total*100:.1f}%)")
        console.print(f"  Clustered points: {n_total - n_noise} ({(n_total-n_noise)/n_total*100:.1f}%)")

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
                persistence[cluster_id] = self.clusterer.cluster_persistence_[cluster_id]

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
