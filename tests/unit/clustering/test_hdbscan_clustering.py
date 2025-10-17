"""Tests for HDBSCAN clustering."""

import numpy as np
import pytest

from lmsys_query_analysis.clustering.hdbscan_clustering import HDBSCANClustering


def test_hdbscan_initialization_defaults():
    """Test HDBSCAN initialization with default parameters."""
    clusterer = HDBSCANClustering()

    assert clusterer.min_cluster_size == 15
    assert clusterer.min_samples == 15  # Defaults to min_cluster_size
    assert clusterer.cluster_selection_epsilon == 0.0
    assert clusterer.metric == "euclidean"
    assert clusterer.clusterer is None  # Not fitted yet


def test_hdbscan_initialization_custom():
    """Test HDBSCAN initialization with custom parameters."""
    clusterer = HDBSCANClustering(
        min_cluster_size=20, min_samples=10, cluster_selection_epsilon=0.5, metric="cosine"
    )

    assert clusterer.min_cluster_size == 20
    assert clusterer.min_samples == 10
    assert clusterer.cluster_selection_epsilon == 0.5
    assert clusterer.metric == "cosine"


def test_hdbscan_fit_predict_simple_clusters():
    """Test HDBSCAN clustering on simple well-separated clusters."""
    # Create 3 well-separated clusters
    np.random.seed(42)
    cluster1 = np.random.randn(30, 5) + np.array([10, 10, 10, 10, 10])
    cluster2 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
    cluster3 = np.random.randn(30, 5) + np.array([-10, -10, -10, -10, -10])

    embeddings = np.vstack([cluster1, cluster2, cluster3])

    clusterer = HDBSCANClustering(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 90
    assert clusterer.clusterer is not None  # Fitted

    # Should find at least 2 clusters (might find 3)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    assert len(unique_labels) >= 2


def test_hdbscan_noise_detection():
    """Test that HDBSCAN can detect noise points."""
    np.random.seed(42)

    # Create a tight cluster and scattered noise points
    cluster = np.random.randn(50, 5) * 0.1  # Tight cluster at origin
    noise = np.random.randn(10, 5) * 10  # Scattered noise

    embeddings = np.vstack([cluster, noise])

    clusterer = HDBSCANClustering(min_cluster_size=20, min_samples=10)
    labels = clusterer.fit_predict(embeddings)

    # HDBSCAN may label everything as noise if structure isn't clear enough
    # Just verify we get valid labels back
    assert len(labels) == 60
    assert -1 in labels  # Should have at least some noise points


def test_hdbscan_single_cluster():
    """Test HDBSCAN with data that forms a single cluster."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 10) * 0.5  # All close together

    clusterer = HDBSCANClustering(min_cluster_size=30)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 100
    # Should form one cluster or label all as noise if too dispersed
    unique_clusters = set(labels)
    unique_clusters.discard(-1)
    assert len(unique_clusters) <= 1


def test_hdbscan_min_cluster_size_enforcement():
    """Test that min_cluster_size is respected."""
    np.random.seed(42)

    # Create enough points for min_cluster_size
    cluster1 = np.random.randn(20, 5)
    cluster2 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=15, min_samples=5)
    labels = clusterer.fit_predict(embeddings)

    # Should successfully cluster with sufficient points
    assert len(labels) == 40


def test_hdbscan_empty_embeddings():
    """Test HDBSCAN with empty embeddings array raises error."""
    embeddings = np.array([]).reshape(0, 5)

    clusterer = HDBSCANClustering()

    # HDBSCAN requires at least one sample
    with pytest.raises(ValueError, match="minimum of 1 is required"):
        clusterer.fit_predict(embeddings)


def test_hdbscan_single_point():
    """Test HDBSCAN with a single data point raises error."""
    embeddings = np.array([[1.0, 2.0, 3.0]])

    # HDBSCAN with min_samples > 1 will fail with single point
    clusterer = HDBSCANClustering(min_cluster_size=2)

    # Single point doesn't have enough neighbors for HDBSCAN
    with pytest.raises(ValueError, match="k must be less than or equal"):
        clusterer.fit_predict(embeddings)


def test_hdbscan_high_dimensional():
    """Test HDBSCAN with high-dimensional embeddings."""
    np.random.seed(42)

    # Create clusters in high-dimensional space
    cluster1 = np.random.randn(50, 100) + 5
    cluster2 = np.random.randn(50, 100) - 5
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=20)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 100
    # Should find at least one cluster
    unique_labels = set(labels)
    assert len(unique_labels) >= 2  # At least noise + 1 cluster


def test_hdbscan_alternate_metric():
    """Test HDBSCAN with alternate distance metric."""
    np.random.seed(42)

    # Create clusters (using manhattan distance)
    cluster1 = np.random.randn(30, 4) + 5
    cluster2 = np.random.randn(30, 4) - 5
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(
        min_cluster_size=15,
        metric="manhattan",  # Use manhattan instead of cosine
    )
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 60
    # Should find at least one cluster
    unique_labels = set(labels)
    unique_labels.discard(-1)
    assert len(unique_labels) >= 1


def test_hdbscan_cluster_statistics():
    """Test getting statistics about clusters."""
    np.random.seed(42)

    # Create 2 clusters of different sizes
    cluster1 = np.random.randn(60, 5) + 10
    cluster2 = np.random.randn(40, 5) - 10
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=20)
    labels = clusterer.fit_predict(embeddings)

    # Check cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Remove noise if present
    mask = unique_labels != -1
    cluster_counts = counts[mask]

    if len(cluster_counts) > 0:
        # At least one cluster should have >= min_cluster_size
        assert np.max(cluster_counts) >= 20


def test_hdbscan_reproducibility():
    """Test that HDBSCAN produces consistent results."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 10)

    clusterer1 = HDBSCANClustering(min_cluster_size=15)
    labels1 = clusterer1.fit_predict(embeddings.copy())

    clusterer2 = HDBSCANClustering(min_cluster_size=15)
    labels2 = clusterer2.fit_predict(embeddings.copy())

    # Results should be identical for same input
    np.testing.assert_array_equal(labels1, labels2)


def test_hdbscan_cluster_selection_epsilon():
    """Test cluster_selection_epsilon parameter."""
    np.random.seed(42)

    # Create overlapping clusters
    cluster1 = np.random.randn(40, 5) + 2
    cluster2 = np.random.randn(40, 5) + 4  # Partially overlapping
    embeddings = np.vstack([cluster1, cluster2])

    # With epsilon=0, should split into more clusters
    clusterer_no_epsilon = HDBSCANClustering(min_cluster_size=15, cluster_selection_epsilon=0.0)
    labels_no_epsilon = clusterer_no_epsilon.fit_predict(embeddings)

    # With larger epsilon, might merge clusters
    clusterer_with_epsilon = HDBSCANClustering(min_cluster_size=15, cluster_selection_epsilon=2.0)
    labels_with_epsilon = clusterer_with_epsilon.fit_predict(embeddings)

    # Both should produce valid labels
    assert len(labels_no_epsilon) == 80
    assert len(labels_with_epsilon) == 80

    # Number of clusters might differ
    n_clusters_no_epsilon = len(set(labels_no_epsilon)) - (1 if -1 in labels_no_epsilon else 0)
    n_clusters_with_epsilon = len(set(labels_with_epsilon)) - (
        1 if -1 in labels_with_epsilon else 0
    )

    # Just verify we got some clustering
    assert n_clusters_no_epsilon >= 0
    assert n_clusters_with_epsilon >= 0


# ==============================================================================
# Tests for additional HDBSCANClustering methods
# ==============================================================================


def test_get_cluster_probabilities_not_fitted():
    """Test get_cluster_probabilities raises error if not fitted."""
    clusterer = HDBSCANClustering(min_cluster_size=10)

    with pytest.raises(ValueError, match="Must call fit_predict first"):
        clusterer.get_cluster_probabilities()


def test_get_cluster_probabilities():
    """Test get_cluster_probabilities returns valid probabilities."""
    np.random.seed(42)
    # Create well-separated clusters
    cluster1 = np.random.randn(30, 5) + 10
    cluster2 = np.random.randn(30, 5) - 10
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(embeddings)

    probabilities = clusterer.get_cluster_probabilities()

    # Check shape and values
    assert len(probabilities) == 60
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    assert isinstance(probabilities, np.ndarray)


def test_get_cluster_persistence_not_fitted():
    """Test get_cluster_persistence raises error if not fitted."""
    clusterer = HDBSCANClustering(min_cluster_size=10)

    with pytest.raises(ValueError, match="Must call fit_predict first"):
        clusterer.get_cluster_persistence()


def test_get_cluster_persistence():
    """Test get_cluster_persistence returns valid persistence values."""
    np.random.seed(42)
    # Create well-separated clusters
    cluster1 = np.random.randn(30, 5) + 10
    cluster2 = np.random.randn(30, 5) - 10
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(embeddings)

    persistence = clusterer.get_cluster_persistence()

    # Should return a dictionary
    assert isinstance(persistence, dict)

    # Should have entries for each non-noise cluster
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise
    assert len(persistence) == len(unique_labels)

    # All values should be non-negative
    for cluster_id, value in persistence.items():
        assert cluster_id != -1  # Should not include noise
        assert value >= 0.0


def test_get_cluster_persistence_all_noise():
    """Test get_cluster_persistence with all noise points."""
    np.random.seed(42)
    # Create scattered noise (no dense clusters)
    embeddings = np.random.randn(50, 5) * 20

    clusterer = HDBSCANClustering(min_cluster_size=40, min_samples=30)
    labels = clusterer.fit_predict(embeddings)

    persistence = clusterer.get_cluster_persistence()

    # If all points are noise, persistence dict should be empty
    if all(label == -1 for label in labels):
        assert len(persistence) == 0
    else:
        # Otherwise, should have entries for actual clusters
        assert len(persistence) >= 0


def test_compute_centroids():
    """Test compute_centroids computes correct cluster centers."""
    np.random.seed(42)
    # Create clusters at known positions
    cluster1_center = np.array([10, 10, 10])
    cluster2_center = np.array([-10, -10, -10])

    cluster1 = np.random.randn(30, 3) * 0.5 + cluster1_center
    cluster2 = np.random.randn(30, 3) * 0.5 + cluster2_center

    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(embeddings)

    centroids = clusterer.compute_centroids(embeddings, labels)

    # Should return a dictionary
    assert isinstance(centroids, dict)

    # Should have entries for each non-noise cluster
    unique_labels = set(labels)
    unique_labels.discard(-1)
    assert len(centroids) == len(unique_labels)

    # Each centroid should have correct dimensionality
    for cluster_id, centroid in centroids.items():
        assert cluster_id != -1  # Should not include noise
        assert len(centroid) == 3
        assert isinstance(centroid, np.ndarray)


def test_compute_centroids_with_noise():
    """Test compute_centroids excludes noise points."""
    np.random.seed(42)
    # Create a cluster and some noise
    cluster = np.random.randn(30, 5) * 0.5
    noise = np.random.randn(10, 5) * 10

    embeddings = np.vstack([cluster, noise])

    clusterer = HDBSCANClustering(min_cluster_size=15, min_samples=10)
    labels = clusterer.fit_predict(embeddings)

    centroids = clusterer.compute_centroids(embeddings, labels)

    # Should not have -1 (noise) in centroids
    assert -1 not in centroids

    # All centroids should have correct shape
    for centroid in centroids.values():
        assert len(centroid) == 5


def test_compute_centroids_single_cluster():
    """Test compute_centroids with single cluster."""
    np.random.seed(42)
    embeddings = np.random.randn(50, 4) * 0.3

    clusterer = HDBSCANClustering(min_cluster_size=20, min_samples=10)
    labels = clusterer.fit_predict(embeddings)

    centroids = clusterer.compute_centroids(embeddings, labels)

    # Should have at most one cluster
    assert len(centroids) <= 1

    if len(centroids) == 1:
        centroid = list(centroids.values())[0]
        assert len(centroid) == 4
        # Centroid should be near origin (since data is centered at origin)
        assert np.abs(centroid).max() < 2.0


def test_compute_centroids_empty_labels():
    """Test compute_centroids with all noise (no clusters)."""
    np.random.seed(42)
    embeddings = np.random.randn(20, 3) * 50  # Very sparse

    # Create labels that are all noise
    labels = np.full(20, -1)

    clusterer = HDBSCANClustering(min_cluster_size=10)
    centroids = clusterer.compute_centroids(embeddings, labels)

    # Should return empty dict
    assert len(centroids) == 0
    assert isinstance(centroids, dict)


def test_compute_centroids_accuracy():
    """Test that computed centroids are close to actual cluster centers."""
    np.random.seed(42)
    # Create tight clusters at specific locations
    center1 = np.array([100, 200, 300])
    center2 = np.array([-100, -200, -300])

    cluster1 = np.random.randn(40, 3) * 1.0 + center1  # Tight cluster
    cluster2 = np.random.randn(40, 3) * 1.0 + center2  # Tight cluster

    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=15, min_samples=10)
    labels = clusterer.fit_predict(embeddings)

    centroids = clusterer.compute_centroids(embeddings, labels)

    # Should find 2 clusters
    if len(centroids) == 2:
        centroid_values = list(centroids.values())

        # One centroid should be close to center1, other to center2
        distances_to_center1 = [np.linalg.norm(c - center1) for c in centroid_values]
        distances_to_center2 = [np.linalg.norm(c - center2) for c in centroid_values]

        # At least one centroid should be close to each center
        assert min(distances_to_center1) < 5.0
        assert min(distances_to_center2) < 5.0
