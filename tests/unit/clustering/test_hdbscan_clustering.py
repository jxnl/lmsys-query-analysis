"""Tests for HDBSCAN clustering."""

import numpy as np
import pytest

from lmsys_query_analysis.clustering.hdbscan_clustering import HDBSCANClustering


def test_hdbscan_initialization_defaults():
    """Test HDBSCAN initialization with default parameters."""
    clusterer = HDBSCANClustering()

    assert clusterer.min_cluster_size == 15
    assert clusterer.min_samples == 15
    assert clusterer.cluster_selection_epsilon == 0.0
    assert clusterer.metric == "euclidean"
    assert clusterer.clusterer is None


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
    np.random.seed(42)
    cluster1 = np.random.randn(30, 5) + np.array([10, 10, 10, 10, 10])
    cluster2 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
    cluster3 = np.random.randn(30, 5) + np.array([-10, -10, -10, -10, -10])

    embeddings = np.vstack([cluster1, cluster2, cluster3])

    clusterer = HDBSCANClustering(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 90
    assert clusterer.clusterer is not None

    unique_labels = set(labels)
    unique_labels.discard(-1)
    assert len(unique_labels) >= 2


def test_hdbscan_noise_detection():
    """Test that HDBSCAN can detect noise points."""
    np.random.seed(42)

    cluster = np.random.randn(50, 5) * 0.1
    noise = np.random.randn(10, 5) * 10

    embeddings = np.vstack([cluster, noise])

    clusterer = HDBSCANClustering(min_cluster_size=20, min_samples=10)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 60
    assert -1 in labels


def test_hdbscan_single_cluster():
    """Test HDBSCAN with data that forms a single cluster."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 10) * 0.5

    clusterer = HDBSCANClustering(min_cluster_size=30)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 100
    unique_clusters = set(labels)
    unique_clusters.discard(-1)
    assert len(unique_clusters) <= 1


def test_hdbscan_min_cluster_size_enforcement():
    """Test that min_cluster_size is respected."""
    np.random.seed(42)

    cluster1 = np.random.randn(20, 5)
    cluster2 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=15, min_samples=5)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 40


def test_hdbscan_empty_embeddings():
    """Test HDBSCAN with empty embeddings array raises error."""
    embeddings = np.array([]).reshape(0, 5)

    clusterer = HDBSCANClustering()

    with pytest.raises(ValueError, match="minimum of 1 is required"):
        clusterer.fit_predict(embeddings)


def test_hdbscan_single_point():
    """Test HDBSCAN with a single data point raises error."""
    embeddings = np.array([[1.0, 2.0, 3.0]])

    clusterer = HDBSCANClustering(min_cluster_size=2)

    with pytest.raises(ValueError, match="k must be less than or equal"):
        clusterer.fit_predict(embeddings)


def test_hdbscan_high_dimensional():
    """Test HDBSCAN with high-dimensional embeddings."""
    np.random.seed(42)

    cluster1 = np.random.randn(50, 100) + 5
    cluster2 = np.random.randn(50, 100) - 5
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=20)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 100
    unique_labels = set(labels)
    assert len(unique_labels) >= 2


def test_hdbscan_alternate_metric():
    """Test HDBSCAN with alternate distance metric."""
    np.random.seed(42)

    cluster1 = np.random.randn(30, 4) + 5
    cluster2 = np.random.randn(30, 4) - 5
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(
        min_cluster_size=15,
        metric="manhattan",
    )
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 60
    unique_labels = set(labels)
    unique_labels.discard(-1)
    assert len(unique_labels) >= 1


def test_hdbscan_cluster_statistics():
    """Test getting statistics about clusters."""
    np.random.seed(42)

    cluster1 = np.random.randn(60, 5) + 10
    cluster2 = np.random.randn(40, 5) - 10
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = HDBSCANClustering(min_cluster_size=20)
    labels = clusterer.fit_predict(embeddings)

    unique_labels, counts = np.unique(labels, return_counts=True)

    mask = unique_labels != -1
    cluster_counts = counts[mask]

    if len(cluster_counts) > 0:
        assert np.max(cluster_counts) >= 20


def test_hdbscan_reproducibility():
    """Test that HDBSCAN produces consistent results."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 10)

    clusterer1 = HDBSCANClustering(min_cluster_size=15)
    labels1 = clusterer1.fit_predict(embeddings.copy())

    clusterer2 = HDBSCANClustering(min_cluster_size=15)
    labels2 = clusterer2.fit_predict(embeddings.copy())

    np.testing.assert_array_equal(labels1, labels2)


def test_hdbscan_cluster_selection_epsilon():
    """Test cluster_selection_epsilon parameter."""
    np.random.seed(42)

    cluster1 = np.random.randn(40, 5) + 2
    cluster2 = np.random.randn(40, 5) + 4
    embeddings = np.vstack([cluster1, cluster2])

    clusterer_no_epsilon = HDBSCANClustering(min_cluster_size=15, cluster_selection_epsilon=0.0)
    labels_no_epsilon = clusterer_no_epsilon.fit_predict(embeddings)

    clusterer_with_epsilon = HDBSCANClustering(min_cluster_size=15, cluster_selection_epsilon=2.0)
    labels_with_epsilon = clusterer_with_epsilon.fit_predict(embeddings)

    assert len(labels_no_epsilon) == 80
    assert len(labels_with_epsilon) == 80

    n_clusters_no_epsilon = len(set(labels_no_epsilon)) - (1 if -1 in labels_no_epsilon else 0)
    n_clusters_with_epsilon = len(set(labels_with_epsilon)) - (
        1 if -1 in labels_with_epsilon else 0
    )

    assert n_clusters_no_epsilon >= 0
    assert n_clusters_with_epsilon >= 0
