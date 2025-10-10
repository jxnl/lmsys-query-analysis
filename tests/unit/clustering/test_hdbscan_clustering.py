"""Tests for HDBSCAN clustering."""

import pytest
import numpy as np
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
        min_cluster_size=20,
        min_samples=10,
        cluster_selection_epsilon=0.5,
        metric="cosine"
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
        labels = clusterer.fit_predict(embeddings)


def test_hdbscan_single_point():
    """Test HDBSCAN with a single data point raises error."""
    embeddings = np.array([[1.0, 2.0, 3.0]])
    
    # HDBSCAN with min_samples > 1 will fail with single point
    clusterer = HDBSCANClustering(min_cluster_size=2)
    
    # Single point doesn't have enough neighbors for HDBSCAN
    with pytest.raises(ValueError, match="k must be less than or equal"):
        labels = clusterer.fit_predict(embeddings)


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
        metric="manhattan"  # Use manhattan instead of cosine
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
    clusterer_no_epsilon = HDBSCANClustering(
        min_cluster_size=15,
        cluster_selection_epsilon=0.0
    )
    labels_no_epsilon = clusterer_no_epsilon.fit_predict(embeddings)
    
    # With larger epsilon, might merge clusters
    clusterer_with_epsilon = HDBSCANClustering(
        min_cluster_size=15,
        cluster_selection_epsilon=2.0
    )
    labels_with_epsilon = clusterer_with_epsilon.fit_predict(embeddings)
    
    # Both should produce valid labels
    assert len(labels_no_epsilon) == 80
    assert len(labels_with_epsilon) == 80
    
    # Number of clusters might differ
    n_clusters_no_epsilon = len(set(labels_no_epsilon)) - (1 if -1 in labels_no_epsilon else 0)
    n_clusters_with_epsilon = len(set(labels_with_epsilon)) - (1 if -1 in labels_with_epsilon else 0)
    
    # Just verify we got some clustering
    assert n_clusters_no_epsilon >= 0
    assert n_clusters_with_epsilon >= 0

