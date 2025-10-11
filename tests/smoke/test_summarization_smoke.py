"""Smoke tests for LLM-powered cluster summarization."""

import pytest
from lmsys_query_analysis.clustering.summarizer import ClusterSummarizer, ClusterData


@pytest.mark.smoke
def test_summarize_single_cluster():
    """Test summarizing a cluster with real LLM."""
    summarizer = ClusterSummarizer(
        model="openai/gpt-4o-mini",
        concurrency=1,
    )
    
    cluster_queries = [
        "What is machine learning?",
        "Explain supervised learning",
        "How does deep learning work?",
        "What are neural networks?",
    ]
    
    clusters_data = [ClusterData(cluster_id=0, queries=cluster_queries)]
    
    results = summarizer.generate_batch_summaries(
        clusters_data=clusters_data,
        max_queries=10,
        concurrency=1,
    )
    
    assert 0 in results
    assert "title" in results[0]
    assert "description" in results[0]
    assert "sample_queries" in results[0]
    
    # Title should be related to machine learning
    title = results[0]["title"].lower()
    assert any(word in title for word in ["machine", "learning", "ai", "neural", "model"])


@pytest.mark.smoke
def test_summarize_multiple_clusters():
    """Test summarizing multiple clusters in parallel."""
    summarizer = ClusterSummarizer(
        model="openai/gpt-4o-mini",
        concurrency=2,
    )
    
    clusters_data = [
        ClusterData(cluster_id=0, queries=[
            "What is machine learning?",
            "Explain neural networks",
        ]),
        ClusterData(cluster_id=1, queries=[
            "How do I write a Python function?",
            "Explain Python decorators",
        ]),
    ]
    
    results = summarizer.generate_batch_summaries(
        clusters_data=clusters_data,
        max_queries=10,
        concurrency=2,
    )
    
    assert len(results) == 2
    assert 0 in results
    assert 1 in results
    
    # Both should have titles and descriptions
    for cluster_id in [0, 1]:
        assert results[cluster_id]["title"]
        assert results[cluster_id]["description"]
        assert len(results[cluster_id]["sample_queries"]) > 0


@pytest.mark.smoke
def test_summarize_with_contrast():
    """Test summarization with contrast neighbors."""
    summarizer = ClusterSummarizer(
        model="openai/gpt-4o-mini",
        concurrency=1,
    )
    
    clusters_data = [
        ClusterData(cluster_id=0, queries=[
            "What is Python?",
            "How to write Python code?",
        ]),
    ]
    
    # Test with contrast parameters
    results = summarizer.generate_batch_summaries(
        clusters_data=clusters_data,
        max_queries=10,
        concurrency=1,
        contrast_neighbors=2,
        contrast_examples=2,
        contrast_mode="neighbors",
    )
    
    assert 0 in results
    assert results[0]["title"]
    assert results[0]["description"]

