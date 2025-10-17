"""Smoke tests for embedding generation with real API calls."""

import pytest

from lmsys_query_analysis.clustering.embeddings import EmbeddingGenerator


@pytest.mark.smoke
def test_openai_embeddings():
    """Test OpenAI embedding generation."""
    gen = EmbeddingGenerator(
        model_name="text-embedding-3-small",
        provider="openai",
    )

    texts = [
        "What is machine learning?",
        "How do I write a Python function?",
    ]

    embeddings = gen.generate_embeddings(texts, batch_size=2, show_progress=False)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0  # Has dimensions
    assert isinstance(embeddings[0][0], float)


@pytest.mark.smoke
def test_cohere_embeddings():
    """Test Cohere embedding generation."""
    import os

    if not os.getenv("CO_API_KEY"):
        pytest.skip("CO_API_KEY not set")

    gen = EmbeddingGenerator(
        model_name="embed-v4.0",
        provider="cohere",
        output_dimension=256,
    )

    texts = [
        "Explain neural networks",
        "What is async/await in Python?",
    ]

    embeddings = gen.generate_embeddings(texts, batch_size=2, show_progress=False)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256  # Cohere with dimension specified


@pytest.mark.smoke
def test_embedding_batch_processing():
    """Test that batch processing works with real API."""
    gen = EmbeddingGenerator(
        model_name="text-embedding-3-small",
        provider="openai",
    )

    # Generate embeddings for larger batch
    texts = [f"Test query number {i}" for i in range(10)]
    embeddings = gen.generate_embeddings(texts, batch_size=5, show_progress=False)

    assert len(embeddings) == 10
    # All embeddings should have same dimension
    dims = [len(e) for e in embeddings]
    assert len(set(dims)) == 1  # All same dimension


@pytest.mark.smoke
def test_embedding_consistency():
    """Test that same text produces similar embeddings."""
    gen = EmbeddingGenerator(
        model_name="text-embedding-3-small",
        provider="openai",
    )

    text = "What is machine learning?"

    # Generate embedding twice
    emb1 = gen.generate_embeddings([text], batch_size=1, show_progress=False)[0]
    emb2 = gen.generate_embeddings([text], batch_size=1, show_progress=False)[0]

    # Should be identical
    import numpy as np

    similarity = np.dot(emb1, emb2)
    assert similarity > 0.99  # Very high similarity for identical text
