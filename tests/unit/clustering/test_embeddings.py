"""Tests for embedding generation functionality."""

import numpy as np
import pytest

from lmsys_query_analysis.clustering.embeddings import EmbeddingGenerator


def test_embedding_generator_initialization():
    """Test that EmbeddingGenerator initializes with correct parameters."""
    # Sentence transformers provider (no API key needed)
    embedder_st = EmbeddingGenerator(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
    )
    assert embedder_st.provider == "sentence-transformers"
    assert embedder_st.model_name == "all-MiniLM-L6-v2"

    # Verify concurrency parameter
    embedder_concurrent = EmbeddingGenerator(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        concurrency=50,
    )
    assert embedder_concurrent.concurrency == 50


def test_get_embedding_dim():
    """Test getting embedding dimensions for different models."""
    # Sentence transformers (needs to load model to get dim)
    st_gen = EmbeddingGenerator(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")
    # Should return 384 for all-MiniLM-L6-v2
    dim = st_gen.get_embedding_dim()
    assert isinstance(dim, int)
    assert dim == 384

    # Can call it multiple times and get same result
    dim2 = st_gen.get_embedding_dim()
    assert dim == dim2


def test_text_filtering():
    """Test that empty and whitespace texts are filtered correctly."""
    texts = [
        "valid text 1",
        "",
        "  ",
        "\t\n",
        "valid text 2",
        None if False else "valid text 3",  # Test that this doesn't break
    ]

    filtered_texts = []
    original_indices = []

    for i, text in enumerate(texts):
        if text and text.strip():
            filtered_texts.append(text.strip())
            original_indices.append(i)

    assert len(filtered_texts) == 3
    assert "valid text 1" in filtered_texts
    assert "valid text 2" in filtered_texts
    assert "valid text 3" in filtered_texts
    assert original_indices == [0, 4, 5]


def test_sentence_transformers_embeddings():
    """Test generating embeddings with sentence transformers."""
    embedder = EmbeddingGenerator(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")

    texts = ["test query 1", "test query 2"]
    embeddings = embedder.generate_embeddings(texts, batch_size=2, show_progress=False)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension
    assert embeddings.dtype == np.float32


def test_sentence_transformers_with_empty_texts():
    """Test sentence transformers handles empty texts correctly."""
    embedder = EmbeddingGenerator(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")

    texts = ["valid text", "", "  ", "another valid"]
    embeddings = embedder.generate_embeddings(texts, batch_size=4, show_progress=False)

    # Should return embeddings for all inputs
    assert embeddings.shape[0] == 4
    assert embeddings.shape[1] == 384

    # Empty texts should have zero embeddings
    assert np.allclose(embeddings[1], np.zeros(384))
    assert np.allclose(embeddings[2], np.zeros(384))

    # Valid texts should have non-zero embeddings
    assert not np.allclose(embeddings[0], np.zeros(384))
    assert not np.allclose(embeddings[3], np.zeros(384))


def test_sentence_transformers_batch_processing():
    """Test that batch processing works correctly."""
    embedder = EmbeddingGenerator(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")

    # Test with more texts than batch size
    texts = [f"test query {i}" for i in range(10)]
    embeddings = embedder.generate_embeddings(texts, batch_size=3, show_progress=False)

    assert embeddings.shape == (10, 384)

    # Verify all embeddings are unique (not all zeros)
    for i, emb in enumerate(embeddings):
        assert not np.allclose(emb, np.zeros(384)), f"Embedding {i} is all zeros"


def test_embedding_consistency():
    """Test that same text produces same embedding."""
    embedder = EmbeddingGenerator(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")

    text = "consistent test query"

    # Generate embeddings twice
    emb1 = embedder.generate_embeddings([text], show_progress=False)
    emb2 = embedder.generate_embeddings([text], show_progress=False)

    # Should be identical (deterministic)
    assert np.allclose(emb1, emb2)


def test_load_model_lazy():
    """Test that model loading is lazy."""
    embedder = EmbeddingGenerator(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")

    # Model should not be loaded yet
    assert embedder.model is None

    # After generating embeddings, model should be loaded
    embedder.generate_embeddings(["test"], show_progress=False)
    assert embedder.model is not None


def test_multiple_calls_reuse_model():
    """Test that multiple embedding calls reuse the same model."""
    embedder = EmbeddingGenerator(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")

    # First call loads model
    embedder.generate_embeddings(["test 1"], show_progress=False)
    model1 = embedder.model

    # Second call should reuse model
    embedder.generate_embeddings(["test 2"], show_progress=False)
    model2 = embedder.model

    assert model1 is model2  # Same object


@pytest.mark.anyio(backend="asyncio")
async def test_async_context_detection():
    """Test that async context is detected correctly."""
    import asyncio

    # Inside async function, should detect running loop
    try:
        loop = asyncio.get_running_loop()
        assert loop is not None, "Should have running loop in async test"
    except RuntimeError:
        pytest.fail("Should be able to get running loop")


def test_embedding_generator_with_all_empty_texts():
    """Test handling when all input texts are empty."""
    embedder = EmbeddingGenerator(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")

    # All empty texts
    texts = ["", "  ", "\t", "\n"]
    embeddings = embedder.generate_embeddings(texts, show_progress=False)

    # Should return zero embeddings for all
    assert embeddings.shape == (4, 384)
    for emb in embeddings:
        assert np.allclose(emb, np.zeros(384))
