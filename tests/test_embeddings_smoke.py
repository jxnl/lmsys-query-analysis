import os
import numpy as np
import pytest

from lmsys_query_analysis.clustering.embeddings import EmbeddingGenerator


def has_env(var: str) -> bool:
    return bool(os.getenv(var))


@pytest.mark.smoke
@pytest.mark.skipif(not has_env("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_embeddings_smoke():
    texts = [
        "How to write a Python function?",
        "What is the capital of France?",
        "Vector databases for semantic search",
    ]
    eg = EmbeddingGenerator(
        model_name="text-embedding-3-small",
        provider="openai",
        request_timeout=60.0,
        concurrency=5,
    )

    emb = eg.generate_embeddings(texts, batch_size=3, show_progress=False)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == len(texts)
    assert emb.shape[1] == eg.get_embedding_dim()
    # Basic sanity: non-zero vectors
    norms = np.linalg.norm(emb, axis=1)
    assert np.all(norms > 0)


@pytest.mark.smoke
@pytest.mark.skipif(not has_env("COHERE_API_KEY"), reason="COHERE_API_KEY not set")
def test_cohere_embeddings_smoke_clustering_512():
    texts = [
        "Summarize LLM usage patterns",
        "Group similar questions using embeddings",
        "Contrast clusters by representative queries",
    ]
    # Use Matryoshka dimension 512 for a quick, lightweight smoke
    eg = EmbeddingGenerator(
        model_name="embed-v4.0",
        provider="cohere",
        output_dimension=512,
        request_timeout=60.0,
        concurrency=5,
    )

    emb = eg.generate_embeddings(texts, batch_size=3, show_progress=False)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (len(texts), 512)
    norms = np.linalg.norm(emb, axis=1)
    assert np.all(norms > 0)


@pytest.mark.smoke
@pytest.mark.skipif(not has_env("COHERE_API_KEY"), reason="COHERE_API_KEY not set")
def test_summary_embeddings_use_clustering_mode_cohere():
    # We can't inspect the API payload here, but our generator sets input_type='clustering'
    # for Cohere. This smoke ensures summary-like texts embed successfully with Cohere.
    summaries = [
        "Title: Code Debugging\n\nDescription: Users ask to debug Python exceptions and tracebacks.",
        "Title: Prompt Engineering\n\nDescription: Iterative refinement of system and user prompts.",
    ]
    eg = EmbeddingGenerator(
        model_name="embed-v4.0",
        provider="cohere",
        output_dimension=512,
        request_timeout=60.0,
        concurrency=5,
    )
    emb = eg.generate_embeddings(summaries, batch_size=2, show_progress=False)
    assert emb.shape == (len(summaries), 512)
    assert np.all(np.linalg.norm(emb, axis=1) > 0)

