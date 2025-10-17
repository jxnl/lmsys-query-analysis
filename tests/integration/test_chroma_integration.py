"""Integration tests for ChromaDB manager with real ChromaDB instances."""

import tempfile
from pathlib import Path

from lmsys_query_analysis.db.chroma import (
    DEFAULT_CHROMA_PATH,
    ChromaManager,
)


def test_chroma_manager_initialization():
    """Test ChromaManager initialization with real ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir,
            embedding_model="test-model",
            embedding_provider="openai",
            embedding_dimension=1536,
        )

        assert manager.persist_directory == Path(tmpdir)
        assert manager.embedding_model == "test-model"
        assert manager.embedding_provider == "openai"
        assert manager.embedding_dimension == 1536
        assert manager.client is not None
        assert manager.queries_collection is not None
        assert manager.summaries_collection is not None


def test_chroma_manager_default_directory():
    """Test that ChromaManager uses default directory when none specified."""
    manager = ChromaManager()
    assert manager.persist_directory == DEFAULT_CHROMA_PATH
    assert DEFAULT_CHROMA_PATH.exists()


def test_chroma_manager_collection_naming():
    """Test that collection names are properly formatted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir,
            embedding_model="text-embedding-3-small",
            embedding_provider="openai",
            embedding_dimension=1536,
        )

        queries_name = manager.queries_collection.name
        summaries_name = manager.summaries_collection.name

        assert "queries" in queries_name
        assert "openai" in queries_name
        assert "text-embedding" in queries_name

        assert "summaries" in summaries_name
        assert "openai" in summaries_name


def test_chroma_manager_cohere_with_dimension():
    """Test that Cohere collections include dimension in name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir,
            embedding_model="embed-english-v3.0",
            embedding_provider="cohere",
            embedding_dimension=512,
        )

        queries_name = manager.queries_collection.name
        assert "512" in queries_name or "cohere" in queries_name


def test_chroma_manager_metadata():
    """Test that collections have proper metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir,
            embedding_model="test-model",
            embedding_provider="openai",
            embedding_dimension=1536,
        )

        queries_meta = manager.queries_collection.metadata
        assert queries_meta["embedding_model"] == "test-model"
        assert queries_meta["embedding_provider"] == "openai"
        assert queries_meta["embedding_dimension"] == 1536

        summaries_meta = manager.summaries_collection.metadata
        assert summaries_meta["embedding_model"] == "test-model"
        assert summaries_meta["embedding_provider"] == "openai"


def test_chroma_manager_count_queries():
    """Test counting queries in collection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir, embedding_model="test-model", embedding_provider="test"
        )

        count = manager.count_queries()
        assert count == 0


def test_chroma_manager_count_summaries():
    """Test counting summaries in collection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir, embedding_model="test-model", embedding_provider="test"
        )

        count = manager.count_summaries()
        assert count == 0


def test_chroma_manager_list_runs():
    """Test listing runs in summaries collection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir, embedding_model="test-model", embedding_provider="test"
        )

        runs = manager.list_runs_in_summaries()
        assert runs == []


def test_chroma_manager_list_all_collections():
    """Test listing all collections with metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir,
            embedding_model="test-model",
            embedding_provider="openai",
            embedding_dimension=1536,
        )

        collections = manager.list_all_collections()

        assert len(collections) >= 2

        for coll in collections:
            assert "name" in coll
            assert "count" in coll
            assert "metadata" in coll


def test_chroma_manager_persistence():
    """Test that collections persist across manager instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager1 = ChromaManager(
            persist_directory=tmpdir, embedding_model="test-model", embedding_provider="test"
        )
        queries_name1 = manager1.queries_collection.name

        manager2 = ChromaManager(
            persist_directory=tmpdir, embedding_model="test-model", embedding_provider="test"
        )
        queries_name2 = manager2.queries_collection.name

        assert queries_name1 == queries_name2


def test_chroma_manager_different_models():
    """Test that different models create different collections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager1 = ChromaManager(
            persist_directory=tmpdir, embedding_model="model-a", embedding_provider="test"
        )

        manager2 = ChromaManager(
            persist_directory=tmpdir, embedding_model="model-b", embedding_provider="test"
        )

        assert manager1.queries_collection.name != manager2.queries_collection.name


def test_chroma_manager_get_query_embeddings_map_empty():
    """Test getting embeddings map for queries when collection is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChromaManager(
            persist_directory=tmpdir, embedding_model="test-model", embedding_provider="test"
        )

        embeddings_map = manager.get_query_embeddings_map([1, 2, 3])

        assert embeddings_map == {}
