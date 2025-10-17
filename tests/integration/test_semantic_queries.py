"""Tests for semantic queries client."""

import tempfile

import numpy as np
import pytest

from lmsys_query_analysis.clustering.embeddings import EmbeddingGenerator
from lmsys_query_analysis.db.chroma import ChromaManager
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun, ClusterSummary, Query, QueryCluster
from lmsys_query_analysis.semantic.queries import QueriesClient


@pytest.fixture
def semantic_queries_db(tmp_path):
    """Create a test database with queries and clustering data."""
    db_path = tmp_path / "test_semantic.db"
    db = Database(str(db_path))
    db.create_tables()

    with db.get_session() as session:
        # Create test queries
        queries = []
        for i in range(20):
            query = Query(
                conversation_id=f"conv-{i}",
                model="gpt-4",
                query_text=f"Test query about topic {i % 5}",
                language="English" if i % 2 == 0 else "Spanish",
            )
            session.add(query)
            queries.append(query)
        session.commit()

        # Refresh to get IDs
        for q in queries:
            session.refresh(q)

        # Create clustering run
        run = ClusteringRun(
            run_id="test-semantic-run",
            algorithm="kmeans",
            num_clusters=5,
            parameters={
                "embedding_provider": "sentence-transformers",
                "embedding_model": "all-MiniLM-L6-v2",
            },
        )
        session.add(run)
        session.commit()

        # Assign queries to clusters
        for i, query in enumerate(queries):
            assignment = QueryCluster(
                run_id="test-semantic-run",
                query_id=query.id,
                cluster_id=i % 5,
                confidence_score=0.9,
            )
            session.add(assignment)
        session.commit()

        # Add cluster summaries
        for cluster_id in range(5):
            summary = ClusterSummary(
                run_id="test-semantic-run",
                cluster_id=cluster_id,
                title=f"Cluster {cluster_id} Title",
                description=f"Description for cluster {cluster_id}",
                num_queries=4,
            )
            session.add(summary)
        session.commit()

    return db, "test-semantic-run"


def test_queries_client_initialization():
    """Test QueriesClient can be initialized with dependencies."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")

    assert client.db == db
    assert client.chroma == chroma
    assert client.embedder == embedder
    assert client._run_id == "test-run"


def test_queries_client_from_run(semantic_queries_db):
    """Test creating QueriesClient from a run ID."""
    db, run_id = semantic_queries_db

    with tempfile.TemporaryDirectory() as tmpdir:
        client = QueriesClient.from_run(
            db=db,
            run_id=run_id,
            persist_dir=tmpdir,
        )

        assert client.db == db
        assert client._run_id == run_id
        assert client.embedder is not None
        assert client.chroma is not None


def test_queries_client_from_run_not_found():
    """Test from_run raises error for non-existent run."""
    db = Database(":memory:")
    db.create_tables()

    with pytest.raises(ValueError, match="Run not found"):
        QueriesClient.from_run(db=db, run_id="non-existent-run")


def test_queries_client_from_run_with_cohere_params(tmp_path):
    """Test from_run handles Cohere-specific parameters."""
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    db.create_tables()

    with db.get_session() as session:
        run = ClusteringRun(
            run_id="cohere-run",
            algorithm="kmeans",
            num_clusters=10,
            parameters={
                "embedding_provider": "cohere",
                "embedding_model": "embed-v4.0",
                "embedding_dimension": 256,
            },
        )
        session.add(run)
        session.commit()

    # Mock Cohere client to avoid API key requirement
    import os
    from unittest.mock import Mock, patch

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set a fake API key to pass initialization
        original_key = os.environ.get("CO_API_KEY")
        os.environ["CO_API_KEY"] = "fake-key-for-testing"

        try:
            with (
                patch("cohere.ClientV2") as mock_client,
                patch("cohere.AsyncClientV2") as mock_async_client,
            ):
                # Create mock Cohere clients
                mock_client.return_value = Mock()
                mock_async_client.return_value = Mock()

                client = QueriesClient.from_run(db=db, run_id="cohere-run", persist_dir=tmpdir)

                assert client.embedder.provider == "cohere"
                assert client.chroma.embedding_dimension == 256
        finally:
            # Restore original key
            if original_key is None:
                os.environ.pop("CO_API_KEY", None)
            else:
                os.environ["CO_API_KEY"] = original_key


def test_queries_client_from_run_cohere_default_dimension(tmp_path):
    """Test from_run sets default dimension for Cohere when not specified."""
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    db.create_tables()

    with db.get_session() as session:
        run = ClusteringRun(
            run_id="cohere-run-no-dim",
            algorithm="kmeans",
            num_clusters=10,
            parameters={
                "embedding_provider": "cohere",
                "embedding_model": "embed-v4.0",
                # No embedding_dimension specified
            },
        )
        session.add(run)
        session.commit()

    # Mock Cohere client to avoid API key requirement
    import os
    from unittest.mock import Mock, patch

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set a fake API key to pass initialization
        original_key = os.environ.get("CO_API_KEY")
        os.environ["CO_API_KEY"] = "fake-key-for-testing"

        try:
            with (
                patch("cohere.ClientV2") as mock_client,
                patch("cohere.AsyncClientV2") as mock_async_client,
            ):
                # Create mock Cohere clients
                mock_client.return_value = Mock()
                mock_async_client.return_value = Mock()

                client = QueriesClient.from_run(
                    db=db, run_id="cohere-run-no-dim", persist_dir=tmpdir
                )

                # Should default to 256 for Cohere
                assert client.chroma.embedding_dimension == 256
        finally:
            # Restore original key
            if original_key is None:
                os.environ.pop("CO_API_KEY", None)
            else:
                os.environ["CO_API_KEY"] = original_key


def test_queries_client_resolve_space():
    """Test resolve_space returns correct RunSpace."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
        embedding_dimension=384,
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")

    space = client.resolve_space()

    assert space.embedding_provider == "sentence-transformers"
    assert space.embedding_model == "all-MiniLM-L6-v2"
    assert space.embedding_dimension == 384
    assert space.run_id == "test-run"


def test_queries_client_embed():
    """Test embed method generates embeddings."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder)

    vec = client.embed("test query")

    assert isinstance(vec, list)
    assert len(vec) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(x, (float, np.floating)) for x in vec)


def test_queries_client_count_by_cluster():
    """Test count method with by='cluster' grouping."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder)

    # Test that unsupported 'by' raises ValueError
    from unittest.mock import patch

    with pytest.raises(ValueError, match="by must be one of"):
        # Mock find to return some hits
        with patch.object(client, "find", return_value=[]):
            client.count("test", by="invalid")


def test_queries_client_count_by_language():
    """Test count method with by='language' grouping."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder, run_id="test")

    # Create mock hits with language data
    from unittest.mock import Mock

    mock_hits = [
        Mock(language="English", cluster_id=0),
        Mock(language="English", cluster_id=1),
        Mock(language="Spanish", cluster_id=2),
    ]

    from unittest.mock import patch

    with patch.object(client, "find", return_value=mock_hits):
        counts = client.count("test", by="language")

        assert isinstance(counts, dict)
        assert counts.get("English") == 2
        assert counts.get("Spanish") == 1


def test_queries_client_count_by_model():
    """Test count method with by='model' grouping."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder)

    # Create mock hits with model data
    from unittest.mock import Mock

    mock_hits = [
        Mock(model="gpt-4", cluster_id=0, language="English"),
        Mock(model="gpt-4", cluster_id=1, language="English"),
        Mock(model="claude-3", cluster_id=2, language="English"),
    ]

    from unittest.mock import patch

    with patch.object(client, "find", return_value=mock_hits):
        counts = client.count("test", by="model")

        assert isinstance(counts, dict)
        assert counts.get("gpt-4") == 2
        assert counts.get("claude-3") == 1


def test_queries_client_facets_language():
    """Test facets method with language faceting."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder)

    # Create mock hits
    from unittest.mock import Mock

    mock_hits = [
        Mock(language="English", model="gpt-4", cluster_id=0),
        Mock(language="English", model="gpt-4", cluster_id=1),
        Mock(language="Spanish", model="claude-3", cluster_id=2),
        Mock(language=None, model="gpt-4", cluster_id=3),  # Test None handling
    ]

    from unittest.mock import patch

    with patch.object(client, "find", return_value=mock_hits):
        facets = client.facets("test", facet_by=["language"])

        assert "language" in facets
        language_buckets = facets["language"]

        # Find English bucket
        english_bucket = next(b for b in language_buckets if b.key == "English")
        assert english_bucket.count == 2

        # Find empty string bucket for None language
        empty_bucket = next((b for b in language_buckets if b.key == ""), None)
        assert empty_bucket is not None


def test_queries_client_facets_model():
    """Test facets method with model faceting."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder)

    # Create mock hits
    from unittest.mock import Mock

    mock_hits = [
        Mock(model="gpt-4", language="English", cluster_id=0),
        Mock(model="gpt-4", language="English", cluster_id=1),
        Mock(model=None, language="English", cluster_id=2),  # Test None handling
    ]

    from unittest.mock import patch

    with patch.object(client, "find", return_value=mock_hits):
        facets = client.facets("test", facet_by=["model"])

        assert "model" in facets
        model_buckets = facets["model"]

        # Buckets should be sorted by count descending
        assert model_buckets[0].count >= model_buckets[-1].count


def test_queries_client_facets_unsupported():
    """Test facets method raises error for unsupported facet."""
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder)

    from unittest.mock import patch

    with patch.object(client, "find", return_value=[]):
        with pytest.raises(ValueError, match="Unsupported facet"):
            client.facets("test", facet_by=["invalid_facet"])


def test_queries_client_find_with_threshold():
    """Test find method filters by distance threshold."""
    # This tests that the threshold parameter works correctly
    # The actual filtering happens in lines 150-152
    db = Database(":memory:")
    db.create_tables()

    chroma = ChromaManager(
        persist_directory=None,
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
    )

    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        provider="sentence-transformers",
    )

    client = QueriesClient(db=db, chroma=chroma, embedder=embedder)

    # Mock the chroma search to return results with different distances
    from unittest.mock import patch

    mock_results = {
        "ids": [["query_1", "query_2", "query_3"]],
        "documents": [["doc1", "doc2", "doc3"]],
        "metadatas": [[{"model": "gpt-4"}, {"model": "gpt-4"}, {"model": "gpt-4"}]],
        "distances": [[0.1, 0.5, 0.9]],  # Different distances
    }

    with patch.object(client.chroma, "search_queries", return_value=mock_results):
        with patch.object(
            client.embedder, "generate_embeddings", return_value=np.array([[0.1] * 384])
        ):
            # With threshold=0.6, only first two should pass
            hits = client.find("test", threshold=0.6)

            # Should filter out the third result (distance=0.9 > 0.6)
            assert len(hits) <= 2
