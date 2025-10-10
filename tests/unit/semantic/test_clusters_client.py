"""Unit tests for ClustersClient."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun
from lmsys_query_analysis.db.chroma import ChromaManager
from lmsys_query_analysis.clustering.embeddings import EmbeddingGenerator
from lmsys_query_analysis.semantic.clusters import ClustersClient


def test_clusters_client_initialization():
    """Test ClustersClient can be initialized with dependencies."""
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")
    
    assert client.db == db
    assert client.chroma == chroma
    assert client.embedder == embedder
    assert client._run_id == "test-run"


def test_clusters_client_initialization_without_run_id():
    """Test ClustersClient can be initialized without run_id."""
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder)
    
    assert client._run_id is None


def test_clusters_client_from_run(tmp_path):
    """Test creating ClustersClient from a run ID."""
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    db.create_tables()
    
    # Create a clustering run
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="test-run",
            algorithm="kmeans",
            num_clusters=10,
            parameters={
                "embedding_provider": "sentence-transformers",
                "embedding_model": "all-MiniLM-L6-v2",
            },
        )
        session.add(run)
        session.commit()
    
    client = ClustersClient.from_run(
        db=db,
        run_id="test-run",
        persist_dir=tmp_path,
    )
    
    assert client.db == db
    assert client._run_id == "test-run"
    assert client.embedder is not None
    assert client.chroma is not None


def test_clusters_client_from_run_not_found():
    """Test from_run raises error for non-existent run."""
    db = Database(":memory:")
    db.create_tables()
    
    with pytest.raises(ValueError, match="Run not found"):
        ClustersClient.from_run(db=db, run_id="non-existent-run")


def test_clusters_client_from_run_with_cohere_params(tmp_path):
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
    with patch('cohere.ClientV2') as mock_client, \
         patch('cohere.AsyncClientV2') as mock_async_client:
        
        original_key = os.environ.get('CO_API_KEY')
        os.environ['CO_API_KEY'] = 'fake-key-for-testing'
        
        try:
            mock_client.return_value = Mock()
            mock_async_client.return_value = Mock()
            
            client = ClustersClient.from_run(db=db, run_id="cohere-run", persist_dir=tmp_path)
            
            assert client.embedder.provider == "cohere"
            assert client.chroma.embedding_dimension == 256
        finally:
            if original_key is None:
                os.environ.pop('CO_API_KEY', None)
            else:
                os.environ['CO_API_KEY'] = original_key


def test_clusters_client_from_run_cohere_default_dimension(tmp_path):
    """Test from_run sets default dimension for Cohere when not specified."""
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
                # No embedding_dimension specified
            },
        )
        session.add(run)
        session.commit()
    
    # Mock Cohere client
    import os
    with patch('cohere.ClientV2') as mock_client, \
         patch('cohere.AsyncClientV2') as mock_async_client:
        
        original_key = os.environ.get('CO_API_KEY')
        os.environ['CO_API_KEY'] = 'fake-key-for-testing'
        
        try:
            mock_client.return_value = Mock()
            mock_async_client.return_value = Mock()
            
            client = ClustersClient.from_run(db=db, run_id="cohere-run", persist_dir=tmp_path)
            
            # Should default to 256 for Cohere
            assert client.chroma.embedding_dimension == 256
        finally:
            if original_key is None:
                os.environ.pop('CO_API_KEY', None)
            else:
                os.environ['CO_API_KEY'] = original_key


def test_clusters_client_resolve_space():
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")
    
    space = client.resolve_space()
    
    assert space.embedding_provider == "sentence-transformers"
    assert space.embedding_model == "all-MiniLM-L6-v2"
    assert space.embedding_dimension == 384
    assert space.run_id == "test-run"


def test_clusters_client_find_basic():
    """Test find method returns cluster hits."""
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")
    
    # Mock the chroma search and embedder
    mock_results = {
        "ids": [["cluster_test-run_1", "cluster_test-run_2"]],
        "documents": [["Cluster 1 summary", "Cluster 2 summary"]],
        "metadatas": [[
            {"cluster_id": 1, "title": "Cluster 1", "description": "Desc 1", "num_queries": 10},
            {"cluster_id": 2, "title": "Cluster 2", "description": "Desc 2", "num_queries": 20},
        ]],
        "distances": [[0.1, 0.3]],
    }
    
    with patch.object(client.chroma, 'search_cluster_summaries', return_value=mock_results):
        with patch.object(client.embedder, 'generate_embeddings', return_value=np.array([[0.1] * 384])):
            hits = client.find(text="test query", run_id="test-run", top_k=5)
    
    assert len(hits) == 2
    assert hits[0].cluster_id == 1
    assert hits[0].title == "Cluster 1"
    assert hits[0].distance == 0.1
    assert hits[1].cluster_id == 2
    assert hits[1].title == "Cluster 2"


def test_clusters_client_find_with_alias_filter():
    """Test find method filters by alias."""
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")
    
    # Mock results with different aliases
    mock_results = {
        "ids": [["cluster_test-run_1", "cluster_test-run_2", "cluster_test-run_3"]],
        "documents": [["Summary 1", "Summary 2", "Summary 3"]],
        "metadatas": [[
            {"cluster_id": 1, "alias": "v1", "title": "Cluster 1", "num_queries": 10},
            {"cluster_id": 2, "alias": "v2", "title": "Cluster 2", "num_queries": 20},
            {"cluster_id": 3, "alias": "v1", "title": "Cluster 3", "num_queries": 15},
        ]],
        "distances": [[0.1, 0.2, 0.3]],
    }
    
    with patch.object(client.chroma, 'search_cluster_summaries', return_value=mock_results):
        with patch.object(client.embedder, 'generate_embeddings', return_value=np.array([[0.1] * 384])):
            hits = client.find(text="test", run_id="test-run", alias="v1")
    
    # Should only return clusters with alias "v1"
    assert len(hits) == 2
    assert all(h.cluster_id in {1, 3} for h in hits)


def test_clusters_client_find_with_summary_run_id_filter():
    """Test find method filters by summary_run_id."""
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")
    
    # Mock results with different summary_run_ids
    mock_results = {
        "ids": [["cluster_1", "cluster_2"]],
        "documents": [["Summary 1", "Summary 2"]],
        "metadatas": [[
            {"cluster_id": 1, "summary_run_id": "s1", "title": "Cluster 1", "num_queries": 10},
            {"cluster_id": 2, "summary_run_id": "s2", "title": "Cluster 2", "num_queries": 20},
        ]],
        "distances": [[0.1, 0.2]],
    }
    
    with patch.object(client.chroma, 'search_cluster_summaries', return_value=mock_results):
        with patch.object(client.embedder, 'generate_embeddings', return_value=np.array([[0.1] * 384])):
            hits = client.find(text="test", run_id="test-run", summary_run_id="s1")
    
    # Should only return cluster with summary_run_id "s1"
    assert len(hits) == 1
    assert hits[0].cluster_id == 1


def test_clusters_client_count_with_text():
    """Test count method with text search."""
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")
    
    # Mock find to return some hits
    with patch.object(client, 'find', return_value=[Mock(), Mock(), Mock()]):
        count = client.count(text="test query", run_id="test-run")
    
    assert count == 3


def test_clusters_client_count_without_text():
    """Test count method without text uses chroma count."""
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")
    
    # Mock chroma count_summaries
    with patch.object(client.chroma, 'count_summaries', return_value=42):
        with patch.object(client.chroma, 'search_cluster_summaries', return_value={"ids": [[]]}):
            count = client.count(text=None, run_id="test-run")
    
    assert count == 42


def test_clusters_client_find_with_top_k_limit():
    """Test find method respects top_k limit."""
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
    
    client = ClustersClient(db=db, chroma=chroma, embedder=embedder, run_id="test-run")
    
    # Mock results with 5 clusters
    mock_results = {
        "ids": [["c1", "c2", "c3", "c4", "c5"]],
        "documents": [["S1", "S2", "S3", "S4", "S5"]],
        "metadatas": [[
            {"cluster_id": i, "title": f"C{i}", "num_queries": 10}
            for i in range(1, 6)
        ]],
        "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
    }
    
    with patch.object(client.chroma, 'search_cluster_summaries', return_value=mock_results):
        with patch.object(client.embedder, 'generate_embeddings', return_value=np.array([[0.1] * 384])):
            hits = client.find(text="test", run_id="test-run", top_k=3)
    
    # Should only return top 3
    assert len(hits) == 3

