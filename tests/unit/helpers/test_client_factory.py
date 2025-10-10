"""Unit tests for client_factory."""

import pytest
import tempfile
from lmsys_query_analysis.cli.helpers.client_factory import (
    parse_embedding_model,
    get_embedding_dimension,
    create_chroma_client,
    create_embedding_generator,
    create_queries_client,
    create_clusters_client,
)
from lmsys_query_analysis.db.connection import Database


def test_parse_embedding_model_valid():
    """Test parsing valid embedding model string."""
    model, provider = parse_embedding_model("cohere/embed-v4.0")
    
    assert provider == "cohere"
    assert model == "embed-v4.0"


def test_parse_embedding_model_openai():
    """Test parsing OpenAI model string."""
    model, provider = parse_embedding_model("openai/text-embedding-3-small")
    
    assert provider == "openai"
    assert model == "text-embedding-3-small"


def test_parse_embedding_model_with_slash_in_model():
    """Test parsing model string with slash in model name."""
    model, provider = parse_embedding_model("provider/model/with/slashes")
    
    assert provider == "provider"
    assert model == "model/with/slashes"


def test_parse_embedding_model_invalid_format():
    """Test parsing invalid format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid embedding model format"):
        parse_embedding_model("invalid-format")


def test_parse_embedding_model_empty():
    """Test parsing empty string raises ValueError."""
    with pytest.raises(ValueError):
        parse_embedding_model("")


def test_get_embedding_dimension_cohere():
    """Test getting dimension for Cohere provider."""
    dim = get_embedding_dimension("cohere")
    
    assert dim == 256


def test_get_embedding_dimension_openai():
    """Test getting dimension for OpenAI provider (returns None for default)."""
    dim = get_embedding_dimension("openai")
    
    assert dim is None


def test_get_embedding_dimension_other():
    """Test getting dimension for other providers."""
    dim = get_embedding_dimension("anthropic")
    
    assert dim is None


def test_get_embedding_dimension_case_sensitive():
    """Test that provider name is case-sensitive."""
    dim = get_embedding_dimension("Cohere")
    
    # Should not match "cohere", so returns None
    assert dim is None


def test_create_chroma_client_cohere():
    """Test creating ChromaDB client for Cohere."""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_chroma_client(
            chroma_path=tmpdir,
            embedding_model="embed-v4.0",
            embedding_provider="cohere"
        )
        
        assert client is not None
        assert client.embedding_model == "embed-v4.0"
        assert client.embedding_provider == "cohere"
        assert client.embedding_dimension == 256  # Cohere default


def test_create_chroma_client_openai():
    """Test creating ChromaDB client for OpenAI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_chroma_client(
            chroma_path=tmpdir,
            embedding_model="text-embedding-3-small",
            embedding_provider="openai"
        )
        
        assert client is not None
        assert client.embedding_model == "text-embedding-3-small"
        assert client.embedding_provider == "openai"
        assert client.embedding_dimension is None  # OpenAI uses provider default


def test_create_embedding_generator_sentence_transformers():
    """Test creating embedding generator for sentence-transformers."""
    generator = create_embedding_generator(
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers"
    )
    
    assert generator is not None
    assert generator.model_name == "all-MiniLM-L6-v2"
    assert generator.provider == "sentence-transformers"
    assert generator.output_dimension is None


def test_create_queries_client_without_run_id():
    """Test creating queries client without run_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(":memory:")
        db.create_tables()
        
        # Use sentence-transformers which doesn't require API keys
        client = create_queries_client(
            db=db,
            run_id=None,
            embedding_model_string="sentence-transformers/all-MiniLM-L6-v2",
            chroma_path=tmpdir
        )
        
        assert client is not None
        assert client.db == db


def test_create_queries_client_with_invalid_model_format():
    """Test that invalid model format raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(":memory:")
        db.create_tables()
        
        with pytest.raises(ValueError, match="Invalid embedding model format"):
            create_queries_client(
                db=db,
                run_id=None,
                embedding_model_string="invalid-format",
                chroma_path=tmpdir
            )


def test_create_clusters_client_without_run_id():
    """Test creating clusters client without run_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(":memory:")
        db.create_tables()
        
        # Use sentence-transformers which doesn't require API keys
        client = create_clusters_client(
            db=db,
            run_id=None,
            embedding_model_string="sentence-transformers/all-MiniLM-L6-v2",
            chroma_path=tmpdir
        )
        
        assert client is not None
        assert client.db == db


def test_create_clusters_client_with_different_providers():
    """Test creating clusters client with different providers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(":memory:")
        db.create_tables()
        
        # Test with sentence-transformers
        client = create_clusters_client(
            db=db,
            run_id=None,
            embedding_model_string="sentence-transformers/all-MiniLM-L6-v2",
            chroma_path=tmpdir
        )
        
        assert client is not None

