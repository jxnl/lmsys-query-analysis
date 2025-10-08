"""Unit tests for client_factory."""

import pytest
from lmsys_query_analysis.cli.helpers.client_factory import (
    parse_embedding_model,
    get_embedding_dimension,
)


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

