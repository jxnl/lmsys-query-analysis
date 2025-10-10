"""Advanced tests for embedding generation with proper API mocking."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from lmsys_query_analysis.clustering.embeddings import EmbeddingGenerator


@pytest.fixture
def mock_openai_env(monkeypatch):
    """Set fake OpenAI API key."""
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-test-key')


@pytest.fixture
def mock_cohere_env(monkeypatch):
    """Set fake Cohere API key."""
    monkeypatch.setenv('CO_API_KEY', 'fake-test-key')


def test_openai_embeddings_with_mocked_responses(mock_openai_env):
    """Test OpenAI embedding generation by mocking the async client's response."""
    
    with patch('openai.AsyncOpenAI') as mock_async_openai:
        # Create a mock async client
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        # Mock the embeddings.create coroutine
        async def mock_create(**kwargs):
            # Return mock response matching OpenAI's structure
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1] * 1536) for _ in kwargs['input']
            ]
            return mock_response
        
        mock_client.embeddings.create = mock_create
        
        # Also mock sync client (even though we don't use it)
        with patch('openai.OpenAI'):
            gen = EmbeddingGenerator(
                provider="openai",
                model_name="text-embedding-3-small",
            )
            
            texts = ["query 1", "query 2"]
            embeddings = gen.generate_embeddings(texts, batch_size=2, show_progress=False)
            
            assert embeddings.shape == (2, 1536)
            assert isinstance(embeddings, np.ndarray)


def test_openai_batch_processing_multiple_batches(mock_openai_env):
    """Test that OpenAI processes multiple batches correctly."""
    
    call_count = 0
    
    with patch('openai.AsyncOpenAI') as mock_async_openai, \
         patch('openai.OpenAI'):
        
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[float(call_count)] * 1536) for _ in kwargs['input']]
            return mock_response
        
        mock_client.embeddings.create = mock_create
        
        gen = EmbeddingGenerator(
            provider="openai",
            model_name="text-embedding-3-small",
        )
        
        # 5 texts with batch_size=2 should make 3 batches
        texts = [f"query {i}" for i in range(5)]
        embeddings = gen.generate_embeddings(texts, batch_size=2, show_progress=False)
        
        assert embeddings.shape == (5, 1536)
        # Verify multiple batches were processed
        assert call_count == 3


def test_cohere_embeddings_with_mocked_responses(mock_cohere_env):
    """Test Cohere embedding generation by mocking the async client's response."""
    
    with patch('cohere.AsyncClientV2') as mock_async_cohere:
        mock_client = Mock()
        mock_async_cohere.return_value = mock_client
        
        # Create proper async mock
        async def mock_embed(**kwargs):
            # Create mock response with proper structure
            mock_resp = Mock()
            mock_resp.embeddings = Mock()
            # Return list of embeddings matching input size (use .float not .float_)
            mock_resp.embeddings.float = [[0.1] * 256 for _ in kwargs['texts']]
            return mock_resp
        
        # Assign as a coroutine
        mock_client.embed = AsyncMock(side_effect=mock_embed)
        
        gen = EmbeddingGenerator(
            provider="cohere",
            model_name="embed-v4.0",
            output_dimension=256,
        )
        
        texts = ["query 1", "query 2"]
        embeddings = gen.generate_embeddings(texts, batch_size=2, show_progress=False)
        
        assert embeddings.shape == (2, 256)
        assert isinstance(embeddings, np.ndarray)


def test_cohere_batch_processing(mock_cohere_env):
    """Test Cohere processes batches correctly."""
    
    batch_sizes_seen = []
    
    with patch('cohere.AsyncClientV2') as mock_async_cohere:
        mock_client = Mock()
        mock_async_cohere.return_value = mock_client
        
        async def mock_embed(**kwargs):
            batch_sizes_seen.append(len(kwargs['texts']))
            mock_resp = Mock()
            mock_resp.embeddings = Mock()
            mock_resp.embeddings.float = [[0.1] * 256 for _ in kwargs['texts']]
            return mock_resp
        
        mock_client.embed = AsyncMock(side_effect=mock_embed)
        
        gen = EmbeddingGenerator(
            provider="cohere",
            model_name="embed-v4.0",
            output_dimension=256,
        )
        
        # 5 texts with batch_size=2
        texts = [f"query {i}" for i in range(5)]
        embeddings = gen.generate_embeddings(texts, batch_size=2, show_progress=False)
        
        assert embeddings.shape == (5, 256)
        # Batches might be processed concurrently, so just check total
        assert sum(batch_sizes_seen) == 5


def test_embedding_filters_empty_strings(mock_openai_env):
    """Test that empty strings are filtered before API calls."""
    
    actual_inputs = []
    
    with patch('openai.AsyncOpenAI') as mock_async_openai, \
         patch('openai.OpenAI'):
        
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        async def mock_create(**kwargs):
            actual_inputs.extend(kwargs['input'])
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in kwargs['input']]
            return mock_response
        
        mock_client.embeddings.create = mock_create
        
        gen = EmbeddingGenerator(
            provider="openai",
            model_name="text-embedding-3-small",
        )
        
        # Mix of valid and empty texts
        texts = ["valid 1", "", "  ", "valid 2"]
        embeddings = gen.generate_embeddings(texts, batch_size=10, show_progress=False)
        
        # Should return embeddings for all 4 inputs
        assert embeddings.shape == (4, 1536)
        
        # But only 2 texts should be sent to API
        assert len(actual_inputs) == 2
        assert "valid 1" in actual_inputs
        assert "valid 2" in actual_inputs
        
        # Empty positions should have zero embeddings
        assert np.allclose(embeddings[1], np.zeros(1536))
        assert np.allclose(embeddings[2], np.zeros(1536))


def test_cohere_invalid_model_validation():
    """Test that Cohere validates model names."""
    with pytest.raises(ValueError, match="Only Cohere v4 supported"):
        EmbeddingGenerator(
            provider="cohere",
            model_name="embed-english-v3.0",  # Invalid v3 model
        )


def test_cohere_invalid_dimension_validation():
    """Test that Cohere validates output dimensions."""
    with pytest.raises(ValueError, match="output_dimension must be one of"):
        EmbeddingGenerator(
            provider="cohere",
            model_name="embed-v4.0",
            output_dimension=999,  # Invalid dimension
        )


def test_cohere_dimension_defaults(mock_cohere_env):
    """Test that Cohere defaults to 256 dimensions."""
    
    with patch('cohere.AsyncClientV2'):
        gen = EmbeddingGenerator(
            provider="cohere",
            model_name="embed-v4.0",
            # No output_dimension specified
        )
        
        # Should default to 256
        assert gen.output_dimension == 256


def test_openai_get_embedding_dimensions(mock_openai_env):
    """Test dimension detection for OpenAI models."""
    
    with patch('openai.OpenAI'), patch('openai.AsyncOpenAI'):
        gen_small = EmbeddingGenerator(
            provider="openai",
            model_name="text-embedding-3-small",
        )
        assert gen_small.get_embedding_dim() == 1536
        
        gen_large = EmbeddingGenerator(
            provider="openai",
            model_name="text-embedding-3-large",
        )
        assert gen_large.get_embedding_dim() == 3072
        
        gen_ada = EmbeddingGenerator(
            provider="openai",
            model_name="text-embedding-ada-002",
        )
        assert gen_ada.get_embedding_dim() == 1536


def test_cohere_get_embedding_dimensions(mock_cohere_env):
    """Test dimension detection for Cohere models."""
    
    with patch('cohere.AsyncClientV2'):
        gen_256 = EmbeddingGenerator(
            provider="cohere",
            model_name="embed-v4.0",
            output_dimension=256,
        )
        assert gen_256.get_embedding_dim() == 256
        
        gen_1024 = EmbeddingGenerator(
            provider="cohere",
            model_name="embed-v4.0",
            output_dimension=1024,
        )
        assert gen_1024.get_embedding_dim() == 1024


def test_openai_async_retry_logic(mock_openai_env):
    """Test retry logic for transient failures."""
    
    attempt_count = 0
    
    with patch('openai.AsyncOpenAI') as mock_async_openai, \
         patch('openai.OpenAI'):
        
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        async def mock_create(**kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                # Fail first attempt
                raise Exception("Transient error")
            # Succeed on retry
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in kwargs['input']]
            return mock_response
        
        mock_client.embeddings.create = mock_create
        
        gen = EmbeddingGenerator(
            provider="openai",
            model_name="text-embedding-3-small",
        )
        
        texts = ["test"]
        embeddings = gen.generate_embeddings(texts, batch_size=1, show_progress=False)
        
        # Should succeed after retry
        assert embeddings.shape == (1, 1536)
        assert attempt_count == 2


def test_all_empty_texts_returns_zeros(mock_openai_env):
    """Test that all empty texts returns zero embeddings without API call."""
    
    api_called = False
    
    with patch('openai.AsyncOpenAI') as mock_async_openai, \
         patch('openai.OpenAI'):
        
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        async def mock_create(**kwargs):
            nonlocal api_called
            api_called = True
            return Mock()
        
        mock_client.embeddings.create = mock_create
        
        gen = EmbeddingGenerator(
            provider="openai",
            model_name="text-embedding-3-small",
        )
        
        # All empty texts
        texts = ["", "  ", "\t", "\n"]
        embeddings = gen.generate_embeddings(texts, batch_size=10, show_progress=False)
        
        # Should return zeros without calling API
        assert embeddings.shape == (4, 1536)
        assert np.allclose(embeddings, np.zeros((4, 1536)))
        assert not api_called  # API should NOT be called
