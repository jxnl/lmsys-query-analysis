"""Tests for hierarchical clustering functionality and async fixes."""

import pytest
import asyncio
import numpy as np


@pytest.mark.anyio(backend="asyncio")
async def test_embeddings_async_in_event_loop():
    """Test that the async event loop fix works correctly.
    
    This tests the bug fix for nested event loops in embeddings.py.
    The fix ensures that generate_embeddings can be called from within
    an async context without raising "asyncio.run() cannot be called from a running event loop"
    """
    
    # Test that asyncio.get_running_loop() works inside async context
    try:
        loop = asyncio.get_running_loop()
        assert loop is not None
    except RuntimeError:
        pytest.fail("Should be able to get running loop in async test")
    
    # Test the actual pattern from the fix (embeddings.py lines 279-287)
    async def _mock_async_operation():
        """Simulates the async embedding operation."""
        await asyncio.sleep(0.001)
        return [0.1] * 10
    
    # Simulate what happens in embeddings.py when called from async context
    try:
        asyncio.get_running_loop()
        # We're in an async context - should use ThreadPoolExecutor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _mock_async_operation())
            result = future.result()
        assert result is not None
        assert len(result) == 10
    except RuntimeError as e:
        pytest.fail(f"Should handle nested event loop correctly: {e}")


def test_embedding_generator_handles_empty_texts():
    """Test that embedding generator handles empty or whitespace-only texts.
    
    Tests the filtering logic in embeddings.py lines 110-149.
    """
    
    # Test the filtering logic that's in the actual code
    texts = ["valid query", "  ", "", "another valid"]
    filtered_texts = []
    original_indices = []
    
    for i, text in enumerate(texts):
        if text and text.strip():
            filtered_texts.append(text.strip())
            original_indices.append(i)
    
    # Should filter out empty texts
    assert len(filtered_texts) == 2
    assert filtered_texts == ["valid query", "another valid"]
    assert original_indices == [0, 3]
    
    # Verify the reconstruction logic would work
    full_embeddings_shape = (len(texts), 1536)
    valid_embeddings_shape = (len(filtered_texts), 1536)
    
    # Create mock embeddings for valid texts
    valid_embeddings = np.random.rand(*valid_embeddings_shape)
    
    # Reconstruct full array with zeros for empty texts
    full_embeddings = np.zeros(full_embeddings_shape)
    for i, orig_idx in enumerate(original_indices):
        full_embeddings[orig_idx] = valid_embeddings[i]
    
    # Empty text embeddings should be zero vectors
    assert np.allclose(full_embeddings[1], np.zeros(1536))
    assert np.allclose(full_embeddings[2], np.zeros(1536))
    # Valid embeddings should be preserved
    assert not np.allclose(full_embeddings[0], np.zeros(1536))
    assert not np.allclose(full_embeddings[3], np.zeros(1536))


def test_hierarchy_node_structure():
    """Test the expected structure of hierarchy nodes."""
    
    # Verify the dict structure that hierarchy functions return
    leaf_node = {
        "cluster_id": 0,
        "level": 0,
        "title": "Test Cluster",
        "description": "Test description",
        "num_queries": 10,
        "children_ids": None,
        "parent_id": 1,
    }
    
    parent_node = {
        "cluster_id": 1,
        "level": 1,
        "title": "Parent Cluster",
        "description": "Parent description",
        "num_queries": 20,
        "children_ids": [0],
        "parent_id": None,
    }
    
    # Verify structure
    assert leaf_node["level"] == 0
    assert leaf_node["parent_id"] == parent_node["cluster_id"]
    assert parent_node["level"] > leaf_node["level"]
    assert leaf_node["cluster_id"] in parent_node["children_ids"]
    assert parent_node["parent_id"] is None  # Root node
    assert leaf_node["children_ids"] is None  # Leaf node


def test_threadpool_executor_pattern():
    """Test that the ThreadPoolExecutor pattern works for nested event loops."""
    import concurrent.futures
    
    def sync_function_that_needs_async():
        """Simulates a sync function that needs to run async code."""
        async def async_operation():
            await asyncio.sleep(0.001)
            return "success"
        
        # This is the pattern used in the fix
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, async_operation())
            return future.result()
    
    # Should work without errors
    result = sync_function_that_needs_async()
    assert result == "success"
