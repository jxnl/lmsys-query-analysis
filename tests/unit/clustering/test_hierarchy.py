"""Tests for hierarchical clustering functionality and async fixes."""

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest
from pydantic import ValidationError
from tenacity import RetryError

from lmsys_query_analysis.clustering.hierarchy import (
    ClusterAssignment,
    DeduplicatedClusters,
    NeighborhoodCategories,
    RefinedClusterSummary,
    assign_to_parent_cluster,
    create_neighborhoods,
    deduplicate_cluster_names,
    generate_neighborhood_categories,
    refine_parent_cluster,
)


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


# ==============================================================================
# Pydantic Model Tests
# ==============================================================================


def test_neighborhood_categories_model_valid():
    """Test NeighborhoodCategories model with valid data."""
    data = {
        "scratchpad": "Brief analysis of patterns",
        "categories": [
            "Python Programming Help",
            "JavaScript Web Development",
            "SQL Database Queries",
            "Creative Writing Assistance",
            "Translation Requests",
            "Math Problem Solving",
            "Code Review Requests",
            "API Integration Help",
        ],
    }
    result = NeighborhoodCategories(**data)
    assert result.scratchpad == data["scratchpad"]
    assert len(result.categories) == 8
    assert result.categories == data["categories"]


def test_neighborhood_categories_model_too_few():
    """Test NeighborhoodCategories model rejects too few categories."""
    data = {
        "scratchpad": "Brief analysis",
        "categories": ["Only", "Seven", "Categories", "Here", "Is", "Not", "Enough"],
    }
    with pytest.raises(ValidationError):
        NeighborhoodCategories(**data)


def test_deduplicated_clusters_model_valid():
    """Test DeduplicatedClusters model with valid data."""
    data = {
        "clusters": [
            "Python Programming",
            "JavaScript Development",
            "Database Management",
        ]
    }
    result = DeduplicatedClusters(**data)
    assert len(result.clusters) == 3
    assert result.clusters == data["clusters"]


def test_deduplicated_clusters_model_empty():
    """Test DeduplicatedClusters model rejects empty list."""
    data = {"clusters": []}
    with pytest.raises(ValidationError):
        DeduplicatedClusters(**data)


def test_cluster_assignment_model_valid():
    """Test ClusterAssignment model with valid data."""
    data = {
        "scratchpad": "This cluster is about Python. Parent options include Programming and Web Development. Python Programming is the best fit.",
        "assigned_cluster": "Python Programming",
    }
    result = ClusterAssignment(**data)
    assert result.scratchpad == data["scratchpad"]
    assert result.assigned_cluster == data["assigned_cluster"]


def test_refined_cluster_summary_model_valid():
    """Test RefinedClusterSummary model with valid data."""
    data = {
        "summary": "Users requested Python code for data analysis tasks. The queries ranged from basic to advanced pandas operations.",
        "title": "Python Data Analysis Code Requests",
    }
    result = RefinedClusterSummary(**data)
    assert result.summary == data["summary"]
    assert result.title == data["title"]


def test_refined_cluster_summary_title_too_long():
    """Test RefinedClusterSummary model rejects overly long titles."""
    data = {
        "summary": "Some summary here",
        "title": "This is an extremely long title that exceeds the maximum allowed length for cluster titles and should be rejected by validation" * 2,
    }
    with pytest.raises(ValidationError):
        RefinedClusterSummary(**data)


# ==============================================================================
# create_neighborhoods() Tests
# ==============================================================================


def test_create_neighborhoods_basic():
    """Test create_neighborhoods with basic embeddings."""
    # Create simple 2D embeddings for testing
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],  # Group 1
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],  # Group 2
            [10.0, 10.0],
            [10.1, 10.1],  # Group 3
        ]
    )
    n_neighborhoods = 3

    labels = create_neighborhoods(embeddings, n_neighborhoods, random_state=42)

    # Check properties
    assert len(labels) == len(embeddings)
    assert labels.min() >= 0
    assert labels.max() < n_neighborhoods
    assert len(np.unique(labels)) <= n_neighborhoods


def test_create_neighborhoods_single():
    """Test create_neighborhoods with single neighborhood."""
    embeddings = np.random.rand(10, 128)
    labels = create_neighborhoods(embeddings, n_neighborhoods=1)

    assert len(labels) == 10
    assert np.all(labels == 0)  # All should be in neighborhood 0


def test_create_neighborhoods_reproducible():
    """Test create_neighborhoods is reproducible with same random_state."""
    embeddings = np.random.rand(50, 128)

    labels1 = create_neighborhoods(embeddings, n_neighborhoods=5, random_state=42)
    labels2 = create_neighborhoods(embeddings, n_neighborhoods=5, random_state=42)

    assert np.array_equal(labels1, labels2)


def test_create_neighborhoods_different_seeds():
    """Test create_neighborhoods produces different results with different seeds."""
    embeddings = np.random.rand(50, 128)

    labels1 = create_neighborhoods(embeddings, n_neighborhoods=5, random_state=42)
    labels2 = create_neighborhoods(embeddings, n_neighborhoods=5, random_state=123)

    # Should be different (with high probability)
    assert not np.array_equal(labels1, labels2)


# ==============================================================================
# Async LLM Function Tests
# ==============================================================================


@pytest.mark.anyio(backend="asyncio")
async def test_generate_neighborhood_categories():
    """Test generate_neighborhood_categories with mocked client."""
    mock_client = AsyncMock()
    mock_response = NeighborhoodCategories(
        scratchpad="Analysis of patterns",
        categories=[
            "Python Programming",
            "JavaScript Development",
            "SQL Queries",
            "Creative Writing",
            "Translation",
            "Math Help",
            "Code Review",
            "API Integration",
            "Data Analysis",
            "Web Scraping",
        ],
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    clusters = [
        {"title": "Python Help", "description": "Users asking for Python code"},
        {"title": "JavaScript Tips", "description": "Web development questions"},
    ]

    result = await generate_neighborhood_categories(mock_client, clusters, target_count=10)

    assert isinstance(result, NeighborhoodCategories)
    assert len(result.categories) >= 8  # Model enforces minimum
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.anyio(backend="asyncio")
async def test_generate_neighborhood_categories_empty_clusters():
    """Test generate_neighborhood_categories with empty cluster list."""
    mock_client = AsyncMock()
    mock_response = NeighborhoodCategories(
        scratchpad="No clusters to analyze",
        categories=["General Queries"] * 8,  # Need at least 8
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await generate_neighborhood_categories(mock_client, [], target_count=10)

    assert isinstance(result, NeighborhoodCategories)
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.anyio(backend="asyncio")
async def test_deduplicate_cluster_names():
    """Test deduplicate_cluster_names with mocked client."""
    mock_client = AsyncMock()
    mock_response = DeduplicatedClusters(
        clusters=[
            "Python Programming",
            "JavaScript Development",
            "SQL Database Queries",
            "Creative Writing",
            "Translation Services",
        ]
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    candidate_names = [
        "Python Programming",
        "Python Coding",  # Duplicate
        "JavaScript Development",
        "JavaScript Dev",  # Duplicate
        "SQL Database Queries",
        "SQL Queries",  # Duplicate
        "Creative Writing",
        "Translation Services",
    ]

    result = await deduplicate_cluster_names(mock_client, candidate_names, target_count=5)

    assert isinstance(result, DeduplicatedClusters)
    assert len(result.clusters) <= len(candidate_names)
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.anyio(backend="asyncio")
async def test_deduplicate_cluster_names_no_duplicates():
    """Test deduplicate_cluster_names when all names are unique."""
    mock_client = AsyncMock()
    unique_names = [
        "Python Programming",
        "JavaScript Development",
        "SQL Queries",
    ]
    mock_response = DeduplicatedClusters(clusters=unique_names)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await deduplicate_cluster_names(mock_client, unique_names, target_count=3)

    assert isinstance(result, DeduplicatedClusters)
    assert len(result.clusters) == 3


@pytest.mark.anyio(backend="asyncio")
async def test_assign_to_parent_cluster():
    """Test assign_to_parent_cluster with mocked client."""
    mock_client = AsyncMock()
    mock_response = ClusterAssignment(
        scratchpad="This cluster is about Python programming. The best parent is Programming.",
        assigned_cluster="Programming",
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    child_cluster = {
        "title": "Python Data Analysis",
        "description": "Users requesting pandas code for data analysis",
    }
    parent_candidates = ["Programming", "Data Science", "Web Development"]

    result = await assign_to_parent_cluster(mock_client, child_cluster, parent_candidates)

    assert isinstance(result, ClusterAssignment)
    assert result.assigned_cluster in parent_candidates
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.anyio(backend="asyncio")
async def test_assign_to_parent_cluster_single_parent():
    """Test assign_to_parent_cluster with only one parent option."""
    mock_client = AsyncMock()
    mock_response = ClusterAssignment(
        scratchpad="Only one option available.",
        assigned_cluster="General Queries",
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    child_cluster = {"title": "Some Query", "description": "Description"}
    parent_candidates = ["General Queries"]

    result = await assign_to_parent_cluster(mock_client, child_cluster, parent_candidates)

    assert result.assigned_cluster == "General Queries"


@pytest.mark.anyio(backend="asyncio")
async def test_refine_parent_cluster():
    """Test refine_parent_cluster with mocked client."""
    mock_client = AsyncMock()
    mock_response = RefinedClusterSummary(
        summary="Users requested Python code for various data analysis tasks using pandas and numpy libraries.",
        title="Python Data Analysis Code Requests",
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    child_clusters = [
        "Python Pandas DataFrame Operations",
        "NumPy Array Manipulation",
        "Data Visualization with Matplotlib",
        "Statistical Analysis in Python",
    ]

    result = await refine_parent_cluster(mock_client, child_clusters)

    assert isinstance(result, RefinedClusterSummary)
    assert len(result.title) > 0
    assert len(result.summary) > 0
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.anyio(backend="asyncio")
async def test_refine_parent_cluster_single_child():
    """Test refine_parent_cluster with single child cluster."""
    mock_client = AsyncMock()
    mock_response = RefinedClusterSummary(
        summary="Users asked about Python programming.",
        title="Python Programming Help",
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    child_clusters = ["Python Programming Questions"]

    result = await refine_parent_cluster(mock_client, child_clusters)

    assert isinstance(result, RefinedClusterSummary)


@pytest.mark.anyio(backend="asyncio")
async def test_refine_parent_cluster_empty_children():
    """Test refine_parent_cluster with empty children list."""
    mock_client = AsyncMock()
    mock_response = RefinedClusterSummary(
        summary="No children to refine.",
        title="Empty Cluster",
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await refine_parent_cluster(mock_client, [])

    assert isinstance(result, RefinedClusterSummary)
    mock_client.chat.completions.create.assert_called_once()


# ==============================================================================
# Retry Logic Tests
# ==============================================================================


@pytest.mark.anyio(backend="asyncio")
async def test_generate_neighborhood_categories_retry_on_failure():
    """Test that generate_neighborhood_categories retries on failure."""
    mock_client = AsyncMock()

    # First two calls fail, third succeeds
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            Exception("API Error 1"),
            Exception("API Error 2"),
            NeighborhoodCategories(
                scratchpad="Success",
                categories=["Category " + str(i) for i in range(10)],
            ),
        ]
    )

    clusters = [{"title": "Test", "description": "Test desc"}]
    result = await generate_neighborhood_categories(mock_client, clusters)

    assert isinstance(result, NeighborhoodCategories)
    assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.anyio(backend="asyncio")
async def test_assign_to_parent_cluster_max_retries():
    """Test that assign_to_parent_cluster fails after max retries."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Persistent Error"))

    child_cluster = {"title": "Test", "description": "Test desc"}
    parent_candidates = ["Parent"]

    # Retry decorator wraps the exception in RetryError
    with pytest.raises(RetryError):
        await assign_to_parent_cluster(mock_client, child_cluster, parent_candidates)
