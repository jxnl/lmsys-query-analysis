"""Unit tests for ChromaDB manager with mocking."""

from unittest.mock import Mock, patch

from lmsys_query_analysis.db.chroma import (
    ChromaManager,
    sanitize_collection_name,
)


def test_sanitize_collection_name_basic():
    """Test basic collection name sanitization."""
    assert sanitize_collection_name("my_collection") == "my_collection"
    assert sanitize_collection_name("my-collection") == "my-collection"
    assert sanitize_collection_name("MyCollection123") == "MyCollection123"


def test_sanitize_collection_name_with_dots():
    """Test sanitization of names with dots."""
    assert sanitize_collection_name("text.embedding.3") == "text_embedding_3"
    assert sanitize_collection_name("gpt.4.turbo") == "gpt_4_turbo"


def test_sanitize_collection_name_with_slashes():
    """Test sanitization of names with slashes."""
    assert sanitize_collection_name("openai/gpt-4") == "openai_gpt-4"
    assert sanitize_collection_name("company/model") == "company_model"


def test_sanitize_collection_name_removes_consecutive_underscores():
    """Test that consecutive underscores are collapsed."""
    assert sanitize_collection_name("my___collection") == "my_collection"
    assert sanitize_collection_name("a__b__c") == "a_b_c"


def test_sanitize_collection_name_strips_special_chars():
    """Test that leading/trailing special chars are stripped."""
    assert sanitize_collection_name("_collection_") == "collection"
    assert sanitize_collection_name("-name-") == "name"
    assert sanitize_collection_name("__test__") == "test"


def test_sanitize_collection_name_truncates_long_names():
    """Test truncation of names longer than 63 characters."""
    long_name = "a" * 100
    result = sanitize_collection_name(long_name)
    assert len(result) <= 63
    assert result == "a" * 63


def test_sanitize_collection_name_ensures_minimum_length():
    """Test that short names are padded to minimum length."""
    assert len(sanitize_collection_name("a")) == 9
    assert len(sanitize_collection_name("ab")) == 10
    assert sanitize_collection_name("a") == "a_default"


def test_sanitize_collection_name_special_characters():
    """Test handling of various special characters."""
    assert sanitize_collection_name("name@2024") == "name_2024"
    assert sanitize_collection_name("test$model") == "test_model"
    assert sanitize_collection_name("v1.0!beta") == "v1_0_beta"


def test_sanitize_collection_name_complex():
    """Test complex real-world collection names."""
    assert sanitize_collection_name("text-embedding-3-small") == "text-embedding-3-small"

    result = sanitize_collection_name("cohere/embed-english-v3.0")
    assert result == "cohere_embed-english-v3_0"

    result = sanitize_collection_name("sentence-transformers/all-MiniLM-L6-v2")
    assert result == "sentence-transformers_all-MiniLM-L6-v2"


def test_sanitize_collection_name_preserves_hyphens():
    """Test that hyphens are preserved in collection names."""
    assert sanitize_collection_name("my-test-collection") == "my-test-collection"
    assert sanitize_collection_name("gpt-4-turbo") == "gpt-4-turbo"


def test_sanitize_collection_name_mixed_cases():
    """Test various edge cases together."""
    result = sanitize_collection_name("__My.Model/v2.0__")
    assert result == "My_Model_v2_0"

    result = sanitize_collection_name("a.")
    assert len(result) >= 3


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_initialization_mocked(mock_client_class):
    """Test ChromaManager initialization with mocked client."""
    mock_client = Mock()
    mock_collection = Mock()
    mock_collection.name = "queries_test_model"
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test",
        embedding_model="test-model",
        embedding_provider="test",
        embedding_dimension=256,
    )

    assert manager.embedding_model == "test-model"
    assert manager.embedding_provider == "test"
    assert manager.embedding_dimension == 256

    mock_client_class.assert_called_once()

    assert mock_client.get_or_create_collection.call_count == 2


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_count_queries_mocked(mock_client_class):
    """Test counting queries with mocked collection."""
    mock_client = Mock()
    mock_collection = Mock()
    mock_collection.count.return_value = 42
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    count = manager.count_queries()

    assert count == 42
    manager.queries_collection.count.assert_called_once()


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_list_all_collections_mocked(mock_client_class):
    """Test listing all collections with mocked client."""
    mock_client = Mock()
    mock_collection1 = Mock()
    mock_collection1.name = "queries_test"
    mock_collection1.count.return_value = 100
    mock_collection1.metadata = {"embedding_model": "test"}

    mock_collection2 = Mock()
    mock_collection2.name = "summaries_test"
    mock_collection2.count.return_value = 50
    mock_collection2.metadata = {"embedding_model": "test"}

    mock_client.get_or_create_collection.side_effect = [mock_collection1, mock_collection2]
    mock_client.list_collections.return_value = [mock_collection1, mock_collection2]
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    collections = manager.list_all_collections()

    assert len(collections) == 2
    assert collections[0]["name"] == "queries_test"
    assert collections[0]["count"] == 100
    assert collections[1]["name"] == "summaries_test"


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_different_providers_create_different_collections(mock_client_class):
    """Test that different providers result in different collection names."""
    mock_client = Mock()
    mock_collection = Mock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    ChromaManager(
        persist_directory="/tmp/test", embedding_model="model-a", embedding_provider="provider-a"
    )

    ChromaManager(
        persist_directory="/tmp/test", embedding_model="model-b", embedding_provider="provider-b"
    )

    calls = mock_client.get_or_create_collection.call_args_list

    names = [call.kwargs.get("name") or call.args[0] for call in calls]

    assert len(names) == 4

    assert "provider-a" in names[0] or "provider_a" in names[0]
    assert "provider-a" in names[1] or "provider_a" in names[1]

    assert "provider-b" in names[2] or "provider_b" in names[2]
    assert "provider-b" in names[3] or "provider_b" in names[3]


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_cohere_includes_dimension(mock_client_class):
    """Test that Cohere collections include dimension in name."""
    mock_client = Mock()
    mock_collection = Mock()
    mock_collection.name = "queries_cohere_model_512"
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    ChromaManager(
        persist_directory="/tmp/test",
        embedding_model="embed-v4.0",
        embedding_provider="cohere",
        embedding_dimension=512,
    )

    calls = mock_client.get_or_create_collection.call_args_list

    names = [call.kwargs.get("name", "") for call in calls]
    assert any("512" in name for name in names)


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_get_query_embeddings_map_empty(mock_client_class):
    """Test getting embeddings map when collection is empty."""
    mock_client = Mock()
    mock_collection = Mock()
    mock_collection.get.return_value = {"ids": [], "embeddings": []}
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    embeddings_map = manager.get_query_embeddings_map([1, 2, 3])

    assert embeddings_map == {}


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_add_queries_batch(mock_client_class):
    """Test adding queries in batch."""
    import numpy as np

    mock_client = Mock()
    mock_collection = Mock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    query_ids = [1, 2, 3]
    texts = ["Query 1", "Query 2", "Query 3"]
    embeddings = np.array([[0.1] * 10, [0.2] * 10, [0.3] * 10])
    metadata = [{"model": "gpt-4"}, {"model": "claude"}, {"model": "gpt-4"}]

    manager.add_queries_batch(query_ids, texts, embeddings, metadata)

    mock_collection.add.assert_called_once()
    call_args = mock_collection.add.call_args
    assert "ids" in call_args.kwargs
    assert len(call_args.kwargs["ids"]) == 3
    assert call_args.kwargs["ids"][0] == "query_1"


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_add_cluster_summary(mock_client_class):
    """Test adding a single cluster summary."""
    import numpy as np

    mock_client = Mock()
    mock_queries_collection = Mock()
    mock_summaries_collection = Mock()

    def get_collection(name, **kwargs):
        if "summaries" in name:
            return mock_summaries_collection
        return mock_queries_collection

    mock_client.get_or_create_collection.side_effect = get_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    embedding = np.array([0.1] * 10)
    manager.add_cluster_summary(
        run_id="test-run",
        cluster_id=5,
        summary="Test summary",
        embedding=embedding,
        metadata={"num_queries": 10},
    )

    mock_summaries_collection.add.assert_called_once()
    call_args = mock_summaries_collection.add.call_args
    assert call_args.kwargs["ids"] == ["cluster_test-run_5"]


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_add_cluster_summary_with_title(mock_client_class):
    """Test adding cluster summary with title and description."""
    import numpy as np

    mock_client = Mock()
    mock_queries_collection = Mock()
    mock_summaries_collection = Mock()

    def get_collection(name, **kwargs):
        if "summaries" in name:
            return mock_summaries_collection
        return mock_queries_collection

    mock_client.get_or_create_collection.side_effect = get_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    embedding = np.array([0.1] * 10)
    manager.add_cluster_summary(
        run_id="test-run",
        cluster_id=5,
        summary="Test summary",
        embedding=embedding,
        metadata={"num_queries": 10},
        title="Test Title",
        description="Test Description",
    )

    call_args = mock_summaries_collection.add.call_args
    assert "Test Title" in call_args.kwargs["documents"][0]
    assert "Test Description" in call_args.kwargs["documents"][0]


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_add_cluster_summaries_batch(mock_client_class):
    """Test adding multiple cluster summaries."""
    import numpy as np

    mock_client = Mock()
    mock_queries_collection = Mock()
    mock_summaries_collection = Mock()

    def get_collection(name, **kwargs):
        if "summaries" in name:
            return mock_summaries_collection
        return mock_queries_collection

    mock_client.get_or_create_collection.side_effect = get_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    cluster_ids = [1, 2, 3]
    summaries = ["Summary 1", "Summary 2", "Summary 3"]
    embeddings = np.array([[0.1] * 10, [0.2] * 10, [0.3] * 10])
    metadata_list = [{"num_queries": 5}, {"num_queries": 10}, {"num_queries": 15}]

    manager.add_cluster_summaries_batch(
        run_id="test-run",
        cluster_ids=cluster_ids,
        summaries=summaries,
        embeddings=embeddings,
        metadata_list=metadata_list,
    )

    mock_summaries_collection.upsert.assert_called_once()
    call_args = mock_summaries_collection.upsert.call_args
    assert len(call_args.kwargs["ids"]) == 3


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_search_queries(mock_client_class):
    """Test searching queries."""

    mock_client = Mock()
    mock_collection = Mock()
    mock_collection.query.return_value = {
        "ids": [["query_1", "query_2"]],
        "documents": [["Doc 1", "Doc 2"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[{"model": "gpt-4"}, {"model": "claude"}]],
    }
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    results = manager.search_queries("test query", n_results=10)

    mock_collection.query.assert_called_once()
    assert results["ids"] == [["query_1", "query_2"]]


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_search_cluster_summaries(mock_client_class):
    """Test searching cluster summaries."""
    mock_client = Mock()
    mock_queries_collection = Mock()
    mock_summaries_collection = Mock()
    mock_summaries_collection.query.return_value = {
        "ids": [["cluster_run_1"]],
        "documents": [["Summary"]],
        "distances": [[0.1]],
        "metadatas": [[{"run_id": "test", "cluster_id": 1}]],
    }

    def get_collection(name, **kwargs):
        if "summaries" in name:
            return mock_summaries_collection
        return mock_queries_collection

    mock_client.get_or_create_collection.side_effect = get_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    manager.search_cluster_summaries("test query", run_id="test", n_results=5)

    mock_summaries_collection.query.assert_called_once()
    call_args = mock_summaries_collection.query.call_args
    assert "where" in call_args.kwargs or "where" in str(call_args)


@patch("lmsys_query_analysis.db.chroma.chromadb.PersistentClient")
def test_chroma_manager_count_summaries(mock_client_class):
    """Test counting summaries in a run."""
    mock_client = Mock()
    mock_queries_collection = Mock()
    mock_summaries_collection = Mock()
    mock_summaries_collection.get.return_value = {
        "ids": ["id1", "id2", "id3"],
        "documents": ["Doc1", "Doc2", "Doc3"],
    }

    def get_collection(name, **kwargs):
        if "summaries" in name:
            return mock_summaries_collection
        return mock_queries_collection

    mock_client.get_or_create_collection.side_effect = get_collection
    mock_client_class.return_value = mock_client

    manager = ChromaManager(
        persist_directory="/tmp/test", embedding_model="test-model", embedding_provider="test"
    )

    count = manager.count_summaries("test-run")

    assert count == 3
