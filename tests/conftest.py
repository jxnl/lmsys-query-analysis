"""Shared test fixtures for pytest."""

import tempfile
from pathlib import Path

import pytest
from sqlmodel import SQLModel, create_engine

from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import (
    ClusteringRun,
    ClusterSummary,
    Query,
    QueryCluster,
)


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure pytest-anyio to only use asyncio backend."""
    return "asyncio"


@pytest.fixture
def temp_db():
    """Create a temporary in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    # Create a Database instance with the test engine
    db = Database(":memory:")
    db.engine = engine

    yield db

    # Cleanup
    SQLModel.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(temp_db):
    """Provide a database session for tests."""
    with temp_db.get_session() as session:
        yield session


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        Query(
            conversation_id="conv1",
            query_text="How do I write a Python function?",
            model="gpt-4",
            language="en",
        ),
        Query(
            conversation_id="conv2",
            query_text="Explain async/await in Python",
            model="gpt-4",
            language="en",
        ),
        Query(
            conversation_id="conv3",
            query_text="What is machine learning?",
            model="claude-3",
            language="en",
        ),
        Query(
            conversation_id="conv4",
            query_text="How to use pandas?",
            model="gpt-3.5-turbo",
            language="en",
        ),
        Query(
            conversation_id="conv5",
            query_text="Explain neural networks",
            model="gpt-4",
            language="en",
        ),
    ]


@pytest.fixture
def sample_clustering_run():
    """Sample clustering run for testing."""
    return ClusteringRun(
        run_id="test-run-001",
        algorithm="kmeans",
        num_clusters=3,
        description="Test clustering run",
        parameters={
            "n_clusters": 3,
            "embedding_model": "text-embedding-3-small",
            "embedding_provider": "openai",
        },
    )


@pytest.fixture
def sample_query_clusters():
    """Sample query-cluster assignments for testing.

    Note: query_ids will be set in populated_db after queries are committed.
    """
    return [
        {"cluster_id": 0},  # First query
        {"cluster_id": 0},  # Second query
        {"cluster_id": 1},  # Third query
        {"cluster_id": 0},  # Fourth query
        {"cluster_id": 1},  # Fifth query
    ]


@pytest.fixture
def sample_cluster_summaries(sample_clustering_run):
    """Sample cluster summaries for testing."""
    run_id = sample_clustering_run.run_id
    return [
        ClusterSummary(
            run_id=run_id,
            cluster_id=0,
            summary_run_id="summary-001",
            alias="test-v1",
            title="Python Programming Questions",
            description="Questions about Python programming, functions, and async",
            summary="Python Programming Questions\n\nQuestions about Python programming, functions, and async",
            num_queries=3,
            representative_queries=[
                "How do I write a Python function?",
                "Explain async/await in Python",
            ],
            model="openai/gpt-4o-mini",
            parameters={"max_queries": 100},
        ),
        ClusterSummary(
            run_id=run_id,
            cluster_id=1,
            summary_run_id="summary-001",
            alias="test-v1",
            title="Machine Learning Concepts",
            description="Questions about machine learning and neural networks",
            summary="Machine Learning Concepts\n\nQuestions about machine learning and neural networks",
            num_queries=2,
            representative_queries=[
                "What is machine learning?",
                "Explain neural networks",
            ],
            model="openai/gpt-4o-mini",
            parameters={"max_queries": 100},
        ),
    ]


@pytest.fixture
def populated_db(
    temp_db,
    db_session,
    sample_queries,
    sample_clustering_run,
    sample_query_clusters,
    sample_cluster_summaries,
):
    """Database populated with sample data for integration tests."""
    # Add queries first and commit to get IDs
    for query in sample_queries:
        db_session.add(query)

    # Add clustering run
    db_session.add(sample_clustering_run)
    db_session.commit()

    # Refresh to get auto-generated IDs
    for query in sample_queries:
        db_session.refresh(query)

    # Now create query-cluster assignments with real query IDs
    run_id = sample_clustering_run.run_id
    for i, qc_data in enumerate(sample_query_clusters):
        qc = QueryCluster(
            run_id=run_id,
            query_id=sample_queries[i].id,
            cluster_id=qc_data["cluster_id"],
        )
        db_session.add(qc)

    # Add cluster summaries
    for summary in sample_cluster_summaries:
        db_session.add(summary)

    db_session.commit()

    return temp_db


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB client for testing."""

    class MockChromaManager:
        def __init__(self):
            self.queries = {}
            self.summaries = {}
            self.persist_directory = "/tmp/test-chroma"

        def add_queries_batch(self, query_ids, texts, embeddings, metadata):
            for qid, text in zip(query_ids, texts, strict=False):
                self.queries[qid] = text

        def add_cluster_summaries_batch(
            self, run_id, cluster_ids, summaries, embeddings, metadata_list, titles, descriptions
        ):
            for cid, summary in zip(cluster_ids, summaries, strict=False):
                self.summaries[(run_id, cid)] = summary

        def count_queries(self):
            return len(self.queries)

        def count_summaries(self, run_id=None):
            if run_id:
                return len([k for k in self.summaries if k[0] == run_id])
            return len(self.summaries)

        def list_runs_in_summaries(self):
            return list(set(k[0] for k in self.summaries))

        def get_query_embeddings_map(self, ids):
            return {qid: True for qid in ids if qid in self.queries}

        def list_all_collections(self):
            return [
                {
                    "name": "queries",
                    "count": len(self.queries),
                    "metadata": {
                        "embedding_provider": "openai",
                        "embedding_model": "text-embedding-3-small",
                        "embedding_dimension": 1536,
                    },
                }
            ]

    return MockChromaManager()
