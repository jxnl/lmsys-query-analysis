"""Tests for database models."""

from datetime import datetime

import pytest
from sqlmodel import Session, create_engine, select

from lmsys_query_analysis.db.models import (
    ClusteringRun,
    ClusterSummary,
    Query,
    QueryCluster,
)


@pytest.fixture
def engine():
    """Create in-memory SQLite engine for testing."""
    from sqlmodel import SQLModel

    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a database session for testing."""
    with Session(engine) as session:
        yield session


def test_create_query(session):
    """Test creating a Query record."""
    query = Query(
        conversation_id="test-123",
        model="gpt-4",
        query_text="What is machine learning?",
        language="en",
        extra_metadata={"test": True},
    )
    session.add(query)
    session.commit()
    session.refresh(query)

    assert query.id is not None
    assert query.conversation_id == "test-123"
    assert query.model == "gpt-4"
    assert query.query_text == "What is machine learning?"
    assert query.language == "en"
    assert query.extra_metadata["test"] is True
    assert isinstance(query.created_at, datetime)


def test_unique_conversation_id(session):
    """Test that conversation_id is unique."""
    query1 = Query(
        conversation_id="test-123",
        model="gpt-4",
        query_text="First query",
    )
    session.add(query1)
    session.commit()

    query2 = Query(
        conversation_id="test-123",
        model="gpt-3.5",
        query_text="Second query",
    )
    session.add(query2)

    with pytest.raises(Exception):  # Should raise integrity error
        session.commit()


def test_create_clustering_run(session):
    """Test creating a ClusteringRun record."""
    run = ClusteringRun(
        run_id="run-001",
        algorithm="kmeans",
        parameters={"n_clusters": 10, "random_state": 42},
        description="Test clustering run",
        num_clusters=10,
    )
    session.add(run)
    session.commit()

    assert run.run_id == "run-001"
    assert run.algorithm == "kmeans"
    assert run.parameters["n_clusters"] == 10
    assert run.num_clusters == 10


def test_query_cluster_relationship(session):
    """Test relationships between Query, ClusteringRun, and QueryCluster."""
    # Create query
    query = Query(
        conversation_id="test-456",
        model="gpt-4",
        query_text="Test query",
    )
    session.add(query)
    session.commit()
    session.refresh(query)

    # Create clustering run
    run = ClusteringRun(run_id="run-002", algorithm="kmeans", num_clusters=5)
    session.add(run)
    session.commit()

    # Create cluster assignment
    cluster = QueryCluster(
        run_id=run.run_id, query_id=query.id, cluster_id=2, confidence_score=0.95
    )
    session.add(cluster)
    session.commit()
    session.refresh(cluster)

    # Test relationships
    assert cluster.query.conversation_id == "test-456"
    assert cluster.run.algorithm == "kmeans"
    assert cluster.cluster_id == 2
    assert cluster.confidence_score == 0.95


def test_cluster_summary(session):
    """Test creating a ClusterSummary."""
    run = ClusteringRun(run_id="run-003", algorithm="hdbscan", num_clusters=8)
    session.add(run)
    session.commit()

    summary = ClusterSummary(
        run_id=run.run_id,
        cluster_id=1,
        summary="This cluster contains queries about Python programming",
        num_queries=150,
        representative_queries=[1, 5, 23, 45],
    )
    session.add(summary)
    session.commit()
    session.refresh(summary)

    assert summary.cluster_id == 1
    assert "Python" in summary.summary
    assert summary.num_queries == 150
    assert len(summary.representative_queries) == 4
    assert summary.run.run_id == "run-003"


def test_query_filtering(session):
    """Test filtering queries by model and language."""
    queries = [
        Query(conversation_id="q1", model="gpt-4", query_text="Query 1", language="en"),
        Query(conversation_id="q2", model="gpt-3.5", query_text="Query 2", language="en"),
        Query(conversation_id="q3", model="gpt-4", query_text="Query 3", language="es"),
    ]
    for q in queries:
        session.add(q)
    session.commit()

    # Filter by model
    statement = select(Query).where(Query.model == "gpt-4")
    gpt4_queries = session.exec(statement).all()
    assert len(gpt4_queries) == 2

    # Filter by language
    statement = select(Query).where(Query.language == "en")
    en_queries = session.exec(statement).all()
    assert len(en_queries) == 2
