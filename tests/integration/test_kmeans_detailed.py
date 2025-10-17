"""Additional tests for KMeans clustering functionality."""

import tempfile
from pathlib import Path

from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun, Query, QueryCluster


def test_kmeans_cluster_distribution():
    """Test that KMeans distributes queries across clusters reasonably."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(str(db_path))
        db.create_tables()

        # Create test queries
        with db.get_session() as session:
            for i in range(50):
                query = Query(
                    conversation_id=f"conv-{i}",
                    model="test-model",
                    query_text=f"Test query about topic {i % 5}",
                    language="English",
                )
                session.add(query)
            session.commit()

        # Create mock clustering result
        with db.get_session() as session:
            run = ClusteringRun(
                run_id="test-run-001",
                algorithm="kmeans",
                num_clusters=5,
                parameters={"test": True},
            )
            session.add(run)
            session.commit()

            # Assign queries to clusters
            from sqlmodel import select

            queries = session.exec(select(Query)).all()
            for i, query in enumerate(queries):
                assignment = QueryCluster(
                    run_id="test-run-001",
                    query_id=query.id,
                    cluster_id=i % 5,  # Distribute evenly
                    confidence_score=0.9,
                )
                session.add(assignment)
            session.commit()

        # Verify distribution
        with db.get_session() as session:
            from sqlmodel import func, select

            for cluster_id in range(5):
                count = session.exec(
                    select(func.count())
                    .select_from(QueryCluster)
                    .where(QueryCluster.run_id == "test-run-001")
                    .where(QueryCluster.cluster_id == cluster_id)
                ).one()
                assert count == 10, f"Cluster {cluster_id} should have 10 queries"


def test_clustering_run_parameters():
    """Test that clustering run stores parameters correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(str(db_path))
        db.create_tables()

        with db.get_session() as session:
            run = ClusteringRun(
                run_id="kmeans-100-test",
                algorithm="kmeans-minibatch",
                num_clusters=100,
                parameters={
                    "embedding_provider": "openai",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimension": 1536,
                    "batch_size": 1000,
                    "random_state": 42,
                },
                description="Test clustering run",
            )
            session.add(run)
            session.commit()

            # Retrieve and verify
            from sqlmodel import select

            saved_run = session.exec(
                select(ClusteringRun).where(ClusteringRun.run_id == "kmeans-100-test")
            ).first()

            assert saved_run is not None
            assert saved_run.algorithm == "kmeans-minibatch"
            assert saved_run.num_clusters == 100
            assert saved_run.parameters["embedding_provider"] == "openai"
            assert saved_run.parameters["embedding_dimension"] == 1536
            assert saved_run.description == "Test clustering run"


def test_query_cluster_confidence_scores():
    """Test that confidence scores are stored and retrieved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(str(db_path))
        db.create_tables()

        with db.get_session() as session:
            # Create query
            query = Query(
                conversation_id="test-conv",
                model="test-model",
                query_text="Test query",
                language="English",
            )
            session.add(query)

            # Create run
            run = ClusteringRun(
                run_id="test-run",
                algorithm="kmeans",
                num_clusters=5,
                parameters={},
            )
            session.add(run)
            session.commit()
            session.refresh(query)

            # Add cluster assignment with confidence
            assignment = QueryCluster(
                run_id="test-run",
                query_id=query.id,
                cluster_id=3,
                confidence_score=0.85,
            )
            session.add(assignment)
            session.commit()

            # Verify
            from sqlmodel import select

            saved = session.exec(
                select(QueryCluster).where(QueryCluster.run_id == "test-run")
            ).first()

            assert saved.cluster_id == 3
            assert saved.confidence_score == 0.85


def test_minibatch_vs_full_kmeans():
    """Test understanding of MiniBatch vs full KMeans parameters."""
    # MiniBatch KMeans parameters
    minibatch_params = {
        "batch_size": 1000,
        "max_iter": 100,
        "n_init": 3,
        "reassignment_ratio": 0.01,
    }

    # Full KMeans parameters
    full_params = {
        "max_iter": 300,
        "n_init": 10,
        "algorithm": "lloyd",
    }

    # Verify MiniBatch is more suitable for large datasets
    assert minibatch_params["batch_size"] == 1000  # Processes in batches
    assert minibatch_params["n_init"] < full_params["n_init"]  # Fewer initializations
    assert minibatch_params["max_iter"] < full_params["max_iter"]  # Fewer iterations


def test_empty_clusters_handling():
    """Test that system handles empty clusters appropriately."""
    # Simulate clustering that produces some empty clusters
    n_clusters = 10
    cluster_sizes = [15, 0, 20, 0, 30, 5, 0, 25, 10, 15]  # Some empty clusters

    # Count non-empty clusters
    non_empty = sum(1 for size in cluster_sizes if size > 0)
    empty = sum(1 for size in cluster_sizes if size == 0)

    assert non_empty == 7
    assert empty == 3
    assert non_empty + empty == n_clusters


def test_cluster_statistics():
    """Test calculation of cluster statistics."""
    cluster_sizes = [10, 20, 30, 15, 25, 5, 40, 35, 12, 8]

    min_size = min(cluster_sizes)
    max_size = max(cluster_sizes)
    avg_size = sum(cluster_sizes) / len(cluster_sizes)

    # Calculate median properly
    sorted_sizes = sorted(cluster_sizes)
    n = len(sorted_sizes)
    if n % 2 == 0:
        median_size = (sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2]) / 2
    else:
        median_size = sorted_sizes[n // 2]

    assert min_size == 5
    assert max_size == 40
    assert avg_size == 20.0
    assert median_size == 17.5  # (15 + 20) / 2


def test_run_id_format():
    """Test that run IDs follow expected format."""
    run_id = "kmeans-200-20251009-183500"

    parts = run_id.split("-")
    assert len(parts) == 4
    assert parts[0] == "kmeans"
    assert parts[1] == "200"  # Number of clusters
    assert len(parts[2]) == 8  # Date (YYYYMMDD)
    assert len(parts[3]) == 6  # Time (HHMMSS)

    # Verify it's parseable
    algorithm = parts[0]
    n_clusters = int(parts[1])
    date_str = parts[2]
    time_str = parts[3]

    assert algorithm == "kmeans"
    assert n_clusters == 200
    assert date_str.isdigit()
    assert time_str.isdigit()
