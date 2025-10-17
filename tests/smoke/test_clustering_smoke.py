"""Smoke tests for clustering with real data and embeddings."""

import tempfile
from pathlib import Path

import pytest
from sqlmodel import SQLModel, create_engine

from lmsys_query_analysis.clustering.kmeans import run_kmeans_clustering
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import Query


@pytest.mark.smoke
def test_kmeans_clustering_end_to_end():
    """Test full KMeans clustering workflow with real embeddings."""
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)

        db = Database(str(db_path))
        db.engine = engine

        # Add sample queries
        test_queries = [
            Query(
                conversation_id=f"conv{i}",
                query_text=text,
                model="gpt-4",
                language="en",
            )
            for i, text in enumerate(
                [
                    "What is machine learning?",
                    "Explain neural networks",
                    "How does deep learning work?",
                    "What is supervised learning?",
                    "How do I write a Python function?",
                    "Explain Python decorators",
                    "What are Python generators?",
                    "How to use async in Python?",
                ]
            )
        ]

        with db.get_session() as session:
            for q in test_queries:
                session.add(q)
            session.commit()

        # Run clustering
        run_id = run_kmeans_clustering(
            db=db,
            n_clusters=2,  # Should group ML and Python questions
            description="Smoke test clustering",
            embedding_model="text-embedding-3-small",
            embedding_provider="openai",
            embed_batch_size=4,
            chunk_size=100,
            mb_batch_size=8,
            chroma=None,
        )

        assert run_id is not None
        assert "kmeans" in run_id.lower()

        # Verify clustering was created
        from sqlmodel import select

        from lmsys_query_analysis.db.models import ClusteringRun, QueryCluster

        with db.get_session() as session:
            # Check run exists
            run = session.exec(select(ClusteringRun).where(ClusteringRun.run_id == run_id)).first()
            assert run is not None
            assert run.num_clusters == 2

            # Check query assignments exist
            assignments = session.exec(
                select(QueryCluster).where(QueryCluster.run_id == run_id)
            ).all()
            assert len(assignments) == 8  # All 8 queries should be assigned

            # Check that we have 2 distinct clusters
            cluster_ids = {a.cluster_id for a in assignments}
            assert len(cluster_ids) == 2
