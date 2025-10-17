"""Smoke tests for semantic search with real ChromaDB and embeddings."""

import tempfile
from pathlib import Path

import pytest
from sqlmodel import SQLModel, create_engine

from lmsys_query_analysis.clustering.embeddings import EmbeddingGenerator
from lmsys_query_analysis.db.chroma import ChromaManager
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import Query
from lmsys_query_analysis.semantic import ClustersClient, QueriesClient


@pytest.mark.smoke
def test_query_search_with_chroma():
    """Test semantic query search with real ChromaDB and embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)

        db = Database(str(db_path))
        db.engine = engine

        test_queries = [
            ("q1", "What is machine learning?"),
            ("q2", "Explain neural networks"),
            ("q3", "How to write Python functions"),
            ("q4", "Python async programming"),
        ]

        with db.get_session() as session:
            for conv_id, text in test_queries:
                session.add(
                    Query(
                        conversation_id=conv_id,
                        query_text=text,
                        model="gpt-4",
                        language="en",
                    )
                )
            session.commit()

        chroma_path = Path(tmpdir) / "chroma"
        chroma = ChromaManager(
            persist_directory=str(chroma_path),
            embedding_model="text-embedding-3-small",
            embedding_provider="openai",
        )

        embedder = EmbeddingGenerator(
            model_name="text-embedding-3-small",
            provider="openai",
        )

        with db.get_session() as session:
            from sqlmodel import select

            queries = session.exec(select(Query)).all()

            ids = [q.id for q in queries]
            texts = [q.query_text for q in queries]
            embeddings = embedder.generate_embeddings(texts, batch_size=4, show_progress=False)
            metadata = [{"model": q.model, "language": q.language} for q in queries]

            chroma.add_queries_batch(ids, texts, embeddings, metadata)

        client = QueriesClient(db, chroma, embedder)

        hits = client.find(
            text="artificial intelligence and deep learning",
            n_results=2,
        )

        assert len(hits) > 0
        assert len(hits) <= 2
        assert any(word in hits[0].snippet.lower() for word in ["machine", "learning", "neural"])


@pytest.mark.smoke
def test_cluster_search_with_chroma():
    """Test cluster summary search with real ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)

        db = Database(str(db_path))
        db.engine = engine

        chroma_path = Path(tmpdir) / "chroma"
        chroma = ChromaManager(
            persist_directory=str(chroma_path),
            embedding_model="text-embedding-3-small",
            embedding_provider="openai",
        )

        embedder = EmbeddingGenerator(
            model_name="text-embedding-3-small",
            provider="openai",
        )

        from lmsys_query_analysis.db.models import ClusteringRun, ClusterSummary

        with db.get_session() as session:
            run = ClusteringRun(
                run_id="test-run-smoke",
                algorithm="kmeans",
                num_clusters=2,
            )
            session.add(run)

            summaries = [
                ClusterSummary(
                    run_id="test-run-smoke",
                    cluster_id=0,
                    summary_run_id="summary-smoke",
                    title="Machine Learning Questions",
                    description="Questions about machine learning, AI, and neural networks",
                    summary="Machine Learning Questions\n\nQuestions about ML and AI",
                    num_queries=100,
                ),
                ClusterSummary(
                    run_id="test-run-smoke",
                    cluster_id=1,
                    summary_run_id="summary-smoke",
                    title="Python Programming",
                    description="Questions about Python syntax, functions, and async programming",
                    summary="Python Programming\n\nQuestions about Python",
                    num_queries=80,
                ),
            ]

            for s in summaries:
                session.add(s)
            session.commit()

        texts = [
            "Machine Learning Questions\n\nQuestions about machine learning, AI, and neural networks",
            "Python Programming\n\nQuestions about Python syntax, functions, and async programming",
        ]
        embeddings = embedder.generate_embeddings(texts, batch_size=2, show_progress=False)

        chroma.add_cluster_summaries_batch(
            run_id="test-run-smoke",
            cluster_ids=[0, 1],
            summaries=texts,
            embeddings=embeddings,
            metadata_list=[
                {"num_queries": 100},
                {"num_queries": 80},
            ],
            titles=["Machine Learning Questions", "Python Programming"],
            descriptions=[
                "Questions about machine learning, AI, and neural networks",
                "Questions about Python syntax, functions, and async programming",
            ],
        )

        client = ClustersClient(db, chroma, embedder)

        hits = client.find(
            text="deep learning and AI models",
            run_id="test-run-smoke",
            top_k=2,
        )

        assert len(hits) > 0
        assert "machine" in hits[0].title.lower() or "learning" in hits[0].title.lower()
