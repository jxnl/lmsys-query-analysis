"""Tests for API dependencies."""

import os
from unittest.mock import patch

import pytest

from lmsys_query_analysis.api.dependencies import (
    create_chroma_manager,
    get_chroma_path,
    get_db,
)
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun


def test_get_db():
    """Test get_db dependency returns a Database instance."""
    db_gen = get_db()
    db = next(db_gen)

    assert isinstance(db, Database)
    assert db.db_path is not None

    # Cleanup
    try:
        next(db_gen)
    except StopIteration:
        pass


def test_get_db_with_custom_path():
    """Test get_db uses DB_PATH environment variable."""
    custom_path = ":memory:"

    with patch.dict(os.environ, {"DB_PATH": custom_path}):
        # Need to reload the module to pick up env var
        from importlib import reload

        from lmsys_query_analysis.api import dependencies

        reload(dependencies)

        db_gen = dependencies.get_db()
        db = next(db_gen)

        assert str(db.db_path) == custom_path

        # Cleanup
        try:
            next(db_gen)
        except StopIteration:
            pass


def test_get_chroma_path_default():
    """Test get_chroma_path returns default path."""
    path = get_chroma_path()

    assert isinstance(path, str)
    assert "chroma" in path.lower()


def test_get_chroma_path_from_env():
    """Test get_chroma_path uses CHROMA_PATH environment variable."""
    custom_path = "/custom/chroma/path"

    with patch.dict(os.environ, {"CHROMA_PATH": custom_path}):
        from importlib import reload

        from lmsys_query_analysis.api import dependencies

        reload(dependencies)

        path = dependencies.get_chroma_path()
        assert path == custom_path


def test_create_chroma_manager_basic(tmp_path):
    """Test create_chroma_manager with basic run parameters."""
    db = Database(db_path=":memory:", auto_create_tables=True)

    # Create a test run
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="test-run-1",
            algorithm="kmeans",
            parameters={
                "embedding_model": "text-embedding-3-small",
                "embedding_provider": "openai",
                "embedding_dimension": 1536,
            },
            num_clusters=10,
        )
        session.add(run)
        session.commit()

    # Create ChromaManager
    chroma_path = str(tmp_path / "chroma")
    chroma = create_chroma_manager("test-run-1", db, chroma_path)

    assert chroma is not None
    assert chroma.embedding_model == "text-embedding-3-small"
    assert chroma.embedding_provider == "openai"
    assert chroma.embedding_dimension == 1536


def test_create_chroma_manager_run_not_found():
    """Test create_chroma_manager raises error for non-existent run."""
    db = Database(db_path=":memory:", auto_create_tables=True)

    with pytest.raises(ValueError, match="Run not found: nonexistent-run"):
        create_chroma_manager("nonexistent-run", db)


def test_create_chroma_manager_defaults(tmp_path):
    """Test create_chroma_manager with missing parameters uses defaults."""
    db = Database(db_path=":memory:", auto_create_tables=True)

    # Create a test run with minimal parameters
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="test-run-2",
            algorithm="kmeans",
            parameters={},  # Empty parameters
            num_clusters=10,
        )
        session.add(run)
        session.commit()

    # Create ChromaManager
    chroma_path = str(tmp_path / "chroma")
    chroma = create_chroma_manager("test-run-2", db, chroma_path)

    # Should use defaults
    assert chroma.embedding_model == "text-embedding-3-small"
    assert chroma.embedding_provider == "openai"


def test_create_chroma_manager_cohere_defaults(tmp_path):
    """Test create_chroma_manager applies Cohere-specific defaults."""
    db = Database(db_path=":memory:", auto_create_tables=True)

    # Create a test run with Cohere provider but no dimension
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="test-run-3",
            algorithm="kmeans",
            parameters={
                "embedding_model": "embed-english-v3.0",
                "embedding_provider": "cohere",
                # No embedding_dimension specified
            },
            num_clusters=10,
        )
        session.add(run)
        session.commit()

    # Create ChromaManager
    chroma_path = str(tmp_path / "chroma")
    chroma = create_chroma_manager("test-run-3", db, chroma_path)

    # Should default to 256 for Cohere
    assert chroma.embedding_provider == "cohere"
    assert chroma.embedding_dimension == 256


def test_create_chroma_manager_cohere_with_dimension(tmp_path):
    """Test create_chroma_manager respects explicit Cohere dimension."""
    db = Database(db_path=":memory:", auto_create_tables=True)

    # Create a test run with Cohere provider and explicit dimension
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="test-run-4",
            algorithm="kmeans",
            parameters={
                "embedding_model": "embed-english-v3.0",
                "embedding_provider": "cohere",
                "embedding_dimension": 1024,
            },
            num_clusters=10,
        )
        session.add(run)
        session.commit()

    # Create ChromaManager
    chroma_path = str(tmp_path / "chroma")
    chroma = create_chroma_manager("test-run-4", db, chroma_path)

    # Should use explicit dimension
    assert chroma.embedding_provider == "cohere"
    assert chroma.embedding_dimension == 1024


def test_create_chroma_manager_uses_default_chroma_path(tmp_path):
    """Test create_chroma_manager uses CHROMA_PATH when path not provided."""
    # Set a valid temp path for CHROMA_PATH
    chroma_path = str(tmp_path / "chroma")
    with patch.dict(os.environ, {"CHROMA_PATH": chroma_path}):
        # Reload module to pick up new env var
        from importlib import reload

        from lmsys_query_analysis.api import dependencies

        reload(dependencies)

        db = Database(db_path=":memory:", auto_create_tables=True)

        # Create a test run
        with db.get_session() as session:
            run = ClusteringRun(
                run_id="test-run-5",
                algorithm="kmeans",
                parameters={
                    "embedding_model": "text-embedding-3-small",
                    "embedding_provider": "openai",
                },
                num_clusters=10,
            )
            session.add(run)
            session.commit()

        # Create ChromaManager without chroma_path parameter
        chroma = dependencies.create_chroma_manager("test-run-5", db)

        # Should create ChromaManager (path will be from CHROMA_PATH env var)
        assert chroma is not None


def test_create_chroma_manager_sentence_transformers(tmp_path):
    """Test create_chroma_manager with sentence-transformers provider."""
    db = Database(db_path=":memory:", auto_create_tables=True)

    # Create a test run with sentence-transformers
    with db.get_session() as session:
        run = ClusteringRun(
            run_id="test-run-6",
            algorithm="kmeans",
            parameters={
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_provider": "sentence-transformers",
            },
            num_clusters=10,
        )
        session.add(run)
        session.commit()

    # Create ChromaManager
    chroma_path = str(tmp_path / "chroma")
    chroma = create_chroma_manager("test-run-6", db, chroma_path)

    assert chroma.embedding_model == "all-MiniLM-L6-v2"
    assert chroma.embedding_provider == "sentence-transformers"

