"""Unit tests for database connection management."""

import pytest
import tempfile
from pathlib import Path
from sqlmodel import select
from sqlalchemy import text

from lmsys_query_analysis.db.connection import Database, get_db, DEFAULT_DB_PATH
from lmsys_query_analysis.db.models import Query


def test_database_initialization_with_memory():
    """Test Database initialization with in-memory database."""
    db = Database(":memory:", auto_create_tables=False)
    
    assert db.db_path == Path(":memory:")
    assert db.engine is not None


def test_database_initialization_with_path(tmp_path):
    """Test Database initialization with file path."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    
    assert db.db_path == db_path
    assert db.engine is not None
    assert db_path.exists()


def test_database_initialization_creates_directory(tmp_path):
    """Test that Database creates parent directories."""
    db_path = tmp_path / "subdir" / "nested" / "test.db"
    db = Database(db_path)
    
    assert db_path.parent.exists()
    assert db_path.parent.is_dir()


def test_database_initialization_with_none_uses_default():
    """Test that None db_path uses DEFAULT_DB_PATH."""
    db = Database(None, auto_create_tables=False)
    
    assert db.db_path == DEFAULT_DB_PATH


def test_database_auto_create_tables_true(tmp_path):
    """Test that auto_create_tables=True creates tables."""
    db_path = tmp_path / "test.db"
    db = Database(db_path, auto_create_tables=True)
    
    # Should be able to use Query model without error
    with db.get_session() as session:
        query = Query(
            conversation_id="test",
            model="gpt-4",
            query_text="test query",
        )
        session.add(query)
        session.commit()
        
        # Verify it was added
        result = session.exec(select(Query)).first()
        assert result is not None
        assert result.query_text == "test query"


def test_database_auto_create_tables_false(tmp_path):
    """Test that auto_create_tables=False doesn't create tables."""
    db_path = tmp_path / "test.db"
    db = Database(db_path, auto_create_tables=False)
    
    # Tables should not exist yet
    # Trying to query will raise an error
    with db.get_session() as session:
        with pytest.raises(Exception):  # SQLite operational error
            session.exec(select(Query)).first()


def test_database_create_tables(tmp_path):
    """Test manual table creation."""
    db_path = tmp_path / "test.db"
    db = Database(db_path, auto_create_tables=False)
    
    # Manually create tables
    db.create_tables()
    
    # Now should work
    with db.get_session() as session:
        query = Query(
            conversation_id="test",
            model="gpt-4",
            query_text="test query",
        )
        session.add(query)
        session.commit()


def test_database_get_session(tmp_path):
    """Test getting database sessions."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    
    session = db.get_session()
    assert session is not None
    
    # Session should be usable
    query = Query(
        conversation_id="test",
        model="gpt-4",
        query_text="test query",
    )
    session.add(query)
    session.commit()
    session.close()


def test_database_get_session_context_manager(tmp_path):
    """Test using session as context manager."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    
    with db.get_session() as session:
        query = Query(
            conversation_id="test",
            model="gpt-4",
            query_text="test query",
        )
        session.add(query)
        session.commit()
    
    # Session should be closed after context
    # Verify data persists
    with db.get_session() as session:
        result = session.exec(select(Query)).first()
        assert result.query_text == "test query"


def test_database_drop_tables(tmp_path):
    """Test dropping all tables."""
    db_path = tmp_path / "test.db"
    db = Database(db_path, auto_create_tables=True)
    
    # Add some data
    with db.get_session() as session:
        query = Query(
            conversation_id="test",
            model="gpt-4",
            query_text="test query",
        )
        session.add(query)
        session.commit()
    
    # Drop tables
    db.drop_tables()
    
    # Tables should not exist
    with db.get_session() as session:
        with pytest.raises(Exception):  # SQLite operational error
            session.exec(select(Query)).first()


def test_database_foreign_keys_enabled(tmp_path):
    """Test that foreign keys are enabled in SQLite."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    
    # Query the PRAGMA to verify foreign keys are enabled
    with db.get_session() as session:
        result = session.exec(text("PRAGMA foreign_keys")).first()
        assert result == (1,)  # 1 means enabled


def test_get_db_function(tmp_path):
    """Test get_db convenience function."""
    db_path = tmp_path / "test.db"
    db = get_db(db_path)
    
    assert isinstance(db, Database)
    assert db.db_path == db_path


def test_get_db_function_with_defaults():
    """Test get_db with default arguments."""
    db = get_db(None, auto_create_tables=False)
    
    assert isinstance(db, Database)
    assert db.db_path == DEFAULT_DB_PATH


def test_database_multiple_sessions(tmp_path):
    """Test that multiple sessions can be created."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    
    session1 = db.get_session()
    session2 = db.get_session()
    
    assert session1 is not session2
    
    # Both should be usable
    query1 = Query(
        conversation_id="test1",
        model="gpt-4",
        query_text="query 1",
    )
    query2 = Query(
        conversation_id="test2",
        model="gpt-4",
        query_text="query 2",
    )
    
    session1.add(query1)
    session1.commit()
    
    session2.add(query2)
    session2.commit()
    
    session1.close()
    session2.close()
    
    # Verify both were saved
    with db.get_session() as session:
        count = len(session.exec(select(Query)).all())
        assert count == 2


def test_database_persistence(tmp_path):
    """Test that data persists across Database instances."""
    db_path = tmp_path / "test.db"
    
    # Create first instance and add data
    db1 = Database(db_path)
    with db1.get_session() as session:
        query = Query(
            conversation_id="test",
            model="gpt-4",
            query_text="persistent query",
        )
        session.add(query)
        session.commit()
    
    # Create second instance and verify data
    db2 = Database(db_path)
    with db2.get_session() as session:
        result = session.exec(select(Query)).first()
        assert result is not None
        assert result.query_text == "persistent query"

