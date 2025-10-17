"""Pytest fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine
from sqlmodel.pool import StaticPool

from lmsys_query_analysis.api.app import app
from lmsys_query_analysis.api.dependencies import get_db
from lmsys_query_analysis.db.connection import Database


@pytest.fixture(name="db")
def db_fixture():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    db = Database(db_path=":memory:", auto_create_tables=False)
    db.engine = engine
    yield db


@pytest.fixture(name="client")
def client_fixture(db: Database):
    """Create a FastAPI test client with test database."""

    def get_db_override():
        yield db

    app.dependency_overrides[get_db] = get_db_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
