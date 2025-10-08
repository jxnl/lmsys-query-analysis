"""Unit tests for run_service."""

import pytest
from lmsys_query_analysis.services import run_service
from lmsys_query_analysis.db.models import ClusteringRun


def test_list_runs_all(populated_db, db_session):
    """Test listing all clustering runs."""
    # Add another run
    run2 = ClusteringRun(
        run_id="test-run-002",
        algorithm="hdbscan",
        num_clusters=5,
        description="Second test run",
    )
    db_session.add(run2)
    db_session.commit()
    
    runs = run_service.list_runs(populated_db, latest=False)
    
    assert len(runs) >= 2
    assert any(r.run_id == "test-run-001" for r in runs)
    assert any(r.run_id == "test-run-002" for r in runs)


def test_list_runs_latest_only(populated_db, db_session):
    """Test listing only the latest run."""
    # Add another run
    run2 = ClusteringRun(
        run_id="test-run-002",
        algorithm="hdbscan",
        num_clusters=5,
        description="Second test run",
    )
    db_session.add(run2)
    db_session.commit()
    
    runs = run_service.list_runs(populated_db, latest=True)
    
    assert len(runs) == 1


def test_list_runs_empty(temp_db):
    """Test listing runs when none exist."""
    runs = run_service.list_runs(temp_db, latest=False)
    
    assert len(runs) == 0


def test_get_run_exists(populated_db):
    """Test getting a specific run that exists."""
    run = run_service.get_run(populated_db, "test-run-001")
    
    assert run is not None
    assert run.run_id == "test-run-001"
    assert run.algorithm == "kmeans"
    assert run.num_clusters == 3


def test_get_run_not_exists(populated_db):
    """Test getting a run that doesn't exist."""
    run = run_service.get_run(populated_db, "nonexistent-run")
    
    assert run is None

