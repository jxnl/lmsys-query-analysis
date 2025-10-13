"""Integration tests for multi-source data loading."""

from pathlib import Path

import pytest
from sqlmodel import select

from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.loader import load_queries, load_queries_from_multiple
from lmsys_query_analysis.db.models import Query
from lmsys_query_analysis.db.sources import CSVSource


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    db = Database(db_path=str(db_path))
    db.create_tables()
    yield db


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


class TestMultiSourceLoading:
    """Test loading from multiple CSV sources."""
    
    def test_load_single_dataset(self, temp_db, fixtures_dir):
        """Test loading a single CSV dataset."""
        dataset1_csv = fixtures_dir / "dataset1.csv"
        source = CSVSource(file_path=str(dataset1_csv))
        
        stats = load_queries(temp_db, source=source, skip_existing=True)
        
        assert stats["loaded"] == 3
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert stats["source"] == f"csv:{dataset1_csv}"
        
        # Verify in database
        with temp_db.get_session() as session:
            queries = session.exec(select(Query)).all()
            assert len(queries) == 3
            conv_ids = {q.conversation_id for q in queries}
            assert conv_ids == {"a", "b", "c"}
    
    def test_load_same_dataset_twice(self, temp_db, fixtures_dir):
        """Test loading the same dataset twice skips duplicates."""
        dataset1_csv = fixtures_dir / "dataset1.csv"
        source = CSVSource(file_path=str(dataset1_csv))
        
        # First load
        stats1 = load_queries(temp_db, source=source, skip_existing=True)
        assert stats1["loaded"] == 3
        assert stats1["skipped"] == 0
        
        # Second load - should skip all
        source2 = CSVSource(file_path=str(dataset1_csv))
        stats2 = load_queries(temp_db, source=source2, skip_existing=True)
        assert stats2["loaded"] == 0
        assert stats2["skipped"] == 3
        
        # Verify still only 3 in database
        with temp_db.get_session() as session:
            count = session.exec(select(Query)).all()
            assert len(count) == 3
    
    def test_load_multiple_datasets_with_overlap(self, temp_db, fixtures_dir):
        """Test loading two datasets with overlapping conversation_ids."""
        dataset1_csv = fixtures_dir / "dataset1.csv"
        dataset2_csv = fixtures_dir / "dataset2.csv"
        
        source1 = CSVSource(file_path=str(dataset1_csv))
        source2 = CSVSource(file_path=str(dataset2_csv))
        
        # Load dataset1 first
        stats1 = load_queries(temp_db, source=source1, skip_existing=True)
        assert stats1["loaded"] == 3  # a, b, c
        assert stats1["skipped"] == 0
        
        # Load dataset2 - should skip 'c'
        stats2 = load_queries(temp_db, source=source2, skip_existing=True)
        assert stats2["loaded"] == 2  # d, e (c is skipped)
        assert stats2["skipped"] == 1  # c
        
        # Verify total count
        with temp_db.get_session() as session:
            queries = session.exec(select(Query)).all()
            assert len(queries) == 5
            conv_ids = {q.conversation_id for q in queries}
            assert conv_ids == {"a", "b", "c", "d", "e"}
            
            # Verify 'c' came from dataset1 (not dataset2)
            c_query = session.exec(
                select(Query).where(Query.conversation_id == "c")
            ).one()
            assert "dataset 1" in c_query.query_text.lower()
    
    def test_load_queries_from_multiple_function(self, temp_db, fixtures_dir):
        """Test the load_queries_from_multiple function."""
        dataset1_csv = fixtures_dir / "dataset1.csv"
        dataset2_csv = fixtures_dir / "dataset2.csv"
        
        source1 = CSVSource(file_path=str(dataset1_csv))
        source2 = CSVSource(file_path=str(dataset2_csv))
        
        # Load both datasets together
        results = load_queries_from_multiple(
            temp_db,
            sources=[source1, source2],
            skip_existing=True
        )
        
        assert len(results) == 2
        
        # Check first source stats
        assert results[0]["source"] == f"csv:{dataset1_csv}"
        assert results[0]["loaded"] == 3
        assert results[0]["skipped"] == 0
        
        # Check second source stats (should skip 'c')
        assert results[1]["source"] == f"csv:{dataset2_csv}"
        assert results[1]["loaded"] == 2
        assert results[1]["skipped"] == 1
        
        # Verify total count
        with temp_db.get_session() as session:
            queries = session.exec(select(Query)).all()
            assert len(queries) == 5
    
    def test_load_queries_from_multiple_empty_list_error(self, temp_db):
        """Test that empty sources list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            load_queries_from_multiple(temp_db, sources=[])
    
    def test_load_queries_from_multiple_preserves_order(self, temp_db, fixtures_dir):
        """Test that sources are processed in order."""
        dataset1_csv = fixtures_dir / "dataset1.csv"
        dataset2_csv = fixtures_dir / "dataset2.csv"
        
        # Load in reverse order
        source1 = CSVSource(file_path=str(dataset2_csv))
        source2 = CSVSource(file_path=str(dataset1_csv))
        
        results = load_queries_from_multiple(
            temp_db,
            sources=[source1, source2],
            skip_existing=True
        )
        
        # First source (dataset2) should load all 3
        assert results[0]["loaded"] == 3  # c, d, e
        assert results[0]["skipped"] == 0
        
        # Second source (dataset1) should skip 'c'
        assert results[1]["loaded"] == 2  # a, b (c is skipped)
        assert results[1]["skipped"] == 1
        
        # Verify 'c' came from dataset2 (first loaded)
        with temp_db.get_session() as session:
            c_query = session.exec(
                select(Query).where(Query.conversation_id == "c")
            ).one()
            assert "dataset 2" in c_query.query_text.lower()
    
    def test_load_multiple_sources_totals(self, temp_db, fixtures_dir):
        """Test aggregate statistics across multiple sources."""
        dataset1_csv = fixtures_dir / "dataset1.csv"
        dataset2_csv = fixtures_dir / "dataset2.csv"
        
        source1 = CSVSource(file_path=str(dataset1_csv))
        source2 = CSVSource(file_path=str(dataset2_csv))
        
        results = load_queries_from_multiple(
            temp_db,
            sources=[source1, source2],
            skip_existing=True
        )
        
        # Calculate totals
        total_loaded = sum(r["loaded"] for r in results)
        total_skipped = sum(r["skipped"] for r in results)
        total_processed = sum(r["total_processed"] for r in results)
        
        assert total_loaded == 5  # a, b, c, d, e
        assert total_skipped == 1  # c in dataset2
        assert total_processed == 6  # 3 + 3 records processed

