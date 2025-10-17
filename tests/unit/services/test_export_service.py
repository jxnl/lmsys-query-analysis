"""Unit tests for export_service."""

import csv
import json

from lmsys_query_analysis.services import export_service


def test_get_export_data(populated_db):
    """Test getting export data for a run."""
    data = export_service.get_export_data(populated_db, "test-run-001")

    assert len(data) == 5
    query, qc, summary = data[0]
    assert query.id is not None
    assert qc.run_id == "test-run-001"
    assert qc.cluster_id in [0, 1]


def test_get_export_data_empty(temp_db):
    """Test getting export data when none exists."""
    data = export_service.get_export_data(temp_db, "nonexistent-run")

    assert len(data) == 0


def test_export_to_csv(populated_db, temp_dir):
    """Test exporting data to CSV format."""
    output_path = temp_dir / "export.csv"
    data = export_service.get_export_data(populated_db, "test-run-001")

    count = export_service.export_to_csv(str(output_path), data)

    assert count == 5
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        assert len(rows) == 5
        assert "query_id" in rows[0]
        assert "cluster_id" in rows[0]
        assert "query_text" in rows[0]
        assert "cluster_title" in rows[0]


def test_export_to_json(populated_db, temp_dir):
    """Test exporting data to JSON format."""
    output_path = temp_dir / "export.json"
    data = export_service.get_export_data(populated_db, "test-run-001")

    count = export_service.export_to_json(str(output_path), data)

    assert count == 5
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        json_data = json.load(f)

        assert len(json_data) == 5
        assert "query_id" in json_data[0]
        assert "cluster_id" in json_data[0]
        assert "query_text" in json_data[0]
        assert "cluster_title" in json_data[0]


def test_export_to_csv_empty(temp_db, temp_dir):
    """Test exporting empty dataset to CSV."""
    output_path = temp_dir / "empty.csv"
    data = export_service.get_export_data(temp_db, "nonexistent-run")

    count = export_service.export_to_csv(str(output_path), data)

    assert count == 0
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 1


def test_export_to_json_preserves_unicode(populated_db, temp_dir):
    """Test that JSON export preserves unicode characters."""
    output_path = temp_dir / "export.json"
    data = export_service.get_export_data(populated_db, "test-run-001")

    export_service.export_to_json(str(output_path), data)

    with open(output_path, encoding="utf-8") as f:
        content = f.read()
        json_data = json.loads(content)

        assert "\\u" not in content or len(json_data) > 0
