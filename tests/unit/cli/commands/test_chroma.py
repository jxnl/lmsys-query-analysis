"""Unit tests for chroma CLI commands."""

from unittest.mock import Mock, patch


@patch("lmsys_query_analysis.cli.commands.chroma.get_chroma")
def test_chroma_info_command(mock_get_chroma):
    """Test chroma-info command."""
    from lmsys_query_analysis.cli.commands.chroma import chroma_info

    # Setup mock
    mock_chroma = Mock()
    mock_chroma.count_queries.return_value = 1000
    mock_chroma.count_summaries.return_value = 50
    mock_chroma.list_all_collections.return_value = [
        {"name": "queries_test", "count": 1000},
        {"name": "summaries_test", "count": 50},
    ]
    mock_get_chroma.return_value = mock_chroma

    # Execute command
    chroma_info(chroma_path="/tmp/chroma", json_out=False, table=False, xml=False)

    # Verify
    mock_get_chroma.assert_called_once()
    mock_chroma.list_all_collections.assert_called_once()
