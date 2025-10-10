"""Tests for CLI common utilities."""

import pytest
from unittest.mock import Mock, patch
import typer

from lmsys_query_analysis.cli.common import (
    with_error_handling,
    db_path_option,
    chroma_path_option,
    embedding_model_option,
    json_output_option,
)


def test_with_error_handling_success():
    """Test that decorator allows successful function execution."""
    @with_error_handling
    def successful_func():
        return "success"
    
    result = successful_func()
    assert result == "success"


def test_with_error_handling_basic_exception():
    """Test that decorator handles basic exceptions."""
    @with_error_handling
    def failing_func():
        raise ValueError("test error")
    
    with pytest.raises(typer.Exit) as exc_info:
        failing_func()
    
    assert exc_info.value.exit_code == 1


def test_with_error_handling_with_args_kwargs():
    """Test that decorator preserves function arguments."""
    @with_error_handling
    def func_with_args(a, b, keyword=None):
        return f"{a}-{b}-{keyword}"
    
    result = func_with_args("x", "y", keyword="z")
    assert result == "x-y-z"


def test_with_error_handling_exception_group():
    """Test that decorator handles ExceptionGroup specially."""
    # Create a mock ExceptionGroup-like exception
    class MockExceptionGroup(Exception):
        def __init__(self, msg, exceptions):
            super().__init__(msg)
            self.exceptions = exceptions
    
    @with_error_handling
    def func_with_exception_group():
        raise MockExceptionGroup(
            "Multiple errors",
            [ValueError("error 1"), TypeError("error 2")]
        )
    
    with patch("lmsys_query_analysis.cli.common.console") as mock_console:
        with pytest.raises(typer.Exit) as exc_info:
            func_with_exception_group()
        
        # Should have printed the "Multiple errors occurred" message
        assert any(
            "Multiple errors occurred" in str(call)
            for call in mock_console.print.call_args_list
        )
        
        # Should exit with code 1
        assert exc_info.value.exit_code == 1


def test_with_error_handling_logs_exception():
    """Test that decorator logs exceptions."""
    @with_error_handling
    def failing_func():
        raise RuntimeError("test runtime error")
    
    with patch("lmsys_query_analysis.cli.common.logger") as mock_logger:
        with pytest.raises(typer.Exit):
            failing_func()
        
        # Should have called logger.exception
        mock_logger.exception.assert_called_once()
        call_args = mock_logger.exception.call_args
        assert "failing_func failed" in str(call_args)


def test_with_error_handling_prints_error_message():
    """Test that decorator prints error message to console."""
    @with_error_handling
    def failing_func():
        raise ValueError("specific error message")
    
    with patch("lmsys_query_analysis.cli.common.console") as mock_console:
        with pytest.raises(typer.Exit):
            failing_func()
        
        # Should have printed error message
        printed_messages = [str(call) for call in mock_console.print.call_args_list]
        assert any("specific error message" in msg for msg in printed_messages)


def test_with_error_handling_preserves_function_metadata():
    """Test that decorator preserves function name and docstring."""
    @with_error_handling
    def documented_func():
        """This is a test function."""
        return "result"
    
    assert documented_func.__name__ == "documented_func"
    assert documented_func.__doc__ == "This is a test function."


def test_typer_option_defaults():
    """Test that Typer options have expected defaults."""
    # These are option factories, not actual values
    assert db_path_option is not None
    assert chroma_path_option is not None
    assert embedding_model_option is not None
    assert json_output_option is not None


def test_with_error_handling_nested_exceptions():
    """Test handling of nested exceptions."""
    @with_error_handling
    def func_with_nested_exception():
        try:
            raise ValueError("inner error")
        except ValueError as e:
            raise RuntimeError("outer error") from e
    
    with pytest.raises(typer.Exit) as exc_info:
        func_with_nested_exception()
    
    assert exc_info.value.exit_code == 1


def test_with_error_handling_system_exit():
    """Test that SystemExit is handled."""
    @with_error_handling
    def func_with_system_exit():
        raise SystemExit(2)
    
    with pytest.raises((SystemExit, typer.Exit)):
        func_with_system_exit()


def test_with_error_handling_multiple_calls():
    """Test that decorator can be used multiple times."""
    @with_error_handling
    def func1():
        return "func1"
    
    @with_error_handling
    def func2():
        return "func2"
    
    assert func1() == "func1"
    assert func2() == "func2"


def test_with_error_handling_return_values():
    """Test that various return values are preserved."""
    @with_error_handling
    def func_returning_none():
        return None
    
    @with_error_handling
    def func_returning_dict():
        return {"key": "value"}
    
    @with_error_handling
    def func_returning_list():
        return [1, 2, 3]
    
    assert func_returning_none() is None
    assert func_returning_dict() == {"key": "value"}
    assert func_returning_list() == [1, 2, 3]

