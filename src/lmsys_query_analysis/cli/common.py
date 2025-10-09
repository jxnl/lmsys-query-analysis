"""Shared CLI utilities and decorators."""

import logging
import functools
import typer
from typing import Callable
from rich.console import Console

console = Console()
logger = logging.getLogger("lmsys")

# Reusable option definitions
db_path_option = typer.Option(None, help="Database path (default: ~/.lmsys-query-analysis/queries.db)")
chroma_path_option = typer.Option(None, help="ChromaDB path (default: ~/.lmsys-query-analysis/chroma)")
embedding_model_option = typer.Option("openai/text-embedding-3-small", help="Embedding model (provider/model)")
json_output_option = typer.Option(False, "--json", help="Emit JSON output")

# Output format options
table_output_option = typer.Option(False, "--table", help="Force table output (default when no format specified)")
xml_output_option = typer.Option(False, "--xml", help="Emit XML output")


def with_error_handling(func: Callable) -> Callable:
    """Decorator to add consistent error handling to CLI commands."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"{func.__name__} failed: %s", e)
            
            # For ExceptionGroup, show all sub-exceptions
            if hasattr(e, 'exceptions'):
                console.print("[red]Multiple errors occurred:[/red]")
                for sub_exc in e.exceptions:
                    console.print(f"[yellow]{type(sub_exc).__name__}: {sub_exc}[/yellow]")
            else:
                console.print(f"[red]Error: {e}[/red]")
            
            raise typer.Exit(1)
    return wrapper

