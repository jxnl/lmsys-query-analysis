"""ChromaDB utility commands."""

import typer
from rich.console import Console

from ..common import with_error_handling, chroma_path_option, json_output_option
from ..formatters import tables, json_output
from ...db.chroma import get_chroma

console = Console()
app = typer.Typer(help="ChromaDB utilities")


@app.command("info")
@with_error_handling
def chroma_info(
    chroma_path: str = chroma_path_option,
    json_out: bool = json_output_option,
):
    """List all Chroma collections with metadata and counts."""
    chroma = get_chroma(chroma_path)
    cols = chroma.list_all_collections()
    
    # Enrich known metadata keys with defaults
    normalized = []
    for c in cols:
        meta = c.get("metadata") or {}
        normalized.append(
            {
                "name": c.get("name"),
                "count": c.get("count"),
                "embedding_provider": meta.get("embedding_provider"),
                "embedding_model": meta.get("embedding_model"),
                "embedding_dimension": meta.get("embedding_dimension"),
                "description": meta.get("description"),
            }
        )
    
    if json_out:
        payload = json_output.format_chroma_collections_json(normalized)
        console.print_json(data=payload)
        return
    
    table = tables.format_chroma_collections_table(normalized)
    console.print(table)

