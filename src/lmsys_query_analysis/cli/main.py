"""Main CLI entry point for LMSYS query analysis."""

import typer
from dotenv import load_dotenv

from ..utils.logging import setup_logging
from .commands import (
    analysis,
    chroma,
    clustering,
    data,
    edit,
    hierarchy,
    search,
    summarization,
    verify,
)

load_dotenv()

app = typer.Typer(help="LMSYS Query Analysis CLI")


@app.callback()
def _configure(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging (DEBUG)."),
):
    """Configure logging before executing a subcommand."""
    setup_logging(verbose=verbose)


app.add_typer(clustering.app, name="cluster")
app.add_typer(chroma.app, name="chroma")
app.add_typer(verify.app, name="verify")
app.add_typer(edit.app, name="edit")

app.command()(data.load)
app.command()(data.clear)
app.command("backfill-chroma")(data.backfill_chroma)

app.command("list")(analysis.list_queries)
app.command()(analysis.runs)
app.command("list-clusters")(analysis.list_clusters)
app.command()(analysis.inspect)
app.command()(analysis.export)

app.command()(search.search)
app.command("search-cluster")(search.search_cluster)

app.command()(summarization.summarize)

app.command("merge-clusters")(hierarchy.merge_clusters_cmd)
app.command("show-hierarchy")(hierarchy.show_hierarchy_cmd)


if __name__ == "__main__":
    app()
