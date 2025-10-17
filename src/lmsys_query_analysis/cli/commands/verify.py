"""Verification and consistency check commands."""

import typer
from rich.console import Console
from sqlmodel import select

from ...db.chroma import get_chroma
from ...db.connection import get_db
from ...db.models import ClusteringRun, ClusterSummary
from ..common import chroma_path_option, db_path_option, json_output_option, with_error_handling
from ..formatters import tables

console = Console()
app = typer.Typer(help="Verification and consistency checks")


@app.command("sync")
@with_error_handling
def verify_sync(
    run_id: str = typer.Argument(..., help="Clustering run ID to verify"),
    chroma_path: str = chroma_path_option,
    db_path: str = db_path_option,
    json_out: bool = json_output_option,
):
    """Verify Chroma summaries for a run match SQLite and vector space config."""
    db = get_db(db_path)

    with db.get_session() as s:
        run = s.exec(select(ClusteringRun).where(ClusteringRun.run_id == run_id)).first()
        if not run:
            console.print(f"[red]Run not found: {run_id}[/red]")
            raise typer.Exit(1)

        params = run.parameters or {}
        provider = params.get("embedding_provider", "openai")
        model = params.get("embedding_model", "text-embedding-3-small")
        dim = params.get("embedding_dimension")
        if provider == "cohere" and dim is None:
            dim = 256

        sqlite_summary_count = len(
            s.exec(select(ClusterSummary).where(ClusterSummary.run_id == run_id)).all()
        )

    chroma = get_chroma(chroma_path, model, provider, dim)
    chroma_summary_count = chroma.count_summaries(run_id=run_id)
    known_runs = chroma.list_runs_in_summaries()

    issues = []
    if run_id not in known_runs:
        issues.append(f"run_id {run_id} not present in Chroma summaries collection")
    if sqlite_summary_count != chroma_summary_count:
        issues.append(
            f"summary count mismatch (sqlite={sqlite_summary_count}, chroma={chroma_summary_count})"
        )

    status = "ok" if not issues else "mismatch"
    report = {
        "run_id": run_id,
        "space": {
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_dimension": dim,
        },
        "sqlite": {"summary_count": sqlite_summary_count},
        "chroma": {
            "summary_count": chroma_summary_count,
            "runs_in_summaries": known_runs,
            "summaries_collection": chroma.summaries_collection.name,
        },
        "status": status,
        "issues": issues,
    }

    if json_out:
        console.print_json(data=report)
        return

    table = tables.format_verify_sync_table(report)
    console.print(table)
