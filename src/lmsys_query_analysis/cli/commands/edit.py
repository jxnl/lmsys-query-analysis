"""Cluster curation and editing commands under 'lmsys edit' namespace."""

import typer
from rich.console import Console
from rich.table import Table

from ...db.connection import get_db
from ...services import curation_service
from ..common import db_path_option, with_error_handling

app = typer.Typer(help="Edit and curate clusters")
console = Console()




@app.command()
@with_error_handling
def view_query(
    query_id: int = typer.Argument(..., help="Query ID to view"),
    db_path: str = db_path_option,
):
    """View query details with all cluster assignments."""
    db = get_db(db_path)
    details = curation_service.get_query_details(db, query_id)

    if not details:
        console.print(f"[red]Query {query_id} not found[/red]")
        raise typer.Exit(1)

    query = details["query"]
    clusters = details["clusters"]

    console.print(f"\n[bold cyan]Query {query_id}[/bold cyan]")
    console.print(f"[yellow]Text:[/yellow] {query.query_text}")
    console.print(f"[yellow]Model:[/yellow] {query.model}")
    console.print(f"[yellow]Language:[/yellow] {query.language or 'unknown'}")
    console.print(f"[yellow]Conversation:[/yellow] {query.conversation_id}")

    if clusters:
        console.print(f"\n[bold yellow]Cluster Assignments ({len(clusters)}):[/bold yellow]")
        table = Table(show_header=True)
        table.add_column("Run ID", style="cyan")
        table.add_column("Cluster ID", style="green")
        table.add_column("Title", style="white")
        table.add_column("Confidence", style="magenta")

        for c in clusters:
            table.add_row(
                c["run_id"],
                str(c["cluster_id"]),
                c["title"] or "N/A",
                f"{c['confidence_score']:.3f}" if c["confidence_score"] else "N/A",
            )

        console.print(table)
    else:
        console.print("\n[yellow]No cluster assignments found[/yellow]")


@app.command()
@with_error_handling
def move_query(
    run_id: str = typer.Argument(..., help="Run ID"),
    query_id: int = typer.Option(..., "--query-id", help="Query ID to move"),
    to_cluster: int = typer.Option(..., "--to-cluster", help="Target cluster ID"),
    reason: str | None = typer.Option(None, "--reason", help="Reason for move"),
    db_path: str = db_path_option,
):
    """Move a query from one cluster to another."""
    db = get_db(db_path)
    result = curation_service.move_query(db, run_id, query_id, to_cluster, reason=reason)

    console.print(
        f"[green]✓[/green] Moved query {query_id} from cluster {result['from_cluster_id']} → {result['to_cluster_id']}"
    )
    if reason:
        console.print(f"  [dim]Reason: {reason}[/dim]")


@app.command()
@with_error_handling
def move_queries(
    run_id: str = typer.Argument(..., help="Run ID"),
    query_ids: str = typer.Option(..., "--query-ids", help="Comma-separated query IDs"),
    to_cluster: int = typer.Option(..., "--to-cluster", help="Target cluster ID"),
    reason: str | None = typer.Option(None, "--reason", help="Reason for move"),
    db_path: str = db_path_option,
):
    """Move multiple queries to a cluster."""
    db = get_db(db_path)
    ids = [int(x.strip()) for x in query_ids.split(",")]
    result = curation_service.move_queries_batch(db, run_id, ids, to_cluster, reason=reason)

    console.print(f"[green]✓[/green] Moved {result['moved']} queries to cluster {to_cluster}")
    if result["failed"] > 0:
        console.print(f"[red]✗[/red] Failed to move {result['failed']} queries")
        for error in result["errors"]:
            console.print(f"  [red]Query {error['query_id']}: {error['error']}[/red]")




@app.command()
@with_error_handling
def rename_cluster(
    run_id: str = typer.Argument(..., help="Run ID"),
    cluster_id: int = typer.Option(..., "--cluster-id", help="Cluster ID to rename"),
    title: str | None = typer.Option(None, "--title", help="New title"),
    description: str | None = typer.Option(None, "--description", help="New description"),
    db_path: str = db_path_option,
):
    """Rename a cluster (update title and/or description)."""
    if not title and not description:
        console.print("[red]Error: Must provide --title and/or --description[/red]")
        raise typer.Exit(1)

    db = get_db(db_path)
    result = curation_service.rename_cluster(
        db, run_id, cluster_id, title=title, description=description
    )

    console.print(f"[green]✓[/green] Renamed cluster {cluster_id}")
    if title:
        console.print(f"  [yellow]Old title:[/yellow] {result['old_title']}")
        console.print(f"  [yellow]New title:[/yellow] {result['new_title']}")
    if description:
        console.print("  [yellow]Description updated[/yellow]")


@app.command()
@with_error_handling
def merge_clusters(
    run_id: str = typer.Argument(..., help="Run ID"),
    source: str = typer.Option(..., "--source", help="Comma-separated source cluster IDs"),
    target: int = typer.Option(..., "--target", help="Target cluster ID"),
    new_title: str | None = typer.Option(None, "--new-title", help="New title for merged cluster"),
    new_description: str | None = typer.Option(None, "--new-description", help="New description"),
    db_path: str = db_path_option,
):
    """Merge multiple clusters into a target cluster."""
    db = get_db(db_path)
    source_ids = [int(x.strip()) for x in source.split(",")]
    result = curation_service.merge_clusters(
        db, run_id, source_ids, target, new_title=new_title, new_description=new_description
    )

    console.print(f"[green]✓[/green] Merged clusters {source_ids} → cluster {target}")
    console.print(f"  [yellow]Queries moved:[/yellow] {result['queries_moved']}")
    console.print(f"  [yellow]New title:[/yellow] {result['new_title']}")


@app.command()
@with_error_handling
def split_cluster(
    run_id: str = typer.Argument(..., help="Run ID"),
    cluster_id: int = typer.Option(..., "--cluster-id", help="Original cluster ID"),
    query_ids: str = typer.Option(..., "--query-ids", help="Comma-separated query IDs to split"),
    new_title: str = typer.Option(..., "--new-title", help="Title for new cluster"),
    new_description: str = typer.Option(
        ..., "--new-description", help="Description for new cluster"
    ),
    db_path: str = db_path_option,
):
    """Split queries from a cluster into a new cluster."""
    db = get_db(db_path)
    ids = [int(x.strip()) for x in query_ids.split(",")]
    result = curation_service.split_cluster(db, run_id, cluster_id, ids, new_title, new_description)

    console.print(
        f"[green]✓[/green] Split cluster {cluster_id} → created cluster {result['new_cluster_id']}"
    )
    console.print(f"  [yellow]Original:[/yellow] Cluster {cluster_id}")
    console.print(
        f"  [yellow]New:[/yellow] Cluster {result['new_cluster_id']} - {result['new_title']}"
    )
    console.print(f"  [yellow]Queries moved:[/yellow] {result['queries_moved']}")


@app.command()
@with_error_handling
def delete_cluster(
    run_id: str = typer.Argument(..., help="Run ID"),
    cluster_id: int = typer.Option(..., "--cluster-id", help="Cluster ID to delete"),
    orphan: bool = typer.Option(False, "--orphan", help="Orphan queries instead of reassigning"),
    move_to: int | None = typer.Option(None, "--move-to", help="Move queries to this cluster"),
    reason: str | None = typer.Option(None, "--reason", help="Reason for deletion"),
    db_path: str = db_path_option,
):
    """Delete a cluster, orphaning or reassigning its queries."""
    if not orphan and move_to is None:
        console.print("[red]Error: Must specify either --orphan or --move-to[/red]")
        raise typer.Exit(1)

    db = get_db(db_path)
    result = curation_service.delete_cluster(
        db, run_id, cluster_id, move_to_cluster_id=move_to, orphan=orphan, reason=reason
    )

    console.print(f"[green]✓[/green] Deleted cluster {cluster_id}")
    console.print(f"  [yellow]Queries:[/yellow] {result['query_count']}")
    if orphan:
        console.print("  [yellow]Status:[/yellow] Orphaned")
    else:
        console.print(f"  [yellow]Moved to:[/yellow] Cluster {result['moved_to']}")
    if reason:
        console.print(f"  [yellow]Reason:[/yellow] {reason}")




@app.command()
@with_error_handling
def tag_cluster(
    run_id: str = typer.Argument(..., help="Run ID"),
    cluster_id: int = typer.Option(..., "--cluster-id", help="Cluster ID to tag"),
    coherence: int | None = typer.Option(None, "--coherence", help="Coherence score (1-5)"),
    quality: str | None = typer.Option(None, "--quality", help="Quality (high/medium/low)"),
    notes: str | None = typer.Option(None, "--notes", help="Free-form notes"),
    db_path: str = db_path_option,
):
    """Tag a cluster with metadata (coherence, quality, notes)."""
    if coherence and (coherence < 1 or coherence > 5):
        console.print("[red]Error: Coherence score must be between 1 and 5[/red]")
        raise typer.Exit(1)

    if quality and quality not in ["high", "medium", "low"]:
        console.print("[red]Error: Quality must be 'high', 'medium', or 'low'[/red]")
        raise typer.Exit(1)

    db = get_db(db_path)
    result = curation_service.tag_cluster(
        db, run_id, cluster_id, coherence_score=coherence, quality=quality, notes=notes
    )

    console.print(f"[green]✓[/green] Tagged cluster {cluster_id}")
    metadata = result["metadata"]
    if metadata["coherence_score"]:
        console.print(f"  [yellow]Coherence:[/yellow] {metadata['coherence_score']}/5")
    if metadata["quality"]:
        console.print(f"  [yellow]Quality:[/yellow] {metadata['quality']}")
    if metadata["notes"]:
        console.print(f"  [yellow]Notes:[/yellow] {metadata['notes']}")


@app.command()
@with_error_handling
def flag_cluster(
    run_id: str = typer.Argument(..., help="Run ID"),
    cluster_id: int = typer.Option(..., "--cluster-id", help="Cluster ID to flag"),
    flag: str = typer.Option(..., "--flag", help="Flag to add (e.g., 'language_mixing')"),
    db_path: str = db_path_option,
):
    """Flag a cluster for review."""
    db = get_db(db_path)

    metadata = curation_service.get_cluster_metadata(db, run_id, cluster_id)
    existing_flags = metadata.flags if metadata and metadata.flags else []

    if flag not in existing_flags:
        existing_flags.append(flag)

    result = curation_service.tag_cluster(db, run_id, cluster_id, flags=existing_flags)

    console.print(f"[green]✓[/green] Flagged cluster {cluster_id}")
    console.print(f"  [yellow]Flags:[/yellow] {', '.join(result['metadata']['flags'])}")




@app.command()
@with_error_handling
def history(
    run_id: str = typer.Argument(..., help="Run ID"),
    cluster_id: int | None = typer.Option(None, "--cluster-id", help="Filter by cluster ID"),
    db_path: str = db_path_option,
):
    """Show edit history for a cluster or entire run."""
    db = get_db(db_path)
    edits = curation_service.get_cluster_edit_history(db, run_id, cluster_id)

    if not edits:
        console.print("[yellow]No edit history found[/yellow]")
        return

    title = f"Edit History for Run {run_id}"
    if cluster_id:
        title += f" - Cluster {cluster_id}"

    console.print(f"\n[bold cyan]{title}[/bold cyan]\n")

    table = Table(show_header=True)
    table.add_column("Timestamp", style="cyan")
    table.add_column("Cluster", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Editor", style="magenta")
    table.add_column("Reason", style="white")

    for edit in edits:
        table.add_row(
            edit.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            str(edit.cluster_id) if edit.cluster_id else "N/A",
            edit.edit_type,
            edit.editor,
            edit.reason or "",
        )

    console.print(table)
    console.print(f"\n[cyan]Total: {len(edits)} edits[/cyan]")


@app.command()
@with_error_handling
def audit(
    run_id: str = typer.Argument(..., help="Run ID"),
    since: str | None = typer.Option(None, "--since", help="Filter edits since date (YYYY-MM-DD)"),
    db_path: str = db_path_option,
):
    """Show full audit log for a run."""
    db = get_db(db_path)
    edits = curation_service.get_cluster_edit_history(db, run_id)

    if not edits:
        console.print("[yellow]No audit log found[/yellow]")
        return

    if since:
        from datetime import datetime

        since_date = datetime.strptime(since, "%Y-%m-%d")
        edits = [e for e in edits if e.timestamp >= since_date]

    console.print(f"\n[bold cyan]Audit Log for Run {run_id}[/bold cyan]\n")

    table = Table(show_header=True)
    table.add_column("Timestamp", style="cyan")
    table.add_column("Cluster", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Editor", style="magenta")
    table.add_column("Changes", style="white")

    for edit in edits:
        changes = f"{edit.edit_type}"
        if edit.old_value and edit.new_value:
            changes += " (see details)"

        table.add_row(
            edit.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            str(edit.cluster_id) if edit.cluster_id else "N/A",
            edit.edit_type,
            edit.editor,
            edit.reason or changes,
        )

    console.print(table)
    console.print(f"\n[cyan]Total: {len(edits)} edits[/cyan]")


@app.command()
@with_error_handling
def orphaned(
    run_id: str = typer.Argument(..., help="Run ID"),
    db_path: str = db_path_option,
):
    """List orphaned queries for a run."""
    db = get_db(db_path)
    orphans = curation_service.get_orphaned_queries(db, run_id)

    if not orphans:
        console.print("[green]No orphaned queries found[/green]")
        return

    console.print(f"\n[bold cyan]Orphaned Queries for Run {run_id}[/bold cyan]\n")

    table = Table(show_header=True)
    table.add_column("Query ID", style="cyan")
    table.add_column("Text", style="white")
    table.add_column("Original Cluster", style="green")
    table.add_column("Orphaned At", style="yellow")
    table.add_column("Reason", style="magenta")

    for orphan, query in orphans:
        text = query.query_text[:80] + "..." if len(query.query_text) > 80 else query.query_text
        table.add_row(
            str(orphan.query_id),
            text,
            str(orphan.original_cluster_id) if orphan.original_cluster_id else "N/A",
            orphan.orphaned_at.strftime("%Y-%m-%d %H:%M"),
            orphan.reason or "",
        )

    console.print(table)
    console.print(f"\n[cyan]Total: {len(orphans)} orphaned queries[/cyan]")




@app.command(name="select-bad-clusters")
@with_error_handling
def select_bad_clusters(
    run_id: str = typer.Argument(..., help="Run ID"),
    max_size: int | None = typer.Option(None, "--max-size", help="Maximum cluster size"),
    min_size: int | None = typer.Option(None, "--min-size", help="Minimum cluster size"),
    min_languages: int | None = typer.Option(
        None, "--min-languages", help="Minimum language count"
    ),
    quality: str | None = typer.Option(None, "--quality", help="Quality filter (high/medium/low)"),
    db_path: str = db_path_option,
):
    """Find clusters matching quality criteria."""
    db = get_db(db_path)
    clusters = curation_service.find_problematic_clusters(
        db,
        run_id,
        max_size=max_size,
        min_size=min_size,
        min_languages=min_languages,
        quality=quality,
    )

    if not clusters:
        console.print("[green]No clusters matching criteria found[/green]")
        return

    console.print(f"\n[bold cyan]Problematic Clusters for Run {run_id}[/bold cyan]\n")

    table = Table(show_header=True)
    table.add_column("Cluster ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Queries", style="green")
    table.add_column("Quality", style="yellow")
    table.add_column("Coherence", style="magenta")
    table.add_column("Flags", style="red")

    for cluster in clusters:
        table.add_row(
            str(cluster["cluster_id"]),
            cluster["title"] or "N/A",
            str(cluster["num_queries"]) if cluster["num_queries"] else "N/A",
            cluster["quality"] or "N/A",
            str(cluster["coherence_score"]) if cluster["coherence_score"] else "N/A",
            ", ".join(cluster["flags"]) if cluster["flags"] else "",
        )

    console.print(table)
    console.print(f"\n[cyan]Total: {len(clusters)} clusters[/cyan]")
