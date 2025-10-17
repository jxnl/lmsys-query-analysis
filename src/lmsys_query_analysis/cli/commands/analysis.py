"""Analysis and inspection commands."""

import typer
from rich.console import Console

from ...db.connection import get_db
from ...services import cluster_service, export_service, query_service, run_service
from ..common import db_path_option, table_output_option, with_error_handling, xml_output_option
from ..formatters import json_output, tables

console = Console()


@with_error_handling
def list_queries(
    run_id: str = typer.Option(None, help="Filter by run ID"),
    cluster_id: int = typer.Option(None, help="Filter by cluster ID"),
    model: str = typer.Option(None, help="Filter by model name"),
    limit: int = typer.Option(50, help="Number of queries to display"),
    db_path: str = db_path_option,
    table: bool = table_output_option,
    xml: bool = xml_output_option,
):
    """List queries with optional filtering by run_id and cluster_id."""
    # Validate format options
    if table and xml:
        console.print("[red]Error: Cannot specify both --table and --xml[/red]")
        raise typer.Exit(1)

    db = get_db(db_path)
    queries = query_service.list_queries(db, run_id, cluster_id, model, limit)

    if not queries:
        console.print("[yellow]No queries found for given filters[/yellow]")
        return

    # Output in requested format
    if xml:
        xml_output = json_output.format_queries_xml(queries, f"Queries ({len(queries)} shown)")
        console.print(xml_output)
    else:
        # Default to table format
        table_output = tables.format_queries_table(queries)
        console.print(table_output)


@with_error_handling
def runs(
    db_path: str = db_path_option,
    latest: bool = typer.Option(False, "--latest", help="Show only the most recent run"),
    table: bool = table_output_option,
    xml: bool = xml_output_option,
):
    """List all clustering runs."""
    # Validate format options
    if table and xml:
        console.print("[red]Error: Cannot specify both --table and --xml[/red]")
        raise typer.Exit(1)

    db = get_db(db_path)
    runs_list = run_service.list_runs(db, latest=latest)

    if not runs_list:
        console.print("[yellow]No clustering runs found[/yellow]")
        return

    # Output in requested format
    if xml:
        xml_output = json_output.format_runs_xml(runs_list, latest=latest)
        console.print(xml_output)
    else:
        # Default to table format
        table_output = tables.format_runs_table(runs_list, latest=latest)
        console.print(table_output)


@with_error_handling
def list_clusters(
    run_id: str = typer.Argument(..., help="Run ID to list clusters for"),
    db_path: str = db_path_option,
    summary_run_id: str = typer.Option(None, help="Filter by specific summary run ID"),
    alias: str = typer.Option(None, help="Filter by summary alias (e.g., 'claude-v1')"),
    limit: int = typer.Option(None, help="Limit number of clusters to show"),
    show_examples: int = typer.Option(0, help="Show up to N example queries per cluster"),
    example_width: int = typer.Option(80, help="Max characters per example query"),
    table: bool = table_output_option,
    xml: bool = xml_output_option,
):
    """List all clusters for a run with their titles and descriptions."""
    # Validate format options
    if table and xml:
        console.print("[red]Error: Cannot specify both --table and --xml[/red]")
        raise typer.Exit(1)

    db = get_db(db_path)
    summaries = cluster_service.list_cluster_summaries(db, run_id, summary_run_id, alias, limit)

    if not summaries:
        console.print(f"[yellow]No summaries found for run {run_id}[/yellow]")
        console.print(f"[cyan]Run 'lmsys summarize {run_id}' to generate summaries[/cyan]")
        return

    # Output in requested format
    if xml:
        xml_output = json_output.format_cluster_summaries_xml(summaries, run_id)
        console.print(xml_output)
    else:
        # Default to table format
        table_output = tables.format_cluster_summaries_table(
            summaries, run_id, show_examples, example_width
        )
        console.print(table_output)

        # Display examples in detail if requested (only for table format)
        if show_examples and show_examples > 0:
            console.print("\n[bold cyan]Examples per cluster[/bold cyan]")
            for summary in summaries:
                reps = summary.representative_queries or []
                if not reps:
                    continue
                console.print(
                    f"[yellow]Cluster {summary.cluster_id}[/yellow] — {summary.title or ''}"
                )
                for ex in reps[:show_examples]:
                    ex_line = ex.splitlines()[0].strip()
                    if len(ex_line) > example_width:
                        ex_line = ex_line[: example_width - 3] + "..."
                    console.print(f"  - {ex_line}")

        console.print(f"\n[cyan]Total: {len(summaries)} clusters[/cyan]")


@with_error_handling
def inspect(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    cluster_id: int = typer.Argument(..., help="Cluster ID to inspect"),
    db_path: str = db_path_option,
    show_queries: int = typer.Option(10, help="Number of queries to show"),
):
    """Inspect specific cluster in detail."""
    db = get_db(db_path)

    # Get cluster summary
    summary = cluster_service.get_cluster_summary(db, run_id, cluster_id)

    # Get queries in cluster
    queries = query_service.get_cluster_queries(db, run_id, cluster_id)

    if not queries:
        console.print(f"[yellow]No queries found in cluster {cluster_id}[/yellow]")
        return

    # Display cluster info
    console.print(f"\n[cyan]{'=' * 80}[/cyan]")
    console.print(f"[bold cyan]Cluster {cluster_id} from run {run_id}[/bold cyan]")
    console.print(f"[cyan]{'=' * 80}[/cyan]\n")

    if summary:
        console.print(f"[bold yellow]Title:[/bold yellow] {summary.title or 'N/A'}")
        console.print("\n[bold yellow]Description:[/bold yellow]")
        console.print(f"{summary.description or 'N/A'}\n")

    console.print(f"[bold green]Total Queries:[/bold green] {len(queries)}\n")

    # Show sample queries
    console.print(
        f"[bold yellow]Sample Queries (showing {min(show_queries, len(queries))}):[/bold yellow]\n"
    )

    for i, query in enumerate(queries[:show_queries], 1):
        console.print(f"[cyan]{i}.[/cyan] {query.query_text}")
        console.print(
            f"   [dim]Model: {query.model} | Language: {query.language or 'unknown'}[/dim]\n"
        )

    if len(queries) > show_queries:
        console.print(f"[dim]... and {len(queries) - show_queries} more queries[/dim]\n")

    console.print(
        f"[cyan]Use 'lmsys list --run-id {run_id} --cluster-id {cluster_id}' to see all queries[/cyan]"
    )


@with_error_handling
def export(
    run_id: str = typer.Argument(..., help="Run ID to export"),
    output: str = typer.Option("export.csv", help="Output file path"),
    format: str = typer.Option("csv", help="Export format: csv or json"),
    db_path: str = db_path_option,
):
    """Export cluster results to file."""
    db = get_db(db_path)

    # Get export data
    data = export_service.get_export_data(db, run_id)

    if not data:
        console.print(f"[yellow]No data found for run {run_id}[/yellow]")
        return

    console.print(f"[cyan]Exporting {len(data)} queries...[/cyan]")

    # Export based on format
    if format == "csv":
        count = export_service.export_to_csv(output, data)
    elif format == "json":
        count = export_service.export_to_json(output, data)
    else:
        console.print(f"[red]Unsupported format: {format}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Exported {count} records to {output}[/green]")
