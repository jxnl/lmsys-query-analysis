"""Reusable table formatting utilities for CLI output."""

from rich.table import Table
from typing import List, Optional
from ...db.models import Query, ClusteringRun, ClusterSummary, ClusterHierarchy


def format_queries_table(queries: List[Query], title: Optional[str] = None) -> Table:
    """Format queries as a rich table.
    
    Args:
        queries: List of Query objects to format
        title: Optional custom title for the table
    
    Returns:
        Formatted Rich table
    """
    table_title = title or f"Queries ({len(queries)} shown)"
    table = Table(title=table_title)
    table.add_column("ID", style="cyan", width=8)
    table.add_column("Model", style="yellow", width=20)
    table.add_column("Query", style="white", width=80)
    table.add_column("Language", style="green", width=10)
    
    for query in queries:
        table.add_row(
            str(query.id),
            query.model[:20] if query.model else "unknown",
            (query.query_text[:77] + "...")
            if len(query.query_text) > 80
            else query.query_text,
            query.language or "?",
        )
    
    return table


def format_runs_table(runs: List[ClusteringRun], latest: bool = False) -> Table:
    """Format clustering runs as a rich table.
    
    Args:
        runs: List of ClusteringRun objects to format
        latest: Whether this is showing only the latest run
    
    Returns:
        Formatted Rich table
    """
    title = "Latest Clustering Run" if latest else "Clustering Runs"
    table = Table(title=title)
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Algorithm", style="yellow")
    table.add_column("Clusters", style="green")
    table.add_column("Created", style="magenta")
    table.add_column("Description", style="white")
    
    for run in runs:
        table.add_row(
            run.run_id,
            run.algorithm,
            str(run.num_clusters) if run.num_clusters else "?",
            run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "?",
            run.description[:50] if run.description else "",
        )
    
    return table


def format_cluster_summaries_table(
    summaries: List[ClusterSummary],
    run_id: str,
    show_examples: int = 0,
    example_width: int = 80,
) -> Table:
    """Format cluster summaries as a rich table.
    
    Args:
        summaries: List of ClusterSummary objects to format
        run_id: The clustering run ID
        show_examples: Number of example queries to show per cluster
        example_width: Max characters per example query
    
    Returns:
        Formatted Rich table
    """
    table = Table(title=f"Clusters for Run: {run_id}")
    table.add_column("Cluster", style="cyan", width=8)
    table.add_column("Title", style="yellow", width=40)
    table.add_column("Queries", style="green", width=8)
    table.add_column("Description", style="white", width=60)
    if show_examples and show_examples > 0:
        table.add_column("Examples", style="white", width=example_width + 6)
    
    for summary in summaries:
        row = [
            str(summary.cluster_id),
            summary.title or "No title",
            str(summary.num_queries) if summary.num_queries else "?",
            (summary.description[:57] + "...")
            if summary.description and len(summary.description) > 60
            else (summary.description or "No description"),
        ]
        
        if show_examples and show_examples > 0:
            reps = summary.representative_queries or []
            examples = reps[:show_examples]
            formatted = []
            for ex in examples:
                ex = ex.splitlines()[0].strip()
                if len(ex) > example_width:
                    ex = ex[: example_width - 3] + "..."
                formatted.append("- " + ex)
            row.append("\n".join(formatted) if formatted else "")
        
        table.add_row(*row)
    
    return table


def format_loading_stats_table(stats: dict) -> Table:
    """Format loading statistics as a rich table.
    
    Args:
        stats: Dictionary with keys: total_processed, loaded, skipped, errors
    
    Returns:
        Formatted Rich table
    """
    table = Table(title="Loading Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Total Processed", str(stats["total_processed"]))
    table.add_row("Loaded", str(stats["loaded"]))
    table.add_row("Skipped", str(stats["skipped"]))
    table.add_row("Errors", str(stats["errors"]))
    
    return table


def format_backfill_summary_table(
    scanned: int, backfilled: int, already_present: int, elapsed: float, rate: float
) -> Table:
    """Format backfill summary as a rich table.
    
    Args:
        scanned: Number of records scanned
        backfilled: Number of records backfilled
        already_present: Number of records already present
        elapsed: Elapsed time in seconds
        rate: Average rate per second
    
    Returns:
        Formatted Rich table
    """
    table = Table(title="Backfill Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Scanned", str(scanned))
    table.add_row("Backfilled", str(backfilled))
    table.add_row("Already Present", str(already_present))
    table.add_row("Elapsed (s)", f"{elapsed:.2f}")
    table.add_row("Avg Rate (/s)", f"{rate:.1f}")
    
    return table


def format_search_results_queries_table(hits: list) -> Table:
    """Format query search results as a rich table.
    
    Args:
        hits: List of query hit objects with attributes: query_id, snippet, model, distance
    
    Returns:
        Formatted Rich table
    """
    table = Table(title=f"Top {len(hits)} Similar Queries")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Query ID", style="yellow", width=12)
    table.add_column("Query Text", style="white", width=60)
    table.add_column("Model", style="green", width=15)
    table.add_column("Distance", style="magenta", width=10)
    
    for rank, h in enumerate(hits, 1):
        doc = h.snippet
        table.add_row(
            str(rank),
            str(h.query_id),
            doc[:60] + "..." if len(doc) > 60 else doc,
            h.model or "unknown",
            f"{h.distance:.4f}",
        )
    
    return table


def format_search_results_clusters_table(hits: list) -> Table:
    """Format cluster search results as a rich table.
    
    Args:
        hits: List of cluster hit objects with attributes: cluster_id, title, distance
    
    Returns:
        Formatted Rich table
    """
    table = Table(title=f"Top {len(hits)} Similar Clusters")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Cluster", style="green", width=10)
    table.add_column("Title", style="yellow", width=40)
    table.add_column("Distance", style="magenta", width=10)
    
    for rank, h in enumerate(hits, 1):
        table.add_row(
            str(rank),
            str(h.cluster_id),
            h.title or "(no title)",
            f"{h.distance:.4f}",
        )
    
    return table


def format_chroma_collections_table(collections: list) -> Table:
    """Format ChromaDB collections as a rich table.
    
    Args:
        collections: List of collection dicts with keys: name, count, embedding_provider, etc.
    
    Returns:
        Formatted Rich table
    """
    table = Table(title="Chroma Collections")
    table.add_column("Name", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Provider", style="yellow")
    table.add_column("Model", style="white")
    table.add_column("Dim", style="magenta")
    table.add_column("Description", style="dim")
    
    for c in collections:
        table.add_row(
            str(c["name"]),
            str(c["count"]),
            str(c.get("embedding_provider") or ""),
            str(c.get("embedding_model") or ""),
            str(c.get("embedding_dimension") or ""),
            (c.get("description") or "")[:60],
        )
    
    return table


def format_verify_sync_table(report: dict) -> Table:
    """Format verification sync report as a rich table.
    
    Args:
        report: Dictionary with verification details
    
    Returns:
        Formatted Rich table
    """
    run_id = report["run_id"]
    table = Table(title=f"Verify Sync: {run_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Provider", report["space"]["embedding_provider"])
    table.add_row("Model", report["space"]["embedding_model"])
    table.add_row("Dimension", str(report["space"]["embedding_dimension"] or ""))
    table.add_row("SQLite summaries", str(report["sqlite"]["summary_count"]))
    table.add_row("Chroma summaries", str(report["chroma"]["summary_count"]))
    table.add_row("Chroma collection", report["chroma"]["summaries_collection"])
    table.add_row("Known runs", ", ".join(report["chroma"]["runs_in_summaries"]) or "-")
    table.add_row("Status", report["status"])
    if report["issues"]:
        table.add_row("Issues", " | ".join(report["issues"]))
    
    return table


def format_hierarchy_summary_table(hierarchy_run_id: str, levels: dict) -> Table:
    """Format hierarchy summary as a rich table.
    
    Args:
        hierarchy_run_id: The hierarchy run ID
        levels: Dictionary mapping level number to count
    
    Returns:
        Formatted Rich table
    """
    table = Table(title=f"Hierarchy Summary: {hierarchy_run_id}")
    table.add_column("Level", style="cyan")
    table.add_column("Clusters", style="green")
    table.add_column("Description", style="yellow")
    
    for level in sorted(levels.keys()):
        desc = "Leaf clusters" if level == 0 else f"Merge level {level}"
        table.add_row(str(level), str(levels[level]), desc)
    
    return table

