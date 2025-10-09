#!/usr/bin/env python
"""
Example: Complete LMSYS Analysis Pipeline

This script demonstrates the full workflow:
1. Load 1000 queries from LMSYS-1M dataset
2. Run KMeans clustering with 100 clusters
3. Generate LLM summaries for clusters
4. Create hierarchical organization (optional)
5. Query and explore results

Usage:
    uv run python examples/example_runner.py

Requirements:
    - ANTHROPIC_API_KEY environment variable
    - COHERE_API_KEY environment variable
    - huggingface-cli login (for LMSYS-1M access)
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmsys_query_analysis import (
    AnalysisRunner,
    RunnerConfig,
    load_config_from_yaml,
    save_config_to_yaml,
)
from lmsys_query_analysis.db.connection import Database, DEFAULT_DB_PATH
from lmsys_query_analysis.db.chroma import ChromaManager, DEFAULT_CHROMA_PATH
from lmsys_query_analysis.semantic.clusters import ClustersClient
from lmsys_query_analysis.semantic.queries import QueriesClient
from lmsys_query_analysis.services import cluster_service

console = Console()


def print_header(title: str):
    """Print a formatted header."""
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(60)}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def display_cluster_results(results, title: str, max_results: int = 3):
    """Display cluster search results in a formatted table with summary grouping."""
    if not results:
        console.print("[yellow]  No clusters found. Try generating summaries first.[/yellow]")
        return

    # Create table
    table = Table(title=title, show_header=True)
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Cluster ID", style="yellow", width=12)
    table.add_column("Title", style="green", width=30)
    table.add_column("Summary ID", style="blue", width=20)
    table.add_column("Score", style="magenta", width=8)

    # Add rows and collect summary stats
    summary_counts = {}
    for i, hit in enumerate(results[:max_results], 1):
        title_text = hit.title[:30] + "..." if len(hit.title) > 30 else hit.title
        summary_id = getattr(hit, 'summary_run_id', 'N/A')

        # Track summary distribution
        summary_counts[summary_id] = summary_counts.get(summary_id, 0) + 1

        # Format summary ID for display
        summary_id_display = summary_id[:17] + "..." if summary_id and len(summary_id) > 20 else summary_id or "N/A"

        table.add_row(
            str(i),
            str(hit.cluster_id),
            title_text,
            summary_id_display,
            f"{hit.score:.3f}"
        )

    console.print(table)

    # Show distribution if multiple summaries
    if len(summary_counts) > 1:
        console.print(f"\n[dim]Results from {len(summary_counts)} different summary runs:[/dim]")
        for summary_id, count in summary_counts.items():
            short_id = summary_id[:30] + "..." if summary_id and len(summary_id) > 30 else summary_id
            console.print(f"[dim]  • {short_id}: {count} results[/dim]")


def display_query_results(results, title: str, max_results: int = 5):
    """Display query search results in a formatted table."""
    if not results:
        console.print("[yellow]  No queries found.[/yellow]")
        return

    table = Table(title=title, show_header=True)
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Query", style="green", width=60)
    table.add_column("Distance", style="magenta", width=10)

    for i, hit in enumerate(results[:max_results], 1):
        query_text = hit.snippet.replace("\n", " ")[:60]
        if len(hit.snippet) > 60:
            query_text += "..."
        table.add_row(str(i), query_text, f"{hit.distance:.4f}")

    console.print(table)


async def demonstrate_searches(run_id: str, hierarchy_run_id: str, db_path: str | None = None):
    """Demonstrate semantic search on clusters and queries."""

    console.print("[bold]Demonstrating semantic search capabilities[/bold]\n")
    console.print(f"[dim]Run ID: {run_id}[/dim]")
    console.print(f"[dim]Hierarchy: {hierarchy_run_id or 'Not created'}[/dim]\n")

    db = Database(db_path)
    chroma = ChromaManager()

    try:
        # Initialize search clients (run_id defines the vector space)
        clusters_client = ClustersClient.from_run(db, run_id)
        queries_client = QueriesClient.from_run(db, run_id)

        # Cluster Search 1: Programming
        console.print("[cyan]Search 1: Clusters about 'programming languages'[/cyan]")
        try:
            results = clusters_client.find("programming languages", top_k=3)
            display_cluster_results(results, "Top 3 Programming Clusters")
        except Exception as e:
            console.print(f"[yellow]  Search failed: {e}[/yellow]")

        console.print()

        # Cluster Search 2: AI/ML
        console.print("[cyan]Search 2: Clusters about 'artificial intelligence'[/cyan]")
        try:
            results = clusters_client.find("artificial intelligence machine learning", top_k=3)
            display_cluster_results(results, "Top 3 AI/ML Clusters")
        except Exception as e:
            console.print(f"[yellow]  Search failed: {e}[/yellow]")

        console.print()

        # Optional: Demonstrate filtering by summary_run_id
        try:
            with db.get_session() as session:
                from lmsys_query_analysis.db.models import ClusterSummary
                from sqlmodel import select

                stmt = select(ClusterSummary.summary_run_id).where(
                    ClusterSummary.run_id == run_id
                ).distinct().limit(2)
                summaries = list(session.exec(stmt))

                if len(summaries) >= 2:
                    console.print("[cyan]Advanced: Filtering by summary_run_id[/cyan]")
                    console.print(f"[dim]Filtering to: {summaries[0][:40]}...[/dim]")
                    results = clusters_client.find("programming", top_k=2, summary_run_id=summaries[0])
                    console.print(f"  Found {len(results)} clusters from this specific summary\n")
        except Exception:
            pass  # Optional demo, skip if error

        # Query Search: Python Tutorials
        console.print("[cyan]Search 3: Queries about 'python tutorial'[/cyan]")
        try:
            results = queries_client.find("python tutorial for beginners", n_results=5)
            display_query_results(results, "Top 5 Python Queries")
        except Exception as e:
            console.print(f"[yellow]  Search failed: {e}[/yellow]")

        console.print()

        # Display Hierarchy
        console.print("[bold]ID System Summary[/bold]")
        console.print("• [cyan]run_id[/cyan]: Vector space (required for all searches)")
        console.print("• [cyan]summary_run_id[/cyan]: Which LLM descriptions (optional filter)")
        console.print("• [cyan]hierarchy_run_id[/cyan]: Organization tree (display only)\n")

        if hierarchy_run_id:
            console.print("[cyan]Hierarchy Structure[/cyan]")
            display_hierarchy(db, hierarchy_run_id)
        else:
            console.print("[yellow]Hierarchy not available[/yellow]")

    finally:
        pass  # Resources auto-cleanup


def display_hierarchy(db: Database, hierarchy_run_id: str, max_levels: int = 3):
    """Display hierarchical cluster organization with statistics."""
    from lmsys_query_analysis.db.models import ClusterHierarchy
    from sqlmodel import select

    with db.get_session() as session:
        stmt = select(ClusterHierarchy).where(
            ClusterHierarchy.hierarchy_run_id == hierarchy_run_id
        ).order_by(ClusterHierarchy.level, ClusterHierarchy.cluster_id)
        hierarchies = list(session.exec(stmt))

        if not hierarchies:
            console.print("[yellow]  No hierarchy data found[/yellow]")
            return

        # Group by level
        by_level = {}
        for h in hierarchies:
            by_level.setdefault(h.level, []).append(h)

        # Hierarchy table
        table = Table(title="Cluster Hierarchy", show_header=True)
        table.add_column("Level", style="cyan", width=8)
        table.add_column("Clusters", style="yellow", width=10)
        table.add_column("Sample Titles", style="green", width=60)

        for level in sorted(by_level.keys())[:max_levels]:
            nodes = by_level[level]
            samples = [n.title or f"Cluster {n.cluster_id}" for n in nodes[:3]]
            samples = [s[:30] + "..." if len(s) > 30 else s for s in samples]

            if len(nodes) > 3:
                samples.append(f"+{len(nodes) - 3} more")

            table.add_row(str(level), str(len(nodes)), ", ".join(samples))

        console.print(table)

        # Statistics
        max_level = max(by_level.keys()) if by_level else 0
        console.print(f"\n[dim]Total: {len(hierarchies)} nodes across {len(by_level)} levels[/dim]")
        console.print(f"[dim]Leaves: {len(by_level.get(0, []))} | Roots: {len(by_level.get(max_level, []))}[/dim]")


async def run_example():
    """Run the complete analysis pipeline example."""
    print_header("STEP 1: Configuration")

    config = RunnerConfig(
        query_limit=1000,
        n_clusters=100,
        enable_hierarchy=True,
        hierarchy_levels=3,
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet-20241022",
        db_path=str(DEFAULT_DB_PATH),  # ~/.lmsys-query-analysis/queries.db
        chroma_path=str(DEFAULT_CHROMA_PATH),  # ~/.lmsys-query-analysis/chroma
        log_level="INFO",
    )

    console.print("[green]✓[/green] Configuration created")
    console.print(f"  • {config.query_limit} queries, {config.n_clusters} clusters")
    console.print(f"  • Embedding: {config.embedding_provider}/{config.embedding_model}")
    console.print(f"  • LLM: {config.llm_provider}/{config.llm_model}")
    console.print(f"  • Hierarchy: {config.hierarchy_levels} levels")

    config_path = "./examples/example_config.yaml"
    save_config_to_yaml(config, config_path)
    console.print(f"\n[green]✓[/green] Saved to {config_path}")

    print_header("STEP 2: Running Analysis")
    console.print("[yellow]This will take 15-25 minutes. Press Ctrl+C to cancel.[/yellow]\n")

    try:
        runner = AnalysisRunner(config)
        results = await runner.run()

        print_header("STEP 3: Results")
        console.print(f"[green]✓[/green] Run ID: {results['run_id']}")
        if results.get('hierarchy_run_id'):
            console.print(f"[green]✓[/green] Hierarchy: {results['hierarchy_run_id']}")
        console.print(f"[green]✓[/green] Processed: {results['total_queries']} queries")
        console.print(f"[green]✓[/green] Time: {results['execution_time']:.2f}s\n")

        print_header("STEP 4: Search Demonstrations")

        await demonstrate_searches(
            run_id=results['run_id'],
            hierarchy_run_id=results.get('hierarchy_run_id'),
            db_path=config.db_path
        )

        # ============================================================
        # STEP 5: Next Steps
        # ============================================================
        print_header("STEP 5: Explore Your Results")

        next_steps = Panel(
            f"""[bold]Your analysis is complete! Here's how to explore it:[/bold]

[cyan]1. View all runs:[/cyan]
   uv run lmsys runs

[cyan]2. List cluster summaries (after summarization):[/cyan]
   uv run lmsys list-clusters {results['run_id']}

[cyan]3. Generate cluster summaries:[/cyan]
   uv run lmsys summarize {results['run_id']}

[cyan]4. Search queries semantically:[/cyan]
   uv run lmsys search "python programming" --run-id {results['run_id']}

[cyan]5. Inspect a specific cluster:[/cyan]
   uv run lmsys inspect {results['run_id']} <cluster_id>

[cyan]6. Export results to JSON:[/cyan]
   uv run lmsys export {results['run_id']} --output results.json

[cyan]7. Create hierarchy (if not enabled above):[/cyan]
   uv run lmsys merge-clusters {results['run_id']}

[dim]Note: All commands use default paths (~/.lmsys-query-analysis/)[/dim]
""",
            title="Next Steps",
            border_style="green"
        )

        console.print(next_steps)

        return results

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Analysis cancelled by user[/yellow]")
        return None
    except Exception as e:
        console.print(f"\n[red]✗ Analysis failed: {e}[/red]")
        console.print("\n[yellow]Common issues:[/yellow]")
        console.print("  • Missing API keys (COHERE_API_KEY, ANTHROPIC_API_KEY)")
        console.print("  • Not logged into Hugging Face (run: huggingface-cli login)")
        console.print("  • Insufficient disk space or memory")
        raise


async def run_from_yaml_example():
    """Example: Load configuration from YAML and run."""
    print_header("Example: Loading from YAML")

    config_path = "./examples/example_config.yaml"

    if not Path(config_path).exists():
        console.print(f"[yellow]Config file not found: {config_path}[/yellow]")
        console.print("Run the main example first to create it.")
        return

    console.print(f"Loading configuration from {config_path}...")
    config = load_config_from_yaml(config_path)

    console.print("[green]✓[/green] Configuration loaded")
    console.print(f"  • Queries: {config.query_limit}")
    console.print(f"  • Clusters: {config.n_clusters}")

    # Run analysis
    runner = AnalysisRunner(config)
    results = await runner.run()

    console.print(f"\n[green]✓[/green] Analysis complete: {results['run_id']}")


def print_usage():
    """Print usage instructions."""
    usage = Panel(
        """[bold]LMSYS Analysis Pipeline Example[/bold]

This example demonstrates the complete workflow for analyzing LMSYS queries:

[cyan]Prerequisites:[/cyan]
  1. Set environment variables:
     export COHERE_API_KEY="your-key"
     export ANTHROPIC_API_KEY="your-key"

  2. Login to Hugging Face:
     huggingface-cli login

  3. Accept LMSYS-1M dataset terms at:
     https://huggingface.co/datasets/lmsys/lmsys-chat-1m

[cyan]Running:[/cyan]
  uv run python examples/example_runner.py

[cyan]What it does:[/cyan]
  • Loads 1,000 queries from LMSYS-1M dataset
  • Generates embeddings using Cohere embed-v4.0
  • Clusters queries into 100 groups using KMeans
  • Saves results to ~/.lmsys-query-analysis/queries.db
  • Provides commands to explore results

[cyan]Customization:[/cyan]
  Edit the RunnerConfig in this file to adjust:
  • query_limit (number of queries)
  • n_clusters (number of clusters)
  • enable_hierarchy (create multi-level hierarchy)
  • llm_provider/llm_model (for summarization)

[cyan]Duration:[/cyan]
  • 1,000 queries + 100 clusters: ~5-10 minutes
  • With hierarchy: +10-20 minutes
""",
        title="Example Runner",
        border_style="blue"
    )
    console.print(usage)


def main():
    """Main entry point."""
    import os

    # Check if running in non-interactive mode
    auto_run = os.getenv("AUTO_RUN", "false").lower() == "true"

    try:
        # Print usage
        print_usage()

        # Prompt to continue (unless auto-run)
        if not auto_run:
            console.print("\n[yellow]Press Enter to continue or Ctrl+C to cancel...[/yellow]")
            input()

        # Run the example
        results = asyncio.run(run_example())

        if results:
            console.print("\n[bold green]✓ Example completed successfully![/bold green]")
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Example cancelled[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Example failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
