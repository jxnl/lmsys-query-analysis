"""Cluster hierarchy commands."""

import anyio
import typer
from collections import defaultdict
from rich.console import Console

from ..common import with_error_handling, db_path_option, embedding_model_option
from ..formatters import tables
from ..helpers.client_factory import parse_embedding_model
from ...db.connection import get_db
from ...services import cluster_service
from ...db.models import ClusterSummary, ClusterHierarchy
from ...clustering.hierarchy import merge_clusters_hierarchical
from sqlmodel import select

console = Console()


@with_error_handling
def merge_clusters_cmd(
    run_id: str = typer.Argument(..., help="Clustering run ID to create hierarchy from"),
    db_path: str = db_path_option,
    summary_run_id: str = typer.Option(None, help="Specific summary run ID to merge (defaults to latest)"),
    model: str = typer.Option(
        "openai/gpt-4o-mini", help="LLM (provider/model) for merging"
    ),
    embedding_model: str = embedding_model_option,
    target_levels: int = typer.Option(
        3, help="Number of hierarchy levels (1=flat, 2=one merge, 3=two merges, etc.)"
    ),
    merge_ratio: float = typer.Option(
        0.35, help="Target merge ratio per level (0.35 = 1000->350->120->42, higher=more specific)"
    ),
    neighborhood_size: int = typer.Option(
        20, help="Average clusters per neighborhood for LLM context (smaller=more specific, 15-25 recommended)"
    ),
    concurrency: int = typer.Option(
        8, help="Max concurrent LLM requests"
    ),
    rpm: int = typer.Option(
        None, help="Optional rate limit (requests per minute)"
    ),
):
    """Create hierarchical organization of clusters using LLM-driven merging.

    Follows Anthropic's Clio approach:
    1. Embed cluster summaries
    2. Group into neighborhoods
    3. LLM generates higher-level categories
    4. Deduplicate and assign children to parents
    5. Refine parent names based on children
    6. Repeat for multiple hierarchy levels

    Example:
        uv run lmsys merge-clusters kmeans-100-20251004-170442 --target-levels 3
    """
    embed_model, embed_provider = parse_embedding_model(embedding_model)
    llm_provider, llm_model = model.split("/", 1)
    db = get_db(db_path)
    
    with db.get_session() as session:
        # If summary_run_id not provided, find the latest one
        if not summary_run_id:
            summary_run_id = cluster_service.get_latest_summary_run_id(db, run_id)
            
            if not summary_run_id:
                console.print(f"[red]No summaries found for run {run_id}[/red]")
                console.print("[yellow]Run 'lmsys summarize <run_id>' first[/yellow]")
                raise typer.Exit(1)
            
            console.print(f"[cyan]Using latest summary run: {summary_run_id}[/cyan]")
        
        # Load base cluster summaries
        console.print(f"[cyan]Loading cluster summaries for run: {run_id}, summary: {summary_run_id}[/cyan]")
        
        summaries = session.exec(
            select(ClusterSummary)
            .where(ClusterSummary.run_id == run_id)
            .where(ClusterSummary.summary_run_id == summary_run_id)
            .order_by(ClusterSummary.cluster_id)
        ).all()
        
        if not summaries:
            console.print(f"[red]No summaries found for run {run_id} with summary_run_id {summary_run_id}[/red]")
            console.print("[yellow]Run 'lmsys summarize <run_id>' first[/yellow]")
            raise typer.Exit(1)
        
        base_clusters = [
            {
                "cluster_id": s.cluster_id,
                "title": s.title or f"Cluster {s.cluster_id}",
                "description": s.description or "No description"
            }
            for s in summaries
        ]
        
        console.print(f"[green]Found {len(base_clusters)} base clusters[/green]")
        
        # Run hierarchical merging
        console.print(f"[cyan]Starting hierarchical merging with {target_levels} levels[/cyan]")
        
        async def run_merge():
            return await merge_clusters_hierarchical(
                base_clusters,
                run_id,
                embedding_model=embed_model,
                embedding_provider=embed_provider,
                llm_provider=llm_provider,
                llm_model=llm_model,
                target_levels=target_levels,
                merge_ratio=merge_ratio,
                neighborhood_size=neighborhood_size,
                concurrency=concurrency,
                rpm=rpm
            )
        
        hierarchy_run_id, hierarchy_data = anyio.run(run_merge)
        
        # Save hierarchy to database
        console.print(f"[cyan]Saving hierarchy to database...[/cyan]")
        
        for h in hierarchy_data:
            hierarchy_entry = ClusterHierarchy(**h)
            session.add(hierarchy_entry)
        
        session.commit()
        
        # Display summary
        levels = {}
        for h in hierarchy_data:
            level = h["level"]
            if level not in levels:
                levels[level] = 0
            levels[level] += 1
        
        summary_table = tables.format_hierarchy_summary_table(hierarchy_run_id, levels)
        console.print(summary_table)
        console.print(f"\n[green]✓ Hierarchy saved: {hierarchy_run_id}[/green]")
        console.print(f"[dim]Use 'lmsys show-hierarchy {hierarchy_run_id}' to explore[/dim]")


@with_error_handling
def show_hierarchy_cmd(
    hierarchy_run_id: str = typer.Argument(..., help="Hierarchy run ID to visualize"),
    db_path: str = db_path_option,
):
    """Display hierarchical cluster structure as a tree."""
    db = get_db(db_path)
    
    with db.get_session() as session:
        # Get all hierarchy nodes
        hierarchy_nodes = session.exec(
            select(ClusterHierarchy)
            .where(ClusterHierarchy.hierarchy_run_id == hierarchy_run_id)
            .order_by(ClusterHierarchy.level, ClusterHierarchy.cluster_id)
        ).all()
        
        if not hierarchy_nodes:
            console.print(f"[red]No hierarchy found with ID: {hierarchy_run_id}[/red]")
            raise typer.Exit(1)
        
        # Build tree structure
        nodes_by_id = {node.cluster_id: node for node in hierarchy_nodes}
        children_by_parent = defaultdict(list)
        root_nodes = []
        
        for node in hierarchy_nodes:
            if node.parent_cluster_id is None:
                root_nodes.append(node)
            else:
                children_by_parent[node.parent_cluster_id].append(node)
        
        def print_tree(node, prefix="", is_last=True):
            """Recursively print tree structure."""
            connector = "└── " if is_last else "├── "
            title = node.title or f"Cluster {node.cluster_id}"
            
            # Color coding by level
            if node.level == 0:
                color = "cyan"
            elif node.level == 1:
                color = "yellow"
            else:
                color = "green"
            
            console.print(f"{prefix}{connector}[{color}]{title}[/{color}] [dim](ID: {node.cluster_id}, Level: {node.level})[/dim]")
            
            # Get children
            children = children_by_parent.get(node.cluster_id, [])
            for i, child in enumerate(sorted(children, key=lambda x: x.cluster_id)):
                extension = "    " if is_last else "│   "
                print_tree(child, prefix + extension, i == len(children) - 1)
        
        console.print(f"\n[bold]Hierarchy: {hierarchy_run_id}[/bold]\n")
        
        # Print each root and its subtree
        for i, root in enumerate(sorted(root_nodes, key=lambda x: x.cluster_id)):
            print_tree(root, "", i == len(root_nodes) - 1)
        
        console.print()

