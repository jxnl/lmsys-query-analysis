"""Search commands for queries and clusters."""

import typer
from rich.console import Console

from ..common import with_error_handling, chroma_path_option, embedding_model_option, json_output_option
from ..formatters import tables, json_output
from ..helpers.client_factory import create_queries_client, create_clusters_client
from ...db.connection import get_db

console = Console()


@with_error_handling
def search(
    text: str = typer.Argument(..., help="Search text"),
    search_type: str = typer.Option(
        "queries", help="Search type: 'queries' or 'clusters'"
    ),
    run_id: str = typer.Option(None, help="Restrict to a clustering run (provenance)"),
    cluster_ids: str = typer.Option(
        None, help="Comma-separated cluster IDs to filter (requires --run-id)"
    ),
    within_clusters: str = typer.Option(
        None, help="Semantic filter: first find top clusters by this text"
    ),
    top_clusters: int = typer.Option(10, help="How many clusters to use with --within-clusters"),
    n_results: int = typer.Option(10, help="Number of results to return"),
    n_candidates: int = typer.Option(250, help="Candidate over-fetch size from Chroma"),
    by: str = typer.Option(None, help="Group counts by: cluster|language|model"),
    facets: str = typer.Option(None, help="Compute facets for: cluster,language,model (comma-separated)"),
    json_out: bool = json_output_option,
    chroma_path: str = chroma_path_option,
    embedding_model: str = embedding_model_option,
):
    """Semantic search across queries or cluster summaries using ChromaDB.

    For queries: supports provenance via --run-id, semantic conditioning via
    --within-clusters, and grouped counts/facets. Use --json for machine output.
    """
    # Parse cluster_ids list
    cluster_ids_list = None
    if cluster_ids:
        try:
            cluster_ids_list = [int(x.strip()) for x in cluster_ids.split(",") if x.strip()]
        except Exception:
            console.print("[red]Invalid --cluster-ids. Use comma-separated integers.[/red]")
            raise typer.Exit(1)
    
    db = get_db(None)
    
    if search_type == "queries":
        # Create clients
        qclient = create_queries_client(db, run_id, embedding_model, chroma_path)
        cclient = create_clusters_client(db, run_id, embedding_model, chroma_path)
        
        # Optionally compute applied clusters for JSON when using within-clusters
        applied_clusters = []
        if within_clusters:
            applied_clusters = [
                {
                    "cluster_id": h.cluster_id,
                    "title": h.title,
                    "description": h.description,
                    "num_queries": h.num_queries,
                    "distance": h.distance,
                }
                for h in cclient.find(
                    text=within_clusters,
                    run_id=run_id,
                    top_k=top_clusters,
                )
            ]
        
        # Fetch query hits
        hits = qclient.find(
            text=text,
            run_id=run_id,
            cluster_ids=cluster_ids_list,
            within_clusters=within_clusters,
            top_clusters=top_clusters,
            n_results=n_results,
            n_candidates=n_candidates,
        )
        
        # Aggregations
        facets_spec = []
        if facets:
            facets_spec = [f.strip() for f in facets.split(",") if f.strip()]
        
        facets_out = {}
        if by:
            # Map grouped counts to facets JSON shape under key 'clusters' when by=cluster
            grouped = qclient.count(
                text=text,
                run_id=run_id,
                cluster_ids=cluster_ids_list,
                within_clusters=within_clusters,
                top_clusters=top_clusters,
                by=by,
            )
            if isinstance(grouped, dict):
                key = "clusters" if by == "cluster" else by
                facets_out[key] = [
                    {"cluster_id": k, "count": v} if by == "cluster" else {"key": k, "count": v}
                    for k, v in grouped.items()
                ]
        
        if facets_spec:
            fac = qclient.facets(
                text=text,
                run_id=run_id,
                cluster_ids=cluster_ids_list,
                within_clusters=within_clusters,
                top_clusters=top_clusters,
                facet_by=facets_spec,
            )
            # Convert Pydantic objects to plain dicts
            for facet_key, buckets in fac.items():
                if facet_key == "cluster":
                    facets_out["clusters"] = [
                        {"cluster_id": b.key, "count": b.count, **({} if not b.meta else {"meta": b.meta})}
                        for b in buckets
                    ]
                else:
                    facets_out[facet_key] = [
                        {"key": b.key, "count": b.count, **({} if not b.meta else {"meta": b.meta})}
                        for b in buckets
                    ]
        
        if json_out:
            payload = json_output.format_search_queries_json(
                text, run_id, hits, applied_clusters, facets_out
            )
            console.print_json(data=payload)
            return
        else:
            if not hits:
                console.print("[yellow]No results found[/yellow]")
                return
            
            table = tables.format_search_results_queries_table(hits)
            console.print(table)
    
    elif search_type == "clusters":
        # Use SDK for cluster summaries search
        cclient = create_clusters_client(db, run_id, embedding_model, chroma_path)
        
        chits = cclient.find(text=text, run_id=run_id, top_k=n_results)
        if json_out:
            payload = json_output.format_search_clusters_json(text, run_id, chits)
            console.print_json(data=payload)
            return
        if not chits:
            console.print("[yellow]No results found[/yellow]")
            return
        table = tables.format_search_results_clusters_table(chits)
        console.print(table)
    
    else:
        console.print(f"[red]Invalid search_type: {search_type}[/red]")
        raise typer.Exit(1)


@with_error_handling
def search_cluster(
    text: str = typer.Argument(..., help="Search text for cluster summaries"),
    run_id: str = typer.Option(None, help="Restrict to a clustering run"),
    alias: str = typer.Option(None, help="Filter by summary alias"),
    summary_run_id: str = typer.Option(None, help="Filter by summary run id"),
    top_k: int = typer.Option(20, help="Number of cluster hits"),
    json_out: bool = json_output_option,
    chroma_path: str = chroma_path_option,
    embedding_model: str = embedding_model_option,
):
    """Search cluster titles+descriptions (summaries)."""
    db = get_db(None)
    client = create_clusters_client(db, run_id, embedding_model, chroma_path)
    
    hits = client.find(
        text=text,
        run_id=run_id,
        top_k=top_k,
        alias=alias,
        summary_run_id=summary_run_id,
    )
    
    if json_out:
        payload = json_output.format_search_clusters_json(text, run_id, hits)
        console.print_json(data=payload)
        return
    
    if not hits:
        console.print("[yellow]No results found[/yellow]")
        return
    
    table = tables.format_search_results_clusters_table(hits)
    console.print(table)

