"""JSON output formatting utilities for CLI."""

from typing import Any, Dict, List


def format_search_queries_json(
    text: str,
    run_id: str,
    hits: list,
    applied_clusters: List[Dict] = None,
    facets: Dict = None,
) -> Dict[str, Any]:
    """Format query search results as JSON.
    
    Args:
        text: Search text
        run_id: Run ID filter
        hits: List of query hit objects
        applied_clusters: Optional list of applied cluster filters
        facets: Optional facets dictionary
    
    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        "text": text,
        "run_id": run_id,
        "applied_clusters": applied_clusters or [],
        "results": [
            {
                "query_id": h.query_id,
                "distance": h.distance,
                "snippet": h.snippet,
                "model": h.model,
                "language": h.language,
                "cluster_id": h.cluster_id,
            }
            for h in hits
        ],
        "facets": facets or {},
    }


def format_search_clusters_json(
    text: str, run_id: str, hits: list
) -> Dict[str, Any]:
    """Format cluster search results as JSON.
    
    Args:
        text: Search text
        run_id: Run ID filter
        hits: List of cluster hit objects
    
    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        "text": text,
        "run_id": run_id,
        "results": [
            {
                "cluster_id": h.cluster_id,
                "distance": h.distance,
                "title": h.title,
                "description": h.description,
                "num_queries": h.num_queries,
            }
            for h in hits
        ],
    }


def format_chroma_collections_json(collections: list) -> Dict[str, Any]:
    """Format ChromaDB collections as JSON.
    
    Args:
        collections: List of collection dictionaries
    
    Returns:
        Dictionary ready for JSON serialization
    """
    return {"collections": collections}


def format_verify_sync_json(
    run_id: str,
    space: Dict,
    sqlite: Dict,
    chroma: Dict,
    status: str,
    issues: List[str],
) -> Dict[str, Any]:
    """Format verification sync report as JSON.
    
    Args:
        run_id: Run ID being verified
        space: Embedding space configuration
        sqlite: SQLite statistics
        chroma: ChromaDB statistics
        status: Status string ("ok" or "mismatch")
        issues: List of issue descriptions
    
    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        "run_id": run_id,
        "space": space,
        "sqlite": sqlite,
        "chroma": chroma,
        "status": status,
        "issues": issues,
    }

