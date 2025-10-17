"""JSON and XML output formatting utilities for CLI."""

from typing import Any

from jinja2 import Template


def format_search_queries_json(
    text: str,
    run_id: str,
    hits: list,
    applied_clusters: list[dict] = None,
    facets: dict = None,
) -> dict[str, Any]:
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


def format_search_clusters_json(text: str, run_id: str, hits: list) -> dict[str, Any]:
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


def format_chroma_collections_json(collections: list) -> dict[str, Any]:
    """Format ChromaDB collections as JSON.

    Args:
        collections: List of collection dictionaries

    Returns:
        Dictionary ready for JSON serialization
    """
    return {"collections": collections}


def format_verify_sync_json(
    run_id: str,
    space: dict,
    sqlite: dict,
    chroma: dict,
    status: str,
    issues: list[str],
) -> dict[str, Any]:
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


# XML Templates
CLUSTER_SUMMARIES_XML_TEMPLATE = Template("""<?xml version="1.0" encoding="UTF-8"?>
<clusters run_id="{{ run_id }}" count="{{ summaries|length }}">
{%- for summary in summaries %}
  <cluster id="{{ summary.cluster_id }}">
    <title>{{ summary.title or '' }}</title>
    <description>{{ summary.description or '' }}</description>
    <num_queries>{{ summary.num_queries or 0 }}</num_queries>
    {%- if summary.representative_queries %}
    <representative_queries>
      {%- for query in summary.representative_queries %}
      <query>{{ query }}</query>
      {%- endfor %}
    </representative_queries>
    {%- endif %}
  </cluster>
{%- endfor %}
</clusters>""")

QUERIES_XML_TEMPLATE = Template("""<?xml version="1.0" encoding="UTF-8"?>
<queries{% if title %} title="{{ title }}"{% endif %} count="{{ queries|length }}">
{%- for query in queries %}
  <query id="{{ query.id }}">
    <model>{{ query.model or 'unknown' }}</model>
    <text>{{ query.query_text }}</text>
    <language>{{ query.language or 'unknown' }}</language>
  </query>
{%- endfor %}
</queries>""")

RUNS_XML_TEMPLATE = Template("""<?xml version="1.0" encoding="UTF-8"?>
<runs latest_only="{{ latest|lower }}" count="{{ runs|length }}">
{%- for run in runs %}
  <run id="{{ run.run_id }}">
    <algorithm>{{ run.algorithm }}</algorithm>
    <num_clusters>{{ run.num_clusters or 0 }}</num_clusters>
    <created_at>{{ run.created_at.isoformat() if run.created_at else '' }}</created_at>
    <description>{{ run.description or '' }}</description>
  </run>
{%- endfor %}
</runs>""")

SEARCH_QUERIES_XML_TEMPLATE = Template("""<?xml version="1.0" encoding="UTF-8"?>
<search_results search_text="{{ search_text }}" type="queries" count="{{ hits|length }}">
{%- for hit in hits %}
  <result rank="{{ loop.index }}" query_id="{{ hit.query_id }}" distance="{{ "%.4f"|format(hit.distance) }}">
    <snippet>{{ hit.snippet }}</snippet>
    <model>{{ hit.model or 'unknown' }}</model>
  </result>
{%- endfor %}
</search_results>""")

SEARCH_CLUSTERS_XML_TEMPLATE = Template("""<?xml version="1.0" encoding="UTF-8"?>
<search_results search_text="{{ search_text }}" type="clusters" count="{{ hits|length }}">
{%- for hit in hits %}
  <result rank="{{ loop.index }}" cluster_id="{{ hit.cluster_id }}" distance="{{ "%.4f"|format(hit.distance) }}">
    <title>{{ hit.title or '' }}</title>
    <description>{{ hit.description or '' }}</description>
  </result>
{%- endfor %}
</search_results>""")

CHROMA_COLLECTIONS_XML_TEMPLATE = Template("""<?xml version="1.0" encoding="UTF-8"?>
<chroma_collections count="{{ collections|length }}">
{%- for collection in collections %}
  <collection name="{{ collection.name }}">
    <count>{{ collection.count }}</count>
    <embedding_provider>{{ collection.get('embedding_provider', '') }}</embedding_provider>
    <embedding_model>{{ collection.get('embedding_model', '') }}</embedding_model>
    <embedding_dimension>{{ collection.get('embedding_dimension', '') }}</embedding_dimension>
    <description>{{ collection.get('description', '') }}</description>
  </collection>
{%- endfor %}
</chroma_collections>""")


def format_cluster_summaries_xml(summaries: list, run_id: str) -> str:
    """Format cluster summaries as XML using Jinja template."""
    return CLUSTER_SUMMARIES_XML_TEMPLATE.render(summaries=summaries, run_id=run_id)


def format_queries_xml(queries: list, title: str = None) -> str:
    """Format queries as XML using Jinja template."""
    return QUERIES_XML_TEMPLATE.render(queries=queries, title=title)


def format_runs_xml(runs: list, latest: bool = False) -> str:
    """Format clustering runs as XML using Jinja template."""
    return RUNS_XML_TEMPLATE.render(runs=runs, latest=latest)


def format_search_results_queries_xml(text: str, hits: list) -> str:
    """Format query search results as XML using Jinja template."""
    return SEARCH_QUERIES_XML_TEMPLATE.render(search_text=text, hits=hits)


def format_search_results_clusters_xml(text: str, hits: list) -> str:
    """Format cluster search results as XML using Jinja template."""
    return SEARCH_CLUSTERS_XML_TEMPLATE.render(search_text=text, hits=hits)


def format_chroma_collections_xml(collections: list) -> str:
    """Format ChromaDB collections as XML using Jinja template."""
    return CHROMA_COLLECTIONS_XML_TEMPLATE.render(collections=collections)
