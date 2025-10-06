"""Usage scaffold for the Semantic SDK stubs.

This script demonstrates the intended flow to consume the SDK once the
implementations land. It does not run end-to-end yet.
"""

from pathlib import Path
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.semantic import ClustersClient, QueriesClient


def main() -> None:
    # Resolve your SQLite DB (defaults to ~/.lmsys-query-analysis/queries.db)
    db = Database()

    # Option A: construct clients from a known run_id (preferred for provenance)
    run_id = "kmeans-50-20250101-abc123"
    clusters = ClustersClient.from_run(db, run_id, persist_dir=Path.home() / ".lmsys-query-analysis" / "chroma")
    queries = QueriesClient.from_run(db, run_id, persist_dir=Path.home() / ".lmsys-query-analysis" / "chroma")

    # Cluster discovery (summary search)
    cluster_hits = clusters.find(
        text="vector databases",
        run_id=run_id,  # optional when using from_run
        top_k=10,
        alias=None,  # or e.g., "gpt4-claude-compare"
    )

    # Query discovery (within selected clusters)
    query_hits = queries.find(
        text="hybrid search",
        run_id=run_id,
        # Option 1: two-stage selection via within_clusters
        within_clusters="vector databases",
        top_clusters=5,
        # Option 2: direct filtering by known cluster_ids
        # cluster_ids=[12, 27, 44],
        n_results=50,
        n_candidates=250,
    )

    # Aggregations and facets over a semantic slice
    counts_by_cluster = queries.count(
        text="vector",
        run_id=run_id,
        by="cluster",
    )

    facet_buckets = queries.facets(
        text="vector",
        run_id=run_id,
        facet_by=["cluster", "language"],
    )

    # The above variables are not printed to avoid noise in a stub script.
    # Replace with actual printing or assertions when implementing.


if __name__ == "__main__":
    main()

