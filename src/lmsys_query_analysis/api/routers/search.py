"""Search endpoints for queries and clusters (semantic and full-text)."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...clustering.embeddings import EmbeddingGenerator
from ...db.connection import Database
from ...db.models import ClusterSummary, QueryCluster
from ...db.models import Query as QueryModel
from ...semantic.clusters import ClustersClient
from ...semantic.queries import QueriesClient
from ..dependencies import create_chroma_manager, get_chroma_path, get_db
from ..schemas import (
    ClusterInfo,
    ClusterSearchResult,
    QueryResponse,
    QuerySearchResult,
    SearchClustersResponse,
    SearchQueriesResponse,
)

router = APIRouter()


@router.get(
    "/queries",
    response_model=SearchQueriesResponse,
    summary="Search queries (semantic or full-text)",
)
async def search_queries(
    text: str = Query(..., description="Search text"),
    mode: Literal["semantic", "fulltext"] = Query("fulltext", description="Search mode"),
    run_id: str | None = Query(None, description="Filter by run ID"),
    cluster_ids: str | None = Query(None, description="Comma-separated cluster IDs"),
    within_clusters: str | None = Query(
        None, description="Semantic filter: find top clusters first"
    ),
    top_clusters: int = Query(10, ge=1, le=50, description="How many clusters for within_clusters"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Results per page"),
    db: Database = Depends(get_db),
    chroma_path: str = Depends(get_chroma_path),
):
    """Search queries using semantic (ChromaDB) or full-text (SQL LIKE) search.

    **Modes:**
    - `fulltext`: SQL LIKE search on query_text (fast, no embeddings needed)
    - `semantic`: ChromaDB vector search (requires embeddings and API keys)

    **Semantic search options:**
    - `within_clusters`: Pre-filter by finding top N clusters semantically
    - `cluster_ids`: Hard filter by specific cluster IDs
    """
    if mode == "semantic":
        if not run_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "type": "ValueError",
                        "message": "run_id required for semantic search",
                    }
                },
            )

        cluster_ids_list = None
        if cluster_ids:
            try:
                cluster_ids_list = [int(x.strip()) for x in cluster_ids.split(",") if x.strip()]
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": {"type": "ValueError", "message": "Invalid cluster_ids format"}
                    },
                ) from e

        chroma_manager = create_chroma_manager(run_id, db, chroma_path)

        params = {}
        with db.get_session() as session:
            from sqlmodel import select

            from ...db.models import ClusteringRun

            run = session.exec(select(ClusteringRun).where(ClusteringRun.run_id == run_id)).first()
            if run:
                params = run.parameters or {}

        embedding_model = params.get("embedding_model", "text-embedding-3-small")
        embedding_provider = params.get("embedding_provider", "openai")
        embedding_dimension = params.get("embedding_dimension")
        if embedding_provider == "cohere" and embedding_dimension is None:
            embedding_dimension = 256

        embedder = EmbeddingGenerator(
            model_name=embedding_model,
            provider=embedding_provider,
            output_dimension=embedding_dimension if embedding_provider == "cohere" else None,
        )

        queries_client = QueriesClient(db, chroma_manager, embedder, run_id=run_id)

        hits = queries_client.find(
            text=text,
            run_id=run_id,
            cluster_ids=cluster_ids_list,
            within_clusters=within_clusters,
            top_clusters=top_clusters,
            n_results=limit * 2,
            n_candidates=500,
        )

        items = []
        for hit in hits[:limit]:
            with db.get_session() as session:
                from sqlmodel import select

                query = session.exec(
                    select(QueryModel).where(QueryModel.id == hit.query_id)
                ).first()
                if not query:
                    continue

                clusters = []
                if hit.cluster_id:
                    cluster_stmt = (
                        select(ClusterSummary)
                        .where(ClusterSummary.run_id == run_id)
                        .where(ClusterSummary.cluster_id == hit.cluster_id)
                    )
                    summary = session.exec(cluster_stmt).first()
                    clusters.append(
                        ClusterInfo(
                            run_id=run_id,
                            cluster_id=hit.cluster_id,
                            title=summary.title if summary else None,
                            confidence_score=None,
                        )
                    )

                items.append(
                    QuerySearchResult(
                        query=QueryResponse.model_validate(query),
                        clusters=clusters,
                        distance=hit.distance,
                    )
                )

        total = len(items)
        pages = 1

        return SearchQueriesResponse(
            items=items,
            total=total,
            page=page,
            pages=pages,
            limit=limit,
            facets=None,
            applied_clusters=None,
        )

    else:
        from sqlmodel import and_, select

        with db.get_session() as session:
            search_pattern = f"%{text}%"
            stmt = select(QueryModel).where(QueryModel.query_text.like(search_pattern))

            if run_id:
                stmt = (
                    select(QueryModel)
                    .join(QueryCluster, QueryModel.id == QueryCluster.query_id)
                    .where(
                        and_(
                            QueryModel.query_text.like(search_pattern),
                            QueryCluster.run_id == run_id,
                        )
                    )
                )

            all_queries = session.exec(stmt).all()
            total = len(all_queries)
            pages = (total + limit - 1) // limit
            start = (page - 1) * limit
            end = start + limit

            items = []
            for query in all_queries[start:end]:
                clusters = []
                if run_id:
                    cluster_stmt = (
                        select(QueryCluster, ClusterSummary)
                        .where(
                            QueryCluster.query_id == query.id,
                            QueryCluster.run_id == run_id,
                        )
                        .outerjoin(
                            ClusterSummary,
                            and_(
                                QueryCluster.run_id == ClusterSummary.run_id,
                                QueryCluster.cluster_id == ClusterSummary.cluster_id,
                            ),
                        )
                    )
                    assignments = session.exec(cluster_stmt).all()
                    for qc, summary in assignments:
                        clusters.append(
                            ClusterInfo(
                                run_id=qc.run_id,
                                cluster_id=qc.cluster_id,
                                title=summary.title if summary else None,
                                confidence_score=qc.confidence_score,
                            )
                        )

                items.append(
                    QuerySearchResult(
                        query=QueryResponse.model_validate(query),
                        clusters=clusters,
                        distance=None,
                    )
                )

        return SearchQueriesResponse(
            items=items,
            total=total,
            page=page,
            pages=pages,
            limit=limit,
            facets=None,
            applied_clusters=None,
        )


@router.get(
    "/clusters",
    response_model=SearchClustersResponse,
    summary="Search clusters (semantic or full-text)",
)
async def search_clusters(
    text: str = Query(..., description="Search text"),
    mode: Literal["semantic", "fulltext"] = Query("fulltext", description="Search mode"),
    run_id: str | None = Query(None, description="Filter by run ID"),
    n_results: int = Query(20, ge=1, le=100, description="Number of results"),
    page: int = Query(1, ge=1, description="Page number"),
    db: Database = Depends(get_db),
    chroma_path: str = Depends(get_chroma_path),
):
    """Search cluster summaries using semantic or full-text search.

    **Modes:**
    - `fulltext`: SQL LIKE search on title/description
    - `semantic`: ChromaDB vector search on embedded summaries
    """
    if mode == "semantic":
        if not run_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "type": "ValueError",
                        "message": "run_id required for semantic search",
                    }
                },
            )

        chroma_manager = create_chroma_manager(run_id, db, chroma_path)

        params = {}
        with db.get_session() as session:
            from sqlmodel import select

            from ...db.models import ClusteringRun

            run = session.exec(select(ClusteringRun).where(ClusteringRun.run_id == run_id)).first()
            if run:
                params = run.parameters or {}

        embedding_model = params.get("embedding_model", "text-embedding-3-small")
        embedding_provider = params.get("embedding_provider", "openai")
        embedding_dimension = params.get("embedding_dimension")
        if embedding_provider == "cohere" and embedding_dimension is None:
            embedding_dimension = 256

        embedder = EmbeddingGenerator(
            model_name=embedding_model,
            provider=embedding_provider,
            output_dimension=embedding_dimension if embedding_provider == "cohere" else None,
        )

        clusters_client = ClustersClient(db, chroma_manager, embedder, run_id=run_id)

        hits = clusters_client.find(text=text, run_id=run_id, top_k=n_results)

        items = [
            ClusterSearchResult(
                run_id=run_id,
                cluster_id=hit.cluster_id,
                title=hit.title,
                description=hit.description,
                summary=None,
                num_queries=hit.num_queries,
                distance=hit.distance,
            )
            for hit in hits
        ]

        return SearchClustersResponse(
            items=items,
            total=len(items),
            page=page,
            pages=1,
            limit=n_results,
        )

    else:
        from sqlmodel import or_, select

        with db.get_session() as session:
            search_pattern = f"%{text}%"
            stmt = select(ClusterSummary).where(
                or_(
                    ClusterSummary.title.like(search_pattern),
                    ClusterSummary.description.like(search_pattern),
                )
            )

            if run_id:
                stmt = stmt.where(ClusterSummary.run_id == run_id)

            stmt = stmt.limit(n_results)
            summaries = session.exec(stmt).all()

            items = [
                ClusterSearchResult(
                    run_id=summary.run_id,
                    cluster_id=summary.cluster_id,
                    title=summary.title,
                    description=summary.description,
                    summary=summary.summary,
                    num_queries=summary.num_queries,
                    distance=None,
                )
                for summary in summaries
            ]

        return SearchClustersResponse(
            items=items,
            total=len(items),
            page=page,
            pages=1,
            limit=n_results,
        )
