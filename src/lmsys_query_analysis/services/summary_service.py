"""Summary service for managing cluster summarization runs and metadata."""

import time
from typing import List, Optional, Tuple
from sqlmodel import select

from ..db.connection import Database
from ..db.models import SummaryRun, ClusterSummary, ClusteringRun
from ..clustering.summarizer import ClusterSummarizer


def create_summary_run(
    db: Database,
    run_id: str,
    summary_run_id: str,
    llm_provider: str,
    llm_model: str,
    max_queries: int,
    concurrency: int,
    rpm: Optional[int],
    contrast_neighbors: int,
    contrast_examples: int,
    contrast_mode: str,
    alias: Optional[str] = None,
) -> SummaryRun:
    """Create a new summary run record."""
    summary_run = SummaryRun(
        summary_run_id=summary_run_id,
        run_id=run_id,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_queries=max_queries,
        concurrency=concurrency,
        rpm=rpm,
        contrast_neighbors=contrast_neighbors,
        contrast_examples=contrast_examples,
        contrast_mode=contrast_mode,
        alias=alias,
    )
    
    with db.get_session() as session:
        session.add(summary_run)
        session.commit()
        session.refresh(summary_run)
        return summary_run


def update_summary_run_execution(
    db: Database,
    summary_run_id: str,
    total_clusters: int,
    execution_time_seconds: float,
) -> None:
    """Update summary run with execution metadata."""
    with db.get_session() as session:
        statement = select(SummaryRun).where(SummaryRun.summary_run_id == summary_run_id)
        summary_run = session.exec(statement).first()
        if summary_run:
            summary_run.total_clusters = total_clusters
            summary_run.execution_time_seconds = execution_time_seconds
            session.add(summary_run)
            session.commit()


def list_summary_runs(db: Database, limit: int = 50) -> List[SummaryRun]:
    """List summary runs."""
    with db.get_session() as session:
        statement = (
            select(SummaryRun)
            .order_by(SummaryRun.created_at.desc())
            .limit(limit)
        )
        return session.exec(statement).all()


def get_summary_run(db: Database, summary_run_id: str) -> Optional[SummaryRun]:
    """Get a summary run by ID."""
    with db.get_session() as session:
        statement = select(SummaryRun).where(SummaryRun.summary_run_id == summary_run_id)
        return session.exec(statement).first()


def get_summary_run_clusters(db: Database, summary_run_id: str) -> List[ClusterSummary]:
    """Get all cluster summaries for a summary run."""
    with db.get_session() as session:
        statement = (
            select(ClusterSummary)
            .where(ClusterSummary.summary_run_id == summary_run_id)
            .order_by(ClusterSummary.cluster_id)
        )
        return session.exec(statement).all()


async def create_summaries(
    db: Database,
    run_id: str,
    summary_run_id: str,
    llm_provider: str,
    llm_model: str,
    max_queries: int,
    concurrency: int,
    rpm: Optional[int],
    contrast_neighbors: int,
    contrast_examples: int,
    contrast_mode: str,
    alias: Optional[str] = None,
    cluster_ids: Optional[List[int]] = None,
) -> Tuple[str, List[dict]]:
    """Create cluster summaries with metadata persistence."""
    from ..services import cluster_service
    
    start_time = time.time()
    
    # Create summary run record
    summary_run = create_summary_run(
        db=db,
        run_id=run_id,
        summary_run_id=summary_run_id,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_queries=max_queries,
        concurrency=concurrency,
        rpm=rpm,
        contrast_neighbors=contrast_neighbors,
        contrast_examples=contrast_examples,
        contrast_mode=contrast_mode,
        alias=alias,
    )
    
    # Get clusters to summarize
    if cluster_ids is None:
        cluster_ids = cluster_service.get_cluster_ids_for_run(db, run_id)
    
    # Prepare clusters data
    clusters_data = []
    for cid in cluster_ids:
        _, query_texts = cluster_service.get_cluster_queries_with_texts(db, run_id, cid)
        clusters_data.append((cid, query_texts))
    
    # Initialize summarizer
    summarizer = ClusterSummarizer(
        model=f"{llm_provider}/{llm_model}",
        concurrency=concurrency,
        rpm=rpm,
    )
    
    # Generate summaries with LLM
    results = await summarizer.generate_batch_summaries(
        clusters_data=clusters_data,
        max_queries=max_queries,
        concurrency=concurrency,
        rpm=rpm,
        contrast_neighbors=contrast_neighbors,
        contrast_examples=contrast_examples,
        contrast_mode=contrast_mode,
    )
    
    # Store summaries in database
    with db.get_session() as session:
        for cluster_id, summary_data in results.items():
            cluster_summary = ClusterSummary(
                run_id=run_id,
                cluster_id=cluster_id,
                summary_run_id=summary_run_id,
                title=summary_data.get("title"),
                description=summary_data.get("description"),
                summary=summary_data.get("summary"),
                num_queries=len(clusters_data[cluster_id][1]) if cluster_id < len(clusters_data) else 0,
                representative_queries=summary_data.get("representative_queries", []),
                model=f"{llm_provider}/{llm_model}",
                parameters={
                    "max_queries": max_queries,
                    "concurrency": concurrency,
                    "rpm": rpm,
                    "contrast_neighbors": contrast_neighbors,
                    "contrast_examples": contrast_examples,
                    "contrast_mode": contrast_mode,
                },
            )
            session.add(cluster_summary)
        session.commit()
    
    # Update execution metadata
    execution_time = time.time() - start_time
    update_summary_run_execution(
        db=db,
        summary_run_id=summary_run_id,
        total_clusters=len(results),
        execution_time_seconds=execution_time,
    )
    
    return summary_run_id, list(results.values())
