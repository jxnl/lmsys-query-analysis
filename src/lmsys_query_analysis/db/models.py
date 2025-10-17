"""Database models for LMSYS query analysis using SQLModel."""

from datetime import datetime

from sqlalchemy import ForeignKey, Index, UniqueConstraint
from sqlmodel import JSON, Column, Field, Relationship, SQLModel


class Query(SQLModel, table=True):
    """Table storing extracted first prompts from LMSYS-1M dataset."""

    __tablename__ = "queries"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: str = Field(unique=True, index=True)
    model: str = Field(index=True)
    query_text: str
    language: str | None = Field(default=None, index=True)
    timestamp: datetime | None = None
    extra_metadata: dict | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    cluster_assignments: list["QueryCluster"] = Relationship(back_populates="query")


class ClusteringRun(SQLModel, table=True):
    """Table tracking clustering experiments."""

    __tablename__ = "clustering_runs"

    run_id: str = Field(primary_key=True)
    algorithm: str
    parameters: dict | None = Field(default=None, sa_column=Column(JSON))
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    num_clusters: int | None = None

    cluster_assignments: list["QueryCluster"] = Relationship(back_populates="run")
    cluster_summaries: list["ClusterSummary"] = Relationship(back_populates="run")


class QueryCluster(SQLModel, table=True):
    """Table mapping queries to clusters per run."""

    __tablename__ = "query_clusters"
    __table_args__ = (
        UniqueConstraint("run_id", "query_id", name="uq_querycluster_run_query"),
        Index("ix_query_clusters_run_id", "run_id"),
        Index("ix_query_clusters_query_id", "query_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    query_id: int = Field(
        sa_column=Column(
            "query_id",
            ForeignKey("queries.id", ondelete="CASCADE"),
        ),
    )
    cluster_id: int = Field(index=True)
    confidence_score: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    run: ClusteringRun | None = Relationship(back_populates="cluster_assignments")
    query: Query | None = Relationship(back_populates="cluster_assignments")


class ClusterSummary(SQLModel, table=True):
    """Table storing generated summaries/analysis for clusters.

    Multiple summary runs can exist for the same clustering run, allowing
    comparison of different LLM models, prompts, or summarization parameters.
    """

    __tablename__ = "cluster_summaries"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "cluster_id", "summary_run_id", name="uq_clustersummary_run_cluster_summary"
        ),
        Index("ix_cluster_summaries_run_id", "run_id"),
        Index("ix_cluster_summaries_cluster_id", "cluster_id"),
        Index("ix_cluster_summaries_summary_run_id", "summary_run_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    cluster_id: int
    summary_run_id: str = Field(
        default_factory=lambda: f"summary-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}"
    )
    alias: str | None = None
    title: str | None = None
    description: str | None = None
    summary: str | None = None
    num_queries: int | None = None
    representative_queries: list | None = Field(default=None, sa_column=Column(JSON))
    model: str | None = None
    parameters: dict | None = Field(
        default=None, sa_column=Column(JSON)
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    run: ClusteringRun | None = Relationship(back_populates="cluster_summaries")


class ClusterHierarchy(SQLModel, table=True):
    """Table storing hierarchical relationships between clusters.

    Supports multi-level hierarchies where clusters can be organized into
    parent-child relationships, enabling drill-down navigation from broad
    categories to specific topics.
    """

    __tablename__ = "cluster_hierarchies"
    __table_args__ = (
        UniqueConstraint("hierarchy_run_id", "cluster_id", name="uq_hierarchy_run_cluster"),
        Index("ix_cluster_hierarchies_hierarchy_run_id", "hierarchy_run_id"),
        Index("ix_cluster_hierarchies_cluster_id", "cluster_id"),
        Index("ix_cluster_hierarchies_parent_cluster_id", "parent_cluster_id"),
        Index("ix_cluster_hierarchies_level", "level"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    hierarchy_run_id: str
    cluster_id: int
    parent_cluster_id: int | None = None
    level: int
    children_ids: list | None = Field(
        default=None, sa_column=Column(JSON)
    )
    title: str | None = None
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    run: ClusteringRun | None = Relationship()


class ClusterEdit(SQLModel, table=True):
    """Table storing audit trail for all cluster curation operations.

    Tracks WHO changed WHAT and WHY, enabling full provenance and history
    for cluster modifications made through CLI or web interface.
    """

    __tablename__ = "cluster_edits"
    __table_args__ = (
        Index("ix_cluster_edits_run_id", "run_id"),
        Index("ix_cluster_edits_cluster_id", "cluster_id"),
        Index("ix_cluster_edits_edit_type", "edit_type"),
        Index("ix_cluster_edits_timestamp", "timestamp"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    cluster_id: int | None = None
    edit_type: str
    editor: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    old_value: dict | None = Field(default=None, sa_column=Column(JSON))
    new_value: dict | None = Field(default=None, sa_column=Column(JSON))
    reason: str | None = None

    run: ClusteringRun | None = Relationship()


class ClusterMetadata(SQLModel, table=True):
    """Table storing quality annotations and metadata for clusters.

    Enables quality tracking, filtering, and flagging of clusters that
    need attention or review.
    """

    __tablename__ = "cluster_metadata"
    __table_args__ = (
        UniqueConstraint("run_id", "cluster_id", name="uq_clustermetadata_run_cluster"),
        Index("ix_cluster_metadata_run_id", "run_id"),
        Index("ix_cluster_metadata_quality", "quality"),
        Index("ix_cluster_metadata_coherence_score", "coherence_score"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    cluster_id: int
    coherence_score: int | None = None
    quality: str | None = None
    flags: list | None = Field(
        default=None, sa_column=Column(JSON)
    )
    notes: str | None = None
    last_edited: datetime = Field(default_factory=datetime.utcnow)

    run: ClusteringRun | None = Relationship()


class OrphanedQuery(SQLModel, table=True):
    """Table tracking queries that have been removed from clusters.

    Maintains provenance for queries removed during cluster deletion or
    manual curation operations.
    """

    __tablename__ = "orphaned_queries"
    __table_args__ = (
        UniqueConstraint("run_id", "query_id", name="uq_orphanedquery_run_query"),
        Index("ix_orphaned_queries_run_id", "run_id"),
        Index("ix_orphaned_queries_query_id", "query_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    query_id: int = Field(
        sa_column=Column(
            "query_id",
            ForeignKey("queries.id", ondelete="CASCADE"),
        ),
    )
    original_cluster_id: int | None = None
    orphaned_at: datetime = Field(default_factory=datetime.utcnow)
    reason: str | None = None

    run: ClusteringRun | None = Relationship()
    query: Query | None = Relationship()
