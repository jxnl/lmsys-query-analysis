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

    # Relationship to cluster assignments
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

    # Relationships
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

    # Relationships
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
    # Unique ID for this summarization run; default to timestamp-based if not provided
    summary_run_id: str = Field(
        default_factory=lambda: f"summary-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}"
    )
    alias: str | None = None  # Friendly name for this summary run (e.g., "claude-v1", "gpt4-test")
    title: str | None = None  # LLM-generated short title
    description: str | None = None  # LLM-generated description
    summary: str | None = None  # Full summary text (backwards compat)
    num_queries: int | None = None
    representative_queries: list | None = Field(default=None, sa_column=Column(JSON))
    model: str | None = None  # LLM model used (e.g., "anthropic/claude-sonnet-4-5-20250929")
    parameters: dict | None = Field(
        default=None, sa_column=Column(JSON)
    )  # Summarization parameters
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
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
    hierarchy_run_id: str  # Unique ID for this hierarchy run (e.g., "hier-20251004-123456")
    cluster_id: int  # The cluster being organized (can be virtual for merged clusters)
    parent_cluster_id: int | None = None  # Parent cluster ID (null for top level)
    level: int  # 0=leaf (base clusters), 1=first merge, 2=second merge, etc.
    children_ids: list | None = Field(
        default=None, sa_column=Column(JSON)
    )  # List of child cluster IDs
    title: str | None = None  # Title for merged/parent clusters
    description: str | None = None  # Description for merged/parent clusters
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
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
    cluster_id: int | None = None  # Null for query-level edits
    edit_type: str  # 'rename', 'move_query', 'merge', 'split', 'delete', 'tag'
    editor: str  # 'claude', 'cli-user', or username
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Change tracking
    old_value: dict | None = Field(default=None, sa_column=Column(JSON))  # Previous state
    new_value: dict | None = Field(default=None, sa_column=Column(JSON))  # New state
    reason: str | None = None  # Why the edit was made

    # Relationship
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
    coherence_score: int | None = None  # 1-5 scale
    quality: str | None = None  # 'high', 'medium', 'low'
    flags: list | None = Field(
        default=None, sa_column=Column(JSON)
    )  # ['language_mixing', 'needs_review', etc.]
    notes: str | None = None  # Free-form notes
    last_edited: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
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

    # Relationships
    run: ClusteringRun | None = Relationship()
    query: Query | None = Relationship()
