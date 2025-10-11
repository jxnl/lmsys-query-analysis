"""Database models for LMSYS query analysis using SQLModel."""

from datetime import datetime
import uuid
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship, Column, JSON
from sqlalchemy import UniqueConstraint, ForeignKey, Index


class Query(SQLModel, table=True):
    """Table storing extracted first prompts from LMSYS-1M dataset."""

    __tablename__ = "queries"

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: str = Field(unique=True, index=True)
    model: str = Field(index=True)
    query_text: str
    language: Optional[str] = Field(default=None, index=True)
    timestamp: Optional[datetime] = None
    extra_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship to cluster assignments
    cluster_assignments: list["QueryCluster"] = Relationship(back_populates="query")


class ClusteringRun(SQLModel, table=True):
    """Table tracking clustering experiments."""

    __tablename__ = "clustering_runs"

    run_id: str = Field(primary_key=True)
    algorithm: str
    parameters: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    num_clusters: Optional[int] = None

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

    id: Optional[int] = Field(default=None, primary_key=True)
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
    confidence_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    run: Optional[ClusteringRun] = Relationship(back_populates="cluster_assignments")
    query: Optional[Query] = Relationship(back_populates="cluster_assignments")


class ClusterSummary(SQLModel, table=True):
    """Table storing generated summaries/analysis for clusters.

    Multiple summary runs can exist for the same clustering run, allowing
    comparison of different LLM models, prompts, or summarization parameters.
    """

    __tablename__ = "cluster_summaries"
    __table_args__ = (
        UniqueConstraint("run_id", "cluster_id", "summary_run_id", name="uq_clustersummary_run_cluster_summary"),
        Index("ix_cluster_summaries_run_id", "run_id"),
        Index("ix_cluster_summaries_cluster_id", "cluster_id"),
        Index("ix_cluster_summaries_summary_run_id", "summary_run_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    cluster_id: int
    # Unique ID for this summarization run; default to timestamp-based if not provided
    summary_run_id: str = Field(default_factory=lambda: f"summary-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}")
    alias: Optional[str] = None  # Friendly name for this summary run (e.g., "claude-v1", "gpt4-test")
    title: Optional[str] = None  # LLM-generated short title
    description: Optional[str] = None  # LLM-generated description
    summary: Optional[str] = None  # Full summary text (backwards compat)
    num_queries: Optional[int] = None
    representative_queries: Optional[list] = Field(default=None, sa_column=Column(JSON))
    model: Optional[str] = None  # LLM model used (e.g., "anthropic/claude-sonnet-4-5-20250929")
    parameters: Optional[dict] = Field(default=None, sa_column=Column(JSON))  # Summarization parameters
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
    run: Optional[ClusteringRun] = Relationship(back_populates="cluster_summaries")


class HierarchyRun(SQLModel, table=True):
    """Table storing metadata for hierarchical merging runs.

    Tracks the configuration and parameters used to create cluster hierarchies,
    enabling reproducibility, comparison, and audit trails for different merge strategies.
    """

    __tablename__ = "hierarchy_runs"
    __table_args__ = (
        Index("ix_hierarchy_runs_run_id", "run_id"),
        Index("ix_hierarchy_runs_llm_provider", "llm_provider"),
        Index("ix_hierarchy_runs_embedding_provider", "embedding_provider"),
        Index("ix_hierarchy_runs_created_at", "created_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    hierarchy_run_id: str = Field(unique=True)  # Primary identifier (e.g., "hier-kmeans-200-20251004-170442-20251004-123456")
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    
    # LLM Configuration
    llm_provider: str  # e.g., "openai", "anthropic", "groq"
    llm_model: str  # e.g., "gpt-4o-mini", "claude-sonnet-4-5"
    
    # Embedding Configuration
    embedding_provider: str  # e.g., "openai", "cohere"
    embedding_model: str  # e.g., "text-embedding-3-small", "embed-v4.0"
    
    # Merge Parameters
    target_levels: int  # Number of hierarchy levels created
    merge_ratio: float  # Target merge ratio per level
    neighborhood_size: int  # Average clusters per neighborhood
    concurrency: int  # Max concurrent LLM requests
    rpm: Optional[int] = None  # Rate limit (requests per minute)
    
    # Summary Configuration
    summary_run_id: Optional[str] = None  # Which summary run was used as input
    
    # Execution Metadata
    total_nodes: Optional[int] = None  # Total hierarchy nodes created
    execution_time_seconds: Optional[float] = None  # Time taken to complete
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationship
    run: Optional[ClusteringRun] = Relationship()


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

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    hierarchy_run_id: str = Field(
        sa_column=Column(
            "hierarchy_run_id",
            ForeignKey("hierarchy_runs.hierarchy_run_id", ondelete="CASCADE"),
        ),
    )
    cluster_id: int  # The cluster being organized (can be virtual for merged clusters)
    parent_cluster_id: Optional[int] = None  # Parent cluster ID (null for top level)
    level: int  # 0=leaf (base clusters), 1=first merge, 2=second merge, etc.
    children_ids: Optional[list] = Field(default=None, sa_column=Column(JSON))  # List of child cluster IDs
    title: Optional[str] = None  # Title for merged/parent clusters
    description: Optional[str] = None  # Description for merged/parent clusters
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    run: Optional[ClusteringRun] = Relationship()
    hierarchy_run: Optional[HierarchyRun] = Relationship()


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

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    cluster_id: Optional[int] = None  # Null for query-level edits
    edit_type: str  # 'rename', 'move_query', 'merge', 'split', 'delete', 'tag'
    editor: str  # 'claude', 'cli-user', or username
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Change tracking
    old_value: Optional[dict] = Field(default=None, sa_column=Column(JSON))  # Previous state
    new_value: Optional[dict] = Field(default=None, sa_column=Column(JSON))  # New state
    reason: Optional[str] = None  # Why the edit was made

    # Relationship
    run: Optional[ClusteringRun] = Relationship()


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

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(
        sa_column=Column(
            "run_id",
            ForeignKey("clustering_runs.run_id", ondelete="CASCADE"),
        ),
    )
    cluster_id: int
    coherence_score: Optional[int] = None  # 1-5 scale
    quality: Optional[str] = None  # 'high', 'medium', 'low'
    flags: Optional[list] = Field(default=None, sa_column=Column(JSON))  # ['language_mixing', 'needs_review', etc.]
    notes: Optional[str] = None  # Free-form notes
    last_edited: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
    run: Optional[ClusteringRun] = Relationship()


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

    id: Optional[int] = Field(default=None, primary_key=True)
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
    original_cluster_id: Optional[int] = None
    orphaned_at: datetime = Field(default_factory=datetime.utcnow)
    reason: Optional[str] = None

    # Relationships
    run: Optional[ClusteringRun] = Relationship()
    query: Optional[Query] = Relationship()
