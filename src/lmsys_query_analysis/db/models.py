"""Database models for LMSYS query analysis using SQLModel."""

from datetime import datetime
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
    summary_run_id: str  # Unique ID for this summarization run
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
    hierarchy_run_id: str  # Unique ID for this hierarchy run (e.g., "hier-20251004-123456")
    cluster_id: int  # The cluster being organized (can be virtual for merged clusters)
    parent_cluster_id: Optional[int] = None  # Parent cluster ID (null for top level)
    level: int  # 0=leaf (base clusters), 1=first merge, 2=second merge, etc.
    children_ids: Optional[list] = Field(default=None, sa_column=Column(JSON))  # List of child cluster IDs
    title: Optional[str] = None  # Title for merged/parent clusters
    description: Optional[str] = None  # Description for merged/parent clusters
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
    run: Optional[ClusteringRun] = Relationship()
