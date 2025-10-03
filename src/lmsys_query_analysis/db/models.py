"""Database models for LMSYS query analysis using SQLModel."""
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship, Column, JSON


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

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="clustering_runs.run_id", index=True)
    query_id: int = Field(foreign_key="queries.id", index=True)
    cluster_id: int = Field(index=True)
    confidence_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    run: Optional[ClusteringRun] = Relationship(back_populates="cluster_assignments")
    query: Optional[Query] = Relationship(back_populates="cluster_assignments")


class ClusterSummary(SQLModel, table=True):
    """Table storing generated summaries/analysis for clusters."""
    __tablename__ = "cluster_summaries"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="clustering_runs.run_id", index=True)
    cluster_id: int = Field(index=True)
    title: Optional[str] = None  # LLM-generated short title
    description: Optional[str] = None  # LLM-generated description
    summary: Optional[str] = None  # Full summary text (backwards compat)
    num_queries: Optional[int] = None
    representative_queries: Optional[list] = Field(default=None, sa_column=Column(JSON))
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
    run: Optional[ClusteringRun] = Relationship(back_populates="cluster_summaries")
