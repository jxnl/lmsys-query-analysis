"""
Configuration for LMSYS Query Analysis Runner.

Provides Pydantic models for configuring the analysis workflow with validation
and support for loading from YAML files.
"""

from typing import Optional, Literal
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator


class RunnerConfig(BaseModel):
    """Configuration for the complete analysis workflow."""

    # Data loading
    query_limit: int = Field(default=1000, gt=0, description="Number of queries to load")
    skip_existing: bool = Field(default=True, description="Skip queries already in database")
    use_streaming: bool = Field(default=False, description="Use streaming for large datasets")

    # Embedding configuration
    embedding_model: str = Field(default="embed-v4.0", description="Embedding model name")
    embedding_provider: Literal["cohere", "openai", "sentence-transformers"] = Field(
        default="cohere",
        description="Embedding provider"
    )
    embedding_batch_size: int = Field(default=100, gt=0, description="Batch size for embedding generation")
    embed_batch_size: int = Field(default=32, gt=0, description="Batch size for clustering embeddings")

    # Clustering configuration
    n_clusters: int = Field(default=50, gt=0, description="Number of clusters for KMeans")
    chunk_size: int = Field(default=5000, gt=0, description="Chunk size for MiniBatch KMeans")
    mb_batch_size: int = Field(default=4096, gt=0, description="MiniBatch size for KMeans")
    random_state: int = Field(default=42, description="Random state for reproducibility")

    # Hierarchy configuration
    enable_hierarchy: bool = Field(default=True, description="Enable hierarchical clustering")
    hierarchy_levels: int = Field(default=3, gt=0, description="Number of hierarchy levels")
    merge_ratio: float = Field(default=0.3, gt=0, lt=1, description="Merge ratio for hierarchy")
    neighborhood_size: int = Field(default=40, gt=0, description="Neighborhood size for hierarchy")
    concurrency: int = Field(default=8, gt=0, description="Concurrent LLM calls for hierarchy")
    rpm: int = Field(default=300, gt=0, description="Rate limit (requests per minute)")

    # LLM configuration
    llm_provider: Literal["anthropic", "openai", "groq"] = Field(
        default="anthropic",
        description="LLM provider for summarization and hierarchy"
    )
    llm_model: str = Field(default="claude-3-5-sonnet-20241022", description="LLM model name")

    # Database configuration
    db_path: Optional[str] = Field(default=None, description="Persistent database path")
    chroma_path: Optional[str] = Field(default=None, description="Persistent ChromaDB path")

    # Runtime configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    cleanup_temp: bool = Field(default=True, description="Clean up temporary files after run")

    @field_validator("n_clusters")
    @classmethod
    def validate_n_clusters(cls, v, info):
        """Ensure n_clusters is reasonable relative to query_limit."""
        query_limit = info.data.get("query_limit")
        if query_limit and v > query_limit:
            raise ValueError(
                f"n_clusters ({v}) cannot exceed query_limit ({query_limit}). "
                f"Consider reducing n_clusters or increasing query_limit."
            )
        return v

    @field_validator("db_path", "chroma_path")
    @classmethod
    def resolve_paths(cls, v):
        """Convert relative paths to absolute paths."""
        if v is None:
            return v
        path = Path(v)
        if not path.is_absolute():
            path = Path.cwd() / path
        return str(path)

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"


def load_config_from_yaml(yaml_path: str) -> RunnerConfig:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Validated RunnerConfig instance

    Example YAML:
        ```yaml
        query_limit: 10000
        n_clusters: 200
        enable_hierarchy: true
        hierarchy_levels: 4
        llm_provider: anthropic
        llm_model: claude-3-5-sonnet-20241022
        log_level: INFO
        db_path: ./analysis.db
        ```
    """
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return RunnerConfig(**config_dict)


def save_config_to_yaml(config: RunnerConfig, yaml_path: str) -> None:
    """Save configuration to a YAML file.

    Args:
        config: RunnerConfig instance to save
        yaml_path: Path to save YAML file
    """
    with open(yaml_path, "w") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
