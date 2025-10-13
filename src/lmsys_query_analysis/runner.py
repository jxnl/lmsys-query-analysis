"""
LMSYS Query Analysis Runner

A simple procedural module that runs the complete LMSYS analysis workflow 
with configurable parameters and comprehensive logging.

Can be used programmatically or as a simple script.
"""

import os
import time
import tempfile
import shutil
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel

# Rich for progress and console output
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# LMSYS SDK imports
from .db.connection import Database
from .db.loader import load_queries
from .db.sources import HuggingFaceSource
from .db.chroma import ChromaManager
from .clustering.kmeans import run_kmeans_clustering
from .clustering.hierarchy import merge_clusters_hierarchical
from .services import cluster_service

# Configuration
from .config import RunnerConfig, load_config_from_yaml

console = Console()
logger = logging.getLogger(__name__)


class BaseCluster(BaseModel):
    """Typed model for base cluster information."""
    cluster_id: int
    title: str
    description: str


class AnalysisRunner:
    """End-to-end LMSYS query analysis workflow runner.

    Simple wrapper around the run_analysis function with async support.
    Resource management is handled internally by run_analysis.

    Example:
        ```python
        config = RunnerConfig(query_limit=1000, n_clusters=50)
        runner = AnalysisRunner(config)
        results = await runner.run()
        ```
    """

    def __init__(self, config: RunnerConfig):
        """Initialize runner with configuration.

        Args:
            config: RunnerConfig with all workflow parameters
        """
        self.config = config

    async def run(self) -> Dict[str, Any]:
        """Run the complete analysis workflow.

        Returns:
            Dictionary with run_id, hierarchy_run_id, stats, and timing info
        """
        return await run_analysis(self.config)


def setup_logging(log_level: str) -> None:
    """Set up logging with Rich handler and proper formatting."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)]
    )
    
    # Set specific logger levels for different components
    logging.getLogger("lmsys").setLevel(getattr(logging, log_level.upper()))
    logging.getLogger("src.lmsys_query_analysis").setLevel(getattr(logging, log_level.upper()))
    
    logger.info(f"Logging initialized at {log_level} level")


def validate_api_keys() -> None:
    """Check required API keys exist."""
    logger.info("Validating required API keys...")
    
    required = ["COHERE_API_KEY", "OPENAI_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    
    if missing:
        logger.error(f"Missing required API keys: {missing}")
        logger.error("Set the following environment variables:")
        for key in missing:
            logger.error(f"  export {key}=your_api_key_here")
        raise ValueError(f"Missing API keys: {missing}")
    
    logger.info("✓ All required API keys found")


def create_temp_directory() -> Path:
    """Create temporary directory for databases and files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="lmsys_runner_"))
    logger.info(f"Created temporary directory: {temp_dir}")
    return temp_dir


def initialize_database(temp_dir: Path, persistent_path: Optional[str] = None) -> Database:
    """Initialize SQLite database in temporary or persistent location."""
    if persistent_path:
        db_path = Path(persistent_path)
        logger.info(f"Using persistent database: {db_path}")
    else:
        db_path = temp_dir / "queries.db"
        logger.info(f"Using temporary database: {db_path}")
    
    # Create parent directory if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    db = Database(str(db_path))
    db.create_tables()
    
    logger.info("✓ Database initialized and tables created")
    return db


def initialize_chroma(temp_dir: Path) -> ChromaManager:
    """Initialize ChromaDB in temporary directory."""
    chroma_path = temp_dir / "chroma"
    logger.info(f"Initializing ChromaDB: {chroma_path}")
    
    chroma = ChromaManager(str(chroma_path))
    
    logger.info("✓ ChromaDB initialized")
    return chroma


def setup_environment(config: RunnerConfig) -> Tuple[Path, Database, ChromaManager]:
    """Set up temporary database and ChromaDB."""
    logger.info("Setting up analysis environment...")
    
    # Set up logging first
    setup_logging(config.log_level)
    
    # Validate API keys
    validate_api_keys()
    
    # Create temporary directory
    temp_dir = create_temp_directory()
    
    # Initialize database
    db = initialize_database(temp_dir, config.db_path)
    
    # Initialize ChromaDB
    chroma = initialize_chroma(temp_dir)
    
    logger.info("✓ Environment setup complete")
    return temp_dir, db, chroma


def load_data(db: Database, chroma: ChromaManager, config: RunnerConfig) -> Dict[str, Any]:
    """Load LMSYS dataset with comprehensive logging."""
    logger.info(f"Starting data loading phase (limit: {config.query_limit})")

    start_time = time.time()

    try:
        # Create HuggingFace source for LMSYS dataset
        source = HuggingFaceSource(
            dataset_id="lmsys/lmsys-chat-1m",
            limit=config.query_limit,
            streaming=config.use_streaming
        )
        
        stats = load_queries(
            db=db,
            source=source,
            skip_existing=config.skip_existing,
            chroma=chroma,
            embedding_model=config.embedding_model,
            embedding_provider=config.embedding_provider,
            batch_size=config.embedding_batch_size,
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"✓ Data loading complete in {elapsed:.2f}s")
        logger.info(f"  Source: {stats['source']}")
        logger.info(f"  Total processed: {stats['total_processed']}")
        logger.info(f"  Successfully loaded: {stats['loaded']}")
        logger.info(f"  Skipped (existing): {stats['skipped']}")
        logger.info(f"  Errors: {stats['errors']}")
        
        if stats['loaded'] == 0:
            logger.warning("No new queries were loaded - they may already exist in the database")
        
        return stats
        
    except Exception as e:
        logger.error(f"Data loading failed after {time.time() - start_time:.2f}s: {e}")
        raise


def run_clustering(db: Database, chroma: ChromaManager, config: RunnerConfig) -> str:
    """Run K-means clustering with comprehensive logging."""
    logger.info(f"Starting clustering phase (n_clusters: {config.n_clusters})")

    start_time = time.time()

    try:
        run_id = run_kmeans_clustering(
            db=db,
            n_clusters=config.n_clusters,
            description=f"Runner clustering with {config.n_clusters} clusters",
            embedding_model=config.embedding_model,
            embedding_provider=config.embedding_provider,
            embed_batch_size=config.embed_batch_size,
            chunk_size=config.chunk_size,
            mb_batch_size=config.mb_batch_size,
            random_state=config.random_state,
            chroma=chroma
        )
        
        elapsed = time.time() - start_time
        
        if not run_id:
            logger.error("Clustering failed - no run ID returned")
            raise RuntimeError("Clustering failed to return a run ID")
        
        logger.info(f"✓ Clustering complete in {elapsed:.2f}s")
        logger.info(f"  Run ID: {run_id}")
        
        # Log cluster statistics
        cluster_ids = cluster_service.get_cluster_ids_for_run(db, run_id)
        logger.info(f"  Created {len(cluster_ids)} non-empty clusters")
        
        return run_id
        
    except Exception as e:
        logger.error(f"Clustering failed after {time.time() - start_time:.2f}s: {e}")
        raise


def extract_base_clusters(db: Database, run_id: str, min_clusters: int = 10) -> List[BaseCluster]:
    """Get base cluster data for hierarchy creation.

    Args:
        db: Database connection
        run_id: Clustering run ID
        min_clusters: Minimum expected clusters (logs warning if below this)

    Returns:
        List of BaseCluster models with cluster_id, title, and description
    """
    logger.info(f"Extracting base clusters for run: {run_id}")

    try:
        summaries = cluster_service.list_cluster_summaries(db, run_id)

        if not summaries:
            logger.warning(f"No cluster summaries found for run {run_id}")
            # Try to get cluster IDs directly
            cluster_ids = cluster_service.get_cluster_ids_for_run(db, run_id)
            logger.info(f"Found {len(cluster_ids)} cluster IDs, creating basic summaries")

            base_clusters = [
                BaseCluster(
                    cluster_id=cluster_id,
                    title=f"Cluster {cluster_id}",
                    description=f"Cluster {cluster_id} (no summary available)"
                )
                for cluster_id in cluster_ids
            ]
        else:
            base_clusters = [
                BaseCluster(
                    cluster_id=summary.cluster_id,
                    title=summary.title or f"Cluster {summary.cluster_id}",
                    description=summary.summary or f"Cluster {summary.cluster_id} (no description)"
                )
                for summary in summaries
            ]

        logger.info(f"✓ Extracted {len(base_clusters)} base clusters")

        # Warn if below minimum threshold
        if len(base_clusters) < min_clusters:
            logger.warning(
                f"Only {len(base_clusters)} base clusters found - hierarchy may not be meaningful "
                f"(expected at least {min_clusters})"
            )

        # Log sample cluster info
        for cluster in base_clusters[:3]:
            logger.debug(f"  Cluster {cluster.cluster_id}: {cluster.title[:50]}...")

        if len(base_clusters) > 3:
            logger.debug(f"  ... and {len(base_clusters) - 3} more clusters")

        return base_clusters
        
    except Exception as e:
        logger.error(f"Failed to extract base clusters for run {run_id}: {e}")
        raise


async def create_hierarchy(db: Database, run_id: str, config: RunnerConfig) -> str:
    """Create cluster hierarchy with comprehensive logging."""
    logger.info(f"Starting hierarchy creation for run: {run_id}")
    logger.info(f"  Target levels: {config.hierarchy_levels}")
    logger.info(f"  Merge ratio: {config.merge_ratio}")
    logger.info(f"  LLM: {config.llm_provider}/{config.llm_model}")
    
    start_time = time.time()
    
    try:
        # Extract base clusters
        base_clusters = extract_base_clusters(db, run_id, min_clusters=10)

        # Convert BaseCluster models to dicts for merge_clusters_hierarchical
        base_clusters_dicts = [cluster.model_dump() for cluster in base_clusters]

        # Run hierarchical merging
        hierarchy_run_id, hierarchy_data = await merge_clusters_hierarchical(
            base_clusters=base_clusters_dicts,
            run_id=run_id,
            embedding_model=config.embedding_model,
            embedding_provider=config.embedding_provider,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            target_levels=config.hierarchy_levels,
            merge_ratio=config.merge_ratio,
            neighborhood_size=config.neighborhood_size,
            concurrency=config.concurrency,
            rpm=config.rpm
        )
        
        elapsed = time.time() - start_time
        
        if not hierarchy_run_id:
            logger.error("Hierarchy creation failed - no hierarchy run ID returned")
            raise RuntimeError("Hierarchy creation failed to return a run ID")
        
        logger.info(f"✓ Hierarchy creation complete in {elapsed:.2f}s")
        logger.info(f"  Hierarchy Run ID: {hierarchy_run_id}")
        logger.info(f"  Created {len(hierarchy_data)} total hierarchy nodes")
        
        # Log hierarchy structure
        levels = {}
        for node in hierarchy_data:
            level = node["level"]
            if level not in levels:
                levels[level] = 0
            levels[level] += 1
        
        for level in sorted(levels.keys()):
            logger.info(f"  Level {level}: {levels[level]} clusters")
        
        return hierarchy_run_id
        
    except Exception as e:
        logger.error(f"Hierarchy creation failed after {time.time() - start_time:.2f}s: {e}")
        raise


def cleanup_resources(temp_dir: Path, db: Database, chroma: ChromaManager, should_cleanup: bool) -> None:
    """Clean up temporary resources with comprehensive logging.

    Note: Database and ChromaManager don't require explicit cleanup as they
    use connection pooling and auto-cleanup mechanisms.
    """
    logger.info("Starting resource cleanup...")

    # ChromaDB and Database don't need explicit closing
    # They use persistent clients that auto-cleanup
    logger.info("✓ Database and ChromaDB connections released")

    # Remove temporary directory
    if should_cleanup and temp_dir and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"✓ Temporary directory removed: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error removing temporary directory {temp_dir}: {e}")
    elif not should_cleanup:
        logger.info(f"Temporary directory preserved for debugging: {temp_dir}")


async def run_analysis(config: RunnerConfig) -> Dict[str, Any]:
    """Run complete analysis workflow procedurally with comprehensive logging."""
    logger.info("Starting LMSYS Query Analysis workflow")
    logger.info(f"Configuration: {config.dict()}")
    
    overall_start_time = time.time()
    
    # Setup
    temp_dir, db, chroma = setup_environment(config)
    
    try:
        # Step 1: Load data
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA LOADING")
        logger.info("=" * 60)
        load_stats = load_data(db, chroma, config)
        
        # Step 2: Run clustering
        logger.info("=" * 60)
        logger.info("PHASE 2: CLUSTERING")
        logger.info("=" * 60)
        run_id = run_clustering(db, chroma, config)
        
        # Step 3: Create hierarchy (optional)
        hierarchy_run_id = None
        if config.enable_hierarchy:
            logger.info("=" * 60)
            logger.info("PHASE 3: HIERARCHY CREATION")
            logger.info("=" * 60)
            hierarchy_run_id = await create_hierarchy(db, run_id, config)
        else:
            logger.info("Hierarchy creation skipped (disabled in configuration)")
        
        # Calculate total execution time
        total_elapsed = time.time() - overall_start_time
        
        # Build results
        results = {
            "run_id": run_id,
            "hierarchy_run_id": hierarchy_run_id,
            "total_queries": load_stats["loaded"],
            "execution_time": total_elapsed,
            "config": config.dict(),
            "stats": {
                "loading": load_stats,
                "total_time": total_elapsed
            }
        }
        
        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {total_elapsed:.2f}s")
        logger.info(f"Base run ID: {run_id}")
        if hierarchy_run_id:
            logger.info(f"Hierarchy run ID: {hierarchy_run_id}")
        logger.info(f"Queries processed: {load_stats['loaded']}")
        
        return results
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("WORKFLOW FAILED")
        logger.error("=" * 60)
        logger.error(f"Error after {time.time() - overall_start_time:.2f}s: {e}")
        raise
        
    finally:
        logger.info("=" * 60)
        logger.info("CLEANUP")
        logger.info("=" * 60)
        cleanup_resources(temp_dir, db, chroma, config.cleanup_temp)


def main():
    """Simple main entry point for direct script execution."""
    # Default configuration for quick runs
    config = RunnerConfig(
        query_limit=1000,
        n_clusters=50,
        enable_hierarchy=False,  # Skip hierarchy for faster execution
        log_level="INFO",
        db_path="quick_analysis.db"  # Persistent by default
    )
    
    console.print("[bold cyan]LMSYS Query Analysis Runner[/bold cyan]")
    console.print("Running with default configuration:")
    console.print(f"  Queries: {config.query_limit}")
    console.print(f"  Clusters: {config.n_clusters}")
    console.print(f"  Hierarchy: {config.enable_hierarchy}")
    console.print(f"  Database: {config.db_path}")
    
    try:
        # Run workflow
        results = asyncio.run(run_analysis(config))
        
        # Print final results
        console.print("\n[bold green]✓ Analysis Complete![/bold green]")
        console.print(f"[cyan]Run ID:[/cyan] {results['run_id']}")
        if results['hierarchy_run_id']:
            console.print(f"[cyan]Hierarchy ID:[/cyan] {results['hierarchy_run_id']}")
        console.print(f"[cyan]Total Queries:[/cyan] {results['total_queries']}")
        console.print(f"[cyan]Execution Time:[/cyan] {results['execution_time']:.2f}s")

        console.print(f"\n[bold yellow]Next Steps:[/bold yellow]")
        console.print(f"Explore your results with the CLI:")
        console.print(f"  uv run lmsys runs --latest")
        console.print(f"  uv run lmsys list-clusters {results['run_id']}")
        console.print(f"  uv run lmsys search 'python' --run-id {results['run_id']}")

        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]Analysis failed: {e}[/bold red]")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
