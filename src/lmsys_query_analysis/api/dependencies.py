"""FastAPI dependency injection for database and configuration."""

import os
from collections.abc import Generator
from pathlib import Path

from ..db.chroma import ChromaManager
from ..db.connection import DEFAULT_DB_PATH, Database

# Read configuration from environment variables (same as CLI)
DB_PATH = os.getenv("DB_PATH") or str(DEFAULT_DB_PATH)
CHROMA_PATH = os.getenv("CHROMA_PATH") or str(Path.home() / ".lmsys-query-analysis" / "chroma")

# API keys are read from environment (server-side only)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def get_db() -> Generator[Database, None, None]:
    """Dependency injection for database connection.

    Yields a Database instance that shares the same configuration as the CLI.
    The database connection is automatically managed by FastAPI's dependency system.
    """
    db = Database(db_path=DB_PATH, auto_create_tables=True)
    try:
        yield db
    finally:
        # Database cleanup happens automatically via context manager
        pass


def get_chroma_path() -> str:
    """Get the ChromaDB persistence path.

    Returns the configured ChromaDB path from environment or default.
    """
    return CHROMA_PATH


def create_chroma_manager(
    run_id: str,
    db: Database,
    chroma_path: str | None = None,
) -> ChromaManager:
    """Create a ChromaManager configured for a specific clustering run.

    This resolves the embedding provider/model/dimension from the run's parameters
    stored in the database, ensuring that semantic search uses the correct vector space.

    Args:
        run_id: Clustering run ID to resolve embedding configuration from
        db: Database instance to query run parameters
        chroma_path: Optional override for ChromaDB path (defaults to CHROMA_PATH)

    Returns:
        ChromaManager configured with the run's embedding settings

    Raises:
        ValueError: If run_id is not found in database
    """
    from sqlmodel import select

    from ..db.models import ClusteringRun

    with db.get_session() as session:
        run = session.exec(select(ClusteringRun).where(ClusteringRun.run_id == run_id)).first()

        if not run:
            raise ValueError(f"Run not found: {run_id}")

        params = run.parameters or {}
        embedding_model = params.get("embedding_model", "text-embedding-3-small")
        embedding_provider = params.get("embedding_provider", "openai")
        embedding_dimension = params.get("embedding_dimension")

        # Cohere Matryoshka default to 256 if not specified
        if embedding_provider == "cohere" and embedding_dimension is None:
            embedding_dimension = 256

    return ChromaManager(
        persist_directory=chroma_path or CHROMA_PATH,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        embedding_dimension=embedding_dimension,
    )
