"""Factory for creating database and ChromaDB clients."""

from typing import Optional, Tuple
from ...db.connection import Database, get_db
from ...db.chroma import ChromaManager, get_chroma
from ...clustering.embeddings import EmbeddingGenerator
from ...semantic import QueriesClient, ClustersClient


def parse_embedding_model(model_string: str) -> Tuple[str, str]:
    """Parse embedding model string into model name and provider.
    
    Args:
        model_string: Format "provider/model" e.g. "cohere/embed-v4.0"
    
    Returns:
        Tuple of (model_name, provider)
    
    Raises:
        ValueError: If model_string format is invalid
    """
    parts = model_string.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid embedding model format: {model_string}. Use 'provider/model'"
        )
    provider, model = parts[0], parts[1]
    return model, provider


def get_embedding_dimension(provider: str) -> Optional[int]:
    """Get default embedding dimension for provider.
    
    Args:
        provider: Embedding provider name (e.g., "cohere", "openai")
    
    Returns:
        Embedding dimension or None for default
    """
    return 256 if provider == "cohere" else None


def create_chroma_client(
    chroma_path: Optional[str],
    embedding_model: str,
    embedding_provider: str,
) -> ChromaManager:
    """Create ChromaDB client with proper dimension settings.
    
    Args:
        chroma_path: Path to ChromaDB directory
        embedding_model: Model name
        embedding_provider: Provider name
    
    Returns:
        Configured ChromaManager instance
    """
    dimension = get_embedding_dimension(embedding_provider)
    return get_chroma(chroma_path, embedding_model, embedding_provider, dimension)


def create_embedding_generator(
    embedding_model: str,
    embedding_provider: str,
) -> EmbeddingGenerator:
    """Create embedding generator with proper dimension settings.
    
    Args:
        embedding_model: Model name
        embedding_provider: Provider name
    
    Returns:
        Configured EmbeddingGenerator instance
    """
    dimension = get_embedding_dimension(embedding_provider)
    return EmbeddingGenerator(
        model_name=embedding_model,
        provider=embedding_provider,
        output_dimension=dimension,
    )


def create_queries_client(
    db: Database,
    run_id: Optional[str] = None,
    embedding_model_string: str = "openai/text-embedding-3-small",
    chroma_path: Optional[str] = None,
) -> QueriesClient:
    """Create QueriesClient with proper configuration.
    
    If run_id provided, loads from run configuration.
    Otherwise uses provided embedding model.
    
    Args:
        db: Database instance
        run_id: Optional run ID to load configuration from
        embedding_model_string: Embedding model in "provider/model" format
        chroma_path: Path to ChromaDB directory
    
    Returns:
        Configured QueriesClient instance
    """
    if run_id:
        return QueriesClient.from_run(db, run_id, persist_dir=chroma_path)
    
    model, provider = parse_embedding_model(embedding_model_string)
    chroma = create_chroma_client(chroma_path, model, provider)
    embedder = create_embedding_generator(model, provider)
    return QueriesClient(db, chroma, embedder)


def create_clusters_client(
    db: Database,
    run_id: Optional[str] = None,
    embedding_model_string: str = "openai/text-embedding-3-small",
    chroma_path: Optional[str] = None,
) -> ClustersClient:
    """Create ClustersClient with proper configuration.
    
    If run_id provided, loads from run configuration.
    Otherwise uses provided embedding model.
    
    Args:
        db: Database instance
        run_id: Optional run ID to load configuration from
        embedding_model_string: Embedding model in "provider/model" format
        chroma_path: Path to ChromaDB directory
    
    Returns:
        Configured ClustersClient instance
    """
    if run_id:
        return ClustersClient.from_run(db, run_id, persist_dir=chroma_path)
    
    model, provider = parse_embedding_model(embedding_model_string)
    chroma = create_chroma_client(chroma_path, model, provider)
    embedder = create_embedding_generator(model, provider)
    return ClustersClient(db, chroma, embedder)

