"""ChromaDB manager for semantic search and vector storage.

Supports multiple embedding models by using model-specific collection names.
Collections are named: queries_{provider}_{model} and summaries_{provider}_{model}
"""

from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
import re


DEFAULT_CHROMA_PATH = Path.home() / ".lmsys-query-analysis" / "chroma"


def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to meet ChromaDB requirements.

    Collection names must:
    - Be 3-63 characters long
    - Start and end with alphanumeric
    - Only contain alphanumeric, underscores, or hyphens
    """
    # Replace dots, slashes, and invalid chars with underscores
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Ensure starts/ends with alphanumeric
    name = name.strip('_-')
    # Truncate if too long
    if len(name) > 63:
        name = name[:63].rstrip('_-')
    # Ensure minimum length
    if len(name) < 3:
        name = name + '_default'
    return name


class ChromaManager:
    """Manages ChromaDB collections for queries and cluster summaries.

    Supports multiple embedding models by using model-specific collection names.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        embedding_dimension: int | None = None,
    ):
        """Initialize ChromaDB manager.

        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Embedding model name for collection naming
            embedding_provider: Embedding provider (openai, cohere, sentence-transformers)
        """
        if persist_directory is None:
            persist_directory = DEFAULT_CHROMA_PATH

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_dimension = embedding_dimension

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Create model-specific collection names
        # Optionally include dimension in suffix (useful for Cohere Matryoshka)
        dim_part = f"_{embedding_dimension}" if (embedding_provider == "cohere" and embedding_dimension) else ""
        model_suffix = sanitize_collection_name(f"{embedding_provider}_{embedding_model}{dim_part}")
        queries_name = f"queries_{model_suffix}"
        summaries_name = f"summaries_{model_suffix}"

        # Collection for all queries (model-specific)
        self.queries_collection = self.client.get_or_create_collection(
            name=queries_name,
            metadata={
                "description": f"User queries with {embedding_provider}/{embedding_model} embeddings",
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
            },
        )

        # Collection for cluster summaries (model-specific, filtered by run_id in metadata)
        self.summaries_collection = self.client.get_or_create_collection(
            name=summaries_name,
            metadata={
                "description": f"Cluster summaries with {embedding_provider}/{embedding_model} embeddings",
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
            },
        )

    def add_queries_batch(
        self,
        query_ids: List[int],
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[dict],
    ):
        """Add a batch of queries to ChromaDB.

        Args:
            query_ids: List of SQLite query IDs
            texts: List of query texts
            embeddings: Numpy array of embeddings
            metadata: List of metadata dicts (model, language, etc.)
        """
        # Convert IDs to ChromaDB format
        chroma_ids = [f"query_{qid}" for qid in query_ids]

        # Add SQLite ID to metadata for reference
        enriched_metadata = [
            {**meta, "query_id": qid} for meta, qid in zip(metadata, query_ids)
        ]

        self.queries_collection.add(
            ids=chroma_ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=enriched_metadata,
        )

    def add_cluster_summary(
        self,
        run_id: str,
        cluster_id: int,
        summary: str,
        embedding: np.ndarray,
        metadata: dict,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Add a cluster summary to ChromaDB.

        Args:
            run_id: Clustering run ID
            cluster_id: Cluster ID within the run
            summary: Summary text (used if title/description not provided)
            embedding: Summary embedding
            metadata: Additional metadata (num_queries, etc.)
            title: Optional LLM-generated title
            description: Optional LLM-generated description
        """
        chroma_id = f"cluster_{run_id}_{cluster_id}"

        # Use title + description if available, otherwise fall back to summary
        if title and description:
            document_text = f"{title}\n\n{description}"
            enriched_metadata = {
                **metadata,
                "run_id": run_id,
                "cluster_id": cluster_id,
                "title": title,
                "description": description,
            }
        else:
            document_text = summary
            enriched_metadata = {
                **metadata,
                "run_id": run_id,
                "cluster_id": cluster_id,
            }

        self.summaries_collection.add(
            ids=[chroma_id],
            embeddings=[embedding.tolist()],
            documents=[document_text],
            metadatas=[enriched_metadata],
        )

    def add_cluster_summaries_batch(
        self,
        run_id: str,
        cluster_ids: List[int],
        summaries: List[str],
        embeddings: np.ndarray,
        metadata_list: List[dict],
        titles: Optional[List[str]] = None,
        descriptions: Optional[List[str]] = None,
    ):
        """Add multiple cluster summaries for a run.

        Args:
            run_id: Clustering run ID
            cluster_ids: List of cluster IDs
            summaries: List of summary texts
            embeddings: Numpy array of embeddings
            metadata_list: List of metadata dicts
            titles: Optional list of LLM-generated titles
            descriptions: Optional list of LLM-generated descriptions
        """
        chroma_ids = [f"cluster_{run_id}_{cid}" for cid in cluster_ids]

        # Use titles + descriptions if available
        if titles and descriptions:
            documents = [
                f"{title}\n\n{desc}" for title, desc in zip(titles, descriptions)
            ]
            enriched_metadata = [
                {
                    **meta,
                    "run_id": run_id,
                    "cluster_id": int(cid),
                    "title": title,
                    "description": desc,
                }
                for meta, cid, title, desc in zip(
                    metadata_list, cluster_ids, titles, descriptions
                )
            ]
        else:
            documents = summaries
            enriched_metadata = [
                {**meta, "run_id": run_id, "cluster_id": int(cid)}
                for meta, cid in zip(metadata_list, cluster_ids)
            ]

        self.summaries_collection.add(
            ids=chroma_ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=enriched_metadata,
        )

    def search_queries(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[dict] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """Semantic search across all queries.

        Args:
            query_text: Search query
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"model": "gpt-4"})
            query_embedding: Optional precomputed embedding for the query

        Returns:
            Dictionary with ids, documents, distances, and metadatas
        """
        if query_embedding is not None:
            results = self.queries_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
            )
        else:
            results = self.queries_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
            )
        return results

    def search_cluster_summaries(
        self,
        query_text: str,
        run_id: Optional[str] = None,
        n_results: int = 5,
        query_embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """Search cluster summaries, optionally filtered by run_id.

        Args:
            query_text: Search query
            run_id: Optional run_id to filter by specific clustering run
            n_results: Number of results
            query_embedding: Optional precomputed embedding for the query

        Returns:
            Dictionary with ids, documents, distances, and metadatas
        """
        where = {"run_id": run_id} if run_id else None
        if query_embedding is not None:
            results = self.summaries_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
            )
        else:
            results = self.summaries_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
            )
        return results

    def get_queries_by_ids(self, query_ids: List[int]) -> dict:
        """Get queries by SQLite IDs.

        Args:
            query_ids: List of SQLite query IDs

        Returns:
            Dictionary with documents and metadatas
        """
        chroma_ids = [f"query_{qid}" for qid in query_ids]
        return self.queries_collection.get(
            ids=chroma_ids, include=["documents", "metadatas"]
        )

    def get_query_embeddings_map(self, query_ids: List[int]) -> dict[int, np.ndarray]:
        """Get a mapping from SQLite query ID -> embedding vector.

        Args:
            query_ids: List of SQLite query IDs

        Returns:
            Dict mapping query_id to numpy embedding; missing IDs omitted
        """
        chroma_ids = [f"query_{qid}" for qid in query_ids]
        results = self.queries_collection.get(ids=chroma_ids, include=["embeddings"])

        id_to_embedding: dict[int, np.ndarray] = {}
        if results and results.get("ids"):
            for cid, emb in zip(results.get("ids", []), results.get("embeddings", [])):
                if cid and emb is not None and len(emb) > 0:
                    # Extract integer query_id from "query_{id}"
                    try:
                        qid = (
                            int(str(cid).split("_")[1]) if "_" in str(cid) else int(cid)
                        )
                        id_to_embedding[qid] = np.array(emb, dtype=float)
                    except Exception:
                        continue
        return id_to_embedding

    def get_cluster_summary(self, run_id: str, cluster_id: int) -> dict:
        """Get a specific cluster summary.

        Args:
            run_id: Clustering run ID
            cluster_id: Cluster ID

        Returns:
            Dictionary with document and metadata
        """
        chroma_id = f"cluster_{run_id}_{cluster_id}"
        return self.summaries_collection.get(ids=[chroma_id])

    def list_runs_in_summaries(self) -> List[str]:
        """Get all unique run_ids stored in cluster summaries.

        Returns:
            List of run_ids
        """
        # Get all summaries and extract unique run_ids
        all_summaries = self.summaries_collection.get()
        run_ids = set()
        if all_summaries and all_summaries["metadatas"]:
            for meta in all_summaries["metadatas"]:
                if "run_id" in meta:
                    run_ids.add(meta["run_id"])
        return sorted(run_ids)

    def count_queries(self) -> int:
        """Count total queries in ChromaDB."""
        return self.queries_collection.count()

    def count_summaries(self, run_id: Optional[str] = None) -> int:
        """Count cluster summaries, optionally filtered by run_id."""
        if run_id:
            results = self.summaries_collection.get(where={"run_id": run_id})
            return len(results["ids"]) if results["ids"] else 0
        return self.summaries_collection.count()

    def list_all_collections(self) -> List[dict]:
        """List all collections in ChromaDB with their metadata.

        Returns:
            List of dicts with collection name and metadata
        """
        collections = self.client.list_collections()
        return [
            {
                "name": col.name,
                "metadata": col.metadata,
                "count": col.count(),
            }
            for col in collections
        ]

    def get_collection_info(self) -> dict:
        """Get info about the current model-specific collections.

        Returns:
            Dict with queries and summaries collection info
        """
        return {
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "queries": {
                "name": self.queries_collection.name,
                "count": self.queries_collection.count(),
            },
            "summaries": {
                "name": self.summaries_collection.name,
                "count": self.summaries_collection.count(),
            },
        }


def get_chroma(
    persist_directory: str | Path | None = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_provider: str = "openai",
    embedding_dimension: int | None = None,
) -> ChromaManager:
    """Get ChromaDB manager instance.

    Args:
        persist_directory: Directory to persist ChromaDB data
        embedding_model: Embedding model name for collection naming
        embedding_provider: Embedding provider (openai, cohere, sentence-transformers)
    """
    return ChromaManager(persist_directory, embedding_model, embedding_provider, embedding_dimension)
