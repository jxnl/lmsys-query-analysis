"""ChromaDB manager for semantic search and vector storage.

Supports multiple embedding models by using model-specific collection names.
Collections are named: queries_{provider}_{model} and summaries_{provider}_{model}
"""

import re
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

DEFAULT_CHROMA_PATH = Path.home() / ".lmsys-query-analysis" / "chroma"


def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to meet ChromaDB requirements.

    Collection names must:
    - Be 3-63 characters long
    - Start and end with alphanumeric
    - Only contain alphanumeric, underscores, or hyphens
    """
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_-")
    if len(name) > 63:
        name = name[:63].rstrip("_-")
    if len(name) < 3:
        name = name + "_default"
    return name


class ChromaManager:
    """Manages ChromaDB collections for queries and cluster summaries.

    Supports multiple embedding models by using model-specific collection names.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        embedding_model: str = "embed-v4.0",
        embedding_provider: str = "cohere",
        embedding_dimension: int | None = 256,
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

        dim_part = (
            f"_{embedding_dimension}"
            if (embedding_provider == "cohere" and embedding_dimension)
            else ""
        )
        model_suffix = sanitize_collection_name(f"{embedding_provider}_{embedding_model}{dim_part}")
        queries_name = f"queries_{model_suffix}"
        summaries_name = f"summaries_{model_suffix}"

        q_meta = {
            "description": f"User queries with {embedding_provider}/{embedding_model} embeddings",
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
        }
        if embedding_dimension is not None:
            q_meta["embedding_dimension"] = embedding_dimension
        self.queries_collection = self.client.get_or_create_collection(
            name=queries_name,
            metadata=q_meta,
        )

        s_meta = {
            "description": f"Cluster summaries with {embedding_provider}/{embedding_model} embeddings",
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
        }
        if embedding_dimension is not None:
            s_meta["embedding_dimension"] = embedding_dimension
        self.summaries_collection = self.client.get_or_create_collection(
            name=summaries_name,
            metadata=s_meta,
        )

    def add_queries_batch(
        self,
        query_ids: list[int],
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict],
    ):
        """Add a batch of queries to ChromaDB.

        Args:
            query_ids: List of SQLite query IDs
            texts: List of query texts
            embeddings: Numpy array of embeddings
            metadata: List of metadata dicts (model, language, etc.)
        """
        chroma_ids = [f"query_{qid}" for qid in query_ids]

        enriched_metadata = [
            {**meta, "query_id": qid} for meta, qid in zip(metadata, query_ids, strict=False)
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
        title: str | None = None,
        description: str | None = None,
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
        cluster_ids: list[int],
        summaries: list[str],
        embeddings: np.ndarray,
        metadata_list: list[dict],
        titles: list[str] | None = None,
        descriptions: list[str] | None = None,
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

        if titles and descriptions:
            documents = [
                f"{title}\n\n{desc}" for title, desc in zip(titles, descriptions, strict=False)
            ]
            enriched_metadata = [
                {
                    **{k: v for k, v in meta.items() if v is not None},
                    "run_id": run_id,
                    "cluster_id": int(cid),
                    "title": title,
                    "description": desc,
                }
                for meta, cid, title, desc in zip(
                    metadata_list, cluster_ids, titles, descriptions, strict=False
                )
            ]
        else:
            documents = summaries
            enriched_metadata = [
                {
                    **{k: v for k, v in meta.items() if v is not None},
                    "run_id": run_id,
                    "cluster_id": int(cid),
                }
                for meta, cid in zip(metadata_list, cluster_ids, strict=False)
            ]

        self.summaries_collection.upsert(
            ids=chroma_ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=enriched_metadata,
        )

    def search_queries(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,
        query_embedding: np.ndarray | None = None,
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
        run_id: str | None = None,
        n_results: int = 5,
        query_embedding: np.ndarray | None = None,
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

    def get_queries_by_ids(self, query_ids: list[int]) -> dict:
        """Get queries by SQLite IDs.

        Args:
            query_ids: List of SQLite query IDs

        Returns:
            Dictionary with documents and metadatas
        """
        chroma_ids = [f"query_{qid}" for qid in query_ids]
        return self.queries_collection.get(ids=chroma_ids, include=["documents", "metadatas"])

    def get_query_embeddings_map(self, query_ids: list[int]) -> dict[int, np.ndarray]:
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
            for cid, emb in zip(
                results.get("ids", []), results.get("embeddings", []), strict=False
            ):
                if cid and emb is not None and len(emb) > 0:
                    try:
                        qid = int(str(cid).split("_")[1]) if "_" in str(cid) else int(cid)
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

    def list_runs_in_summaries(self) -> list[str]:
        """Get all unique run_ids stored in cluster summaries.

        Returns:
            List of run_ids
        """
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

    def count_summaries(self, run_id: str | None = None) -> int:
        """Count cluster summaries, optionally filtered by run_id."""
        if run_id:
            results = self.summaries_collection.get(where={"run_id": run_id})
            return len(results["ids"]) if results["ids"] else 0
        return self.summaries_collection.count()

    def list_all_collections(self) -> list[dict]:
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
    embedding_model: str = "embed-v4.0",
    embedding_provider: str = "cohere",
    embedding_dimension: int | None = 256,
) -> ChromaManager:
    """Get ChromaDB manager instance.

    Args:
        persist_directory: Directory to persist ChromaDB data
        embedding_model: Embedding model name for collection naming
        embedding_provider: Embedding provider (openai, cohere, sentence-transformers)
    """
    return ChromaManager(
        persist_directory, embedding_model, embedding_provider, embedding_dimension
    )
