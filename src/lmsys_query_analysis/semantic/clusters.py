from __future__ import annotations

from pathlib import Path

from ..clustering.embeddings import EmbeddingGenerator
from ..db.chroma import ChromaManager
from ..db.connection import Database
from .types import ClusterHit, RunSpace


class ClustersClient:
    """Cluster (summary) search API.

    Provides run-aware search over cluster summaries stored in Chroma, with
    enrichment from SQLite as needed.

    Note: This is a stub/skeleton. Methods raise NotImplementedError.
    """

    def __init__(
        self,
        db: Database,
        chroma: ChromaManager,
        embedder: EmbeddingGenerator,
        run_id: str | None = None,
    ):
        """Initialize with explicit dependencies.

        Args:
            db: Database connection manager
            chroma: Chroma manager bound to a specific vector space
            embedder: Embedding generator aligned to the same vector space
        """
        self.db = db
        self.chroma = chroma
        self.embedder = embedder
        self._run_id = run_id

    @classmethod
    def from_run(
        cls, db: Database, run_id: str, persist_dir: str | Path | None = None
    ) -> ClustersClient:
        """Construct a client by resolving vector space from a clustering run.

        Loads provider/model/dimension from the run parameters and configures
        Chroma and embeddings consistently.
        """
        from sqlmodel import select

        from ..db.models import ClusteringRun

        with db.get_session() as s:
            run = s.exec(select(ClusteringRun).where(ClusteringRun.run_id == run_id)).first()
            if not run:
                raise ValueError(f"Run not found: {run_id}")
            params = run.parameters or {}
            embedding_model = params.get("embedding_model", "text-embedding-3-small")
            embedding_provider = params.get("embedding_provider", "openai")
            embedding_dimension = params.get("embedding_dimension")
            if embedding_provider == "cohere" and embedding_dimension is None:
                embedding_dimension = 256

        chroma = ChromaManager(
            persist_directory=persist_dir or None,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_dimension=embedding_dimension,
        )
        embedder = EmbeddingGenerator(
            model_name=embedding_model,
            provider=embedding_provider,
            output_dimension=embedding_dimension if embedding_provider == "cohere" else None,
        )
        return cls(db=db, chroma=chroma, embedder=embedder, run_id=run_id)

    def find(
        self,
        text: str,
        run_id: str | None = None,
        top_k: int = 20,
        alias: str | None = None,
        summary_run_id: str | None = None,
    ) -> list[ClusterHit]:
        """Search cluster summaries and return ranked clusters.

        Args:
            text: Search text
            run_id: Optional run filter for provenance
            top_k: Number of cluster hits to return
            alias: Optional summary alias filter
            summary_run_id: Optional summary run filter
        """
        effective_run = run_id or self._run_id

        vec = self.embedder.generate_embeddings([text], batch_size=1, show_progress=False)[0]

        results = self.chroma.search_cluster_summaries(
            query_text=text,
            run_id=effective_run,
            n_results=top_k,
            query_embedding=vec,
        )

        hits: list[ClusterHit] = []
        if results and results.get("ids"):
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for cid, _doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
                if alias and meta.get("alias") != alias:
                    continue
                if summary_run_id and meta.get("summary_run_id") != summary_run_id:
                    continue
                cluster_id = (
                    int(meta.get("cluster_id"))
                    if meta and meta.get("cluster_id") is not None
                    else None
                )
                if cluster_id is None:
                    try:
                        if isinstance(cid, str) and "_" in cid:
                            cluster_id = int(cid.split("_")[-1])
                    except Exception:
                        continue
                title = meta.get("title") if meta else None
                description = meta.get("description") if meta else None
                num_queries = meta.get("num_queries") if meta else None
                hits.append(
                    ClusterHit(
                        cluster_id=cluster_id,
                        distance=float(dist),
                        title=title,
                        description=description,
                        num_queries=int(num_queries) if num_queries is not None else None,
                    )
                )

        return hits[:top_k]

    def count(
        self,
        run_id: str | None = None,
        text: str | None = None,
        alias: str | None = None,
        summary_run_id: str | None = None,
    ) -> int:
        """Return count of matching clusters (after filters)."""
        top_k = 1000
        if text is None:
            _results = self.chroma.search_cluster_summaries(
                query_text="*",
                run_id=run_id or self._run_id,
                n_results=1,
                query_embedding=None,
            )
            total = self.chroma.count_summaries(run_id or self._run_id)
            return int(total)
        return len(
            self.find(
                text=text,
                run_id=run_id,
                top_k=top_k,
                alias=alias,
                summary_run_id=summary_run_id,
            )
        )

    def resolve_space(self) -> RunSpace:
        """Return the active run/vector-space configuration."""
        return RunSpace(
            embedding_provider=self.chroma.embedding_provider,
            embedding_model=self.chroma.embedding_model,
            embedding_dimension=self.chroma.embedding_dimension,
            run_id=self._run_id,
        )
