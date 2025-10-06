from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..db.connection import Database
from ..db.chroma import ChromaManager
from ..clustering.embeddings import EmbeddingGenerator
from .types import FacetBucket, QueryHit, RunSpace


class QueriesClient:
    """Query search API with clustering context.

    Supports run-aware search over queries with optional conditioning on
    semantically selected clusters (two-stage within-clusters flow).

    Note: This is a stub/skeleton. Methods raise NotImplementedError.
    """

    def __init__(self, db: Database, chroma: ChromaManager, embedder: EmbeddingGenerator, run_id: str | None = None):
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
    ) -> "QueriesClient":
        """Construct a client by resolving vector space from a clustering run."""
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
        run_id: Optional[str] = None,
        cluster_ids: Optional[List[int]] = None,
        within_clusters: Optional[str] = None,
        top_clusters: int = 10,
        n_results: int = 50,
        n_candidates: int = 250,
        threshold: Optional[float] = None,
    ) -> List[QueryHit]:
        """Return ranked query hits filtered and conditioned as requested."""
        effective_run = run_id or self._run_id

        # If within_clusters specified, select top clusters via summary search
        selected_cluster_ids: Optional[List[int]] = None
        if within_clusters:
            from .clusters import ClustersClient

            clusters_client = ClustersClient(self.db, self.chroma, self.embedder, run_id=effective_run)
            chits = clusters_client.find(
                text=within_clusters,
                run_id=effective_run,
                top_k=top_clusters,
            )
            selected_cluster_ids = [h.cluster_id for h in chits]

        if cluster_ids:
            selected_cluster_ids = list(cluster_ids)

        # Embed search text and query Chroma for candidates
        vec = self.embedder.generate_embeddings([text], batch_size=1, show_progress=False)[0]
        results = self.chroma.search_queries(
            query_text=text,
            n_results=n_candidates,
            where=None,
            query_embedding=vec,
        )

        ids = results.get("ids", [[]])[0] if results else []
        docs = results.get("documents", [[]])[0] if results else []
        metas = results.get("metadatas", [[]])[0] if results else []
        dists = results.get("distances", [[]])[0] if results else []

        # Extract numeric query IDs
        cand_ids: List[int] = []
        for cid in ids:
            try:
                if isinstance(cid, str) and "_" in cid:
                    cand_ids.append(int(cid.split("_")[1]))
                else:
                    cand_ids.append(int(cid))
            except Exception:
                continue

        # If run/cluster filters are requested, filter using SQLite
        keep_idx: List[int] = list(range(len(cand_ids)))
        cluster_map: Dict[int, int] = {}
        if effective_run or selected_cluster_ids is not None:
            from sqlmodel import select
            from ..db.models import QueryCluster

            with self.db.get_session() as s:
                stmt = select(QueryCluster.query_id, QueryCluster.cluster_id).where(
                    QueryCluster.query_id.in_(cand_ids)
                )
                if effective_run:
                    stmt = stmt.where(QueryCluster.run_id == effective_run)
                rows = s.exec(stmt).all()
                cluster_map = {int(qid): int(cid) for (qid, cid) in rows}

            # Build mask of candidates that satisfy run/cluster constraints
            keep_idx = []
            for i, qid in enumerate(cand_ids):
                cid = cluster_map.get(qid)
                if cid is None:
                    continue
                if selected_cluster_ids is not None and cid not in selected_cluster_ids:
                    continue
                keep_idx.append(i)

        # Assemble hits with filtering and optional threshold
        hits: List[QueryHit] = []
        for i in keep_idx:
            dist = float(dists[i])
            if threshold is not None and dist > threshold:
                continue
            meta = metas[i] or {}
            qid = cand_ids[i]
            snippet = docs[i]
            hits.append(
                QueryHit(
                    query_id=qid,
                    distance=dist,
                    snippet=snippet,
                    model=meta.get("model"),
                    language=meta.get("language"),
                    cluster_id=cluster_map.get(qid),
                )
            )

        # Re-rank by distance after filtering, then truncate
        hits.sort(key=lambda h: h.distance)
        return hits[: n_results]

    def count(
        self,
        text: str,
        run_id: Optional[str] = None,
        cluster_ids: Optional[List[int]] = None,
        within_clusters: Optional[str] = None,
        top_clusters: int = 10,
        by: Optional[str] = None,
    ) -> int | Dict:
        """Return total count or grouped counts (by=cluster|language|model)."""
        effective_run = run_id or self._run_id

        # Reuse candidate selection from find() but collect IDs only
        hits = self.find(
            text=text,
            run_id=effective_run,
            cluster_ids=cluster_ids,
            within_clusters=within_clusters,
            top_clusters=top_clusters,
            n_results=10_000,  # large upper bound for counting small datasets
            n_candidates=20_000,
        )
        if by is None:
            return len(hits)

        # Grouped counts
        if by not in {"cluster", "language", "model"}:
            raise ValueError("by must be one of: cluster, language, model")

        if by == "cluster":
            counts: Dict[int, int] = {}
            for h in hits:
                cid = h.cluster_id if h.cluster_id is not None else -1
                counts[cid] = counts.get(cid, 0) + 1
            return counts

        if by in {"language", "model"}:
            counts2: Dict[str, int] = {}
            for h in hits:
                key = getattr(h, by) or ""
                counts2[key] = counts2.get(key, 0) + 1
            return counts2

        return len(hits)

    def facets(
        self,
        text: str,
        run_id: Optional[str] = None,
        cluster_ids: Optional[List[int]] = None,
        within_clusters: Optional[str] = None,
        top_clusters: int = 10,
        facet_by: List[str] = ["cluster"],
    ) -> Dict[str, List[FacetBucket]]:
        """Return facet buckets for the filtered/conditioned result set."""
        effective_run = run_id or self._run_id
        hits = self.find(
            text=text,
            run_id=effective_run,
            cluster_ids=cluster_ids,
            within_clusters=within_clusters,
            top_clusters=top_clusters,
            n_results=10_000,
            n_candidates=20_000,
        )

        facets: Dict[str, List[FacetBucket]] = {}
        for facet in facet_by:
            if facet == "cluster":
                agg: Dict[int, int] = {}
                for h in hits:
                    cid = h.cluster_id if h.cluster_id is not None else -1
                    agg[cid] = agg.get(cid, 0) + 1
                # Optionally enrich with cluster titles when run is known
                meta_map: Dict[int, Dict[str, str]] = {}
                effective_run = run_id or self._run_id
                if effective_run is not None and agg:
                    try:
                        from sqlmodel import select
                        from ..db.models import ClusterSummary
                        with self.db.get_session() as s:
                            # Pick the most recent summary per cluster if multiple exist
                            # Fetch all summaries for relevant clusters
                            rows = s.exec(
                                select(ClusterSummary.cluster_id, ClusterSummary.title, ClusterSummary.generated_at)
                                .where(ClusterSummary.run_id == effective_run)
                                .where(ClusterSummary.cluster_id.in_(list(agg.keys())))
                            ).all()
                            # Keep latest by generated_at
                            tmp: Dict[int, tuple[str | None, any]] = {}
                            for cid, title, gen_at in rows:
                                prev = tmp.get(cid)
                                if prev is None or (gen_at and prev[1] and gen_at > prev[1]):
                                    tmp[cid] = (title, gen_at)
                            for cid, (title, _dt) in tmp.items():
                                if title:
                                    meta_map[cid] = {"title": title}
                    except Exception:
                        pass

                buckets = [
                    FacetBucket(key=k, count=v, meta=meta_map.get(k, {})) for k, v in agg.items()
                ]
                facets["cluster"] = sorted(buckets, key=lambda b: b.count, reverse=True)
            elif facet == "language":
                agg2: Dict[str, int] = {}
                for h in hits:
                    key = h.language or ""
                    agg2[key] = agg2.get(key, 0) + 1
                buckets = [FacetBucket(key=k, count=v, meta={}) for k, v in agg2.items()]
                facets["language"] = sorted(buckets, key=lambda b: b.count, reverse=True)
            elif facet == "model":
                agg3: Dict[str, int] = {}
                for h in hits:
                    key = h.model or ""
                    agg3[key] = agg3.get(key, 0) + 1
                buckets = [FacetBucket(key=k, count=v, meta={}) for k, v in agg3.items()]
                facets["model"] = sorted(buckets, key=lambda b: b.count, reverse=True)
            else:
                raise ValueError("Unsupported facet; use cluster, language, or model")

        return facets

    # --- Utilities ---
    def resolve_space(self) -> RunSpace:
        """Return the active run/vector-space configuration."""
        return RunSpace(
            embedding_provider=self.chroma.embedding_provider,
            embedding_model=self.chroma.embedding_model,
            embedding_dimension=self.chroma.embedding_dimension,
            run_id=self._run_id,
        )

    def embed(self, text: str) -> List[float]:
        """Embed a single text in the configured space and return the vector."""
        vec = self.embedder.generate_embeddings([text], batch_size=1, show_progress=False)[0]
        return list(vec)
