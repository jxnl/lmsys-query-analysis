import numpy as np

from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun, Query, QueryCluster
from lmsys_query_analysis.semantic import ClustersClient, QueriesClient


class FakeChroma:
    def __init__(self):
        self.embedding_model = "text-embedding-3-small"
        self.embedding_provider = "openai"
        self.embedding_dimension = 1536

    def search_queries(self, query_text, n_results=10, where=None, query_embedding=None):
        ids = [
            "query_1",
            "query_2",
            "query_3",
            "query_4",
            "query_5",
            "query_999",
        ]
        documents = [
            "hybrid search with bm25 and vectors",
            "how to tune hnsw ef parameter",
            "bonjour le monde",
            "vector db schema design",
            "modelo espanol",
            "noise",
        ]
        metadatas = [
            {"model": "vicuna", "language": "en"},
            {"model": "llama", "language": "en"},
            {"model": "vicuna", "language": "fr"},
            {"model": "mistral", "language": "en"},
            {"model": "vicuna", "language": "es"},
            {"model": "x", "language": "xx"},
        ]
        distances = [0.02, 0.04, 0.20, 0.06, 0.22, 0.99]
        k = min(n_results, len(ids))
        return {
            "ids": [ids[:k]],
            "documents": [documents[:k]],
            "metadatas": [metadatas[:k]],
            "distances": [distances[:k]],
        }

    def search_cluster_summaries(self, query_text, run_id=None, n_results=5, query_embedding=None):
        metas = [
            {
                "run_id": "runA",
                "cluster_id": 1,
                "title": "Vector Databases",
                "alias": "v1",
                "summary_run_id": "s1",
                "num_queries": 2,
            },
            {
                "run_id": "runA",
                "cluster_id": 3,
                "title": "Greetings FR",
                "alias": "v1",
                "summary_run_id": "s1",
                "num_queries": 1,
            },
            {
                "run_id": "runB",
                "cluster_id": 7,
                "title": "Other",
                "alias": "v1",
                "summary_run_id": "s1",
                "num_queries": 1,
            },
        ]
        docs = [
            "Vector Databases\n\nTopical cluster",
            "Greetings FR\n\nFrench greetings",
            "Other\n\nMisc",
        ]
        dists = [0.01, 0.05, 0.30]
        ids = ["cluster_runA_1", "cluster_runA_3", "cluster_runB_7"]
        if run_id:
            keep = [i for i, m in enumerate(metas) if m["run_id"] == run_id]
        else:
            keep = list(range(len(metas)))
        keep = keep[:n_results]
        return {
            "ids": [[ids[i] for i in keep]],
            "documents": [[docs[i] for i in keep]],
            "metadatas": [[metas[i] for i in keep]],
            "distances": [[dists[i] for i in keep]],
        }

    def count_summaries(self, run_id=None):
        return 2 if run_id == "runA" else 3


class FakeEmbedder:
    def generate_embeddings(self, texts, batch_size=1, show_progress=False):
        arr = np.zeros((len(texts), 4), dtype=float)
        return arr


def seed_db(tmp_path) -> Database:
    db_path = tmp_path / "test.sqlite"
    db = Database(db_path)
    db.create_tables()
    with db.get_session() as s:
        s.add(
            ClusteringRun(
                run_id="runA",
                algorithm="kmeans-minibatch",
                parameters={
                    "embedding_provider": "openai",
                    "embedding_model": "text-embedding-3-small",
                },
                num_clusters=10,
            )
        )
        s.add(
            ClusteringRun(
                run_id="runB",
                algorithm="kmeans-minibatch",
                parameters={
                    "embedding_provider": "openai",
                    "embedding_model": "text-embedding-3-small",
                },
                num_clusters=10,
            )
        )
        q1 = Query(
            conversation_id="c1",
            model="vicuna",
            query_text="hybrid search with bm25 and vectors",
            language="en",
        )
        q2 = Query(
            conversation_id="c2",
            model="llama",
            query_text="how to tune hnsw ef parameter",
            language="en",
        )
        q3 = Query(
            conversation_id="c3", model="vicuna", query_text="bonjour le monde", language="fr"
        )
        q4 = Query(
            conversation_id="c4",
            model="mistral",
            query_text="vector db schema design",
            language="en",
        )
        q5 = Query(conversation_id="c5", model="vicuna", query_text="modelo espanol", language="es")
        s.add(q1)
        s.add(q2)
        s.add(q3)
        s.add(q4)
        s.add(q5)
        s.commit()

        s.add(QueryCluster(run_id="runA", query_id=q1.id, cluster_id=1))
        s.add(QueryCluster(run_id="runA", query_id=q2.id, cluster_id=1))
        s.add(QueryCluster(run_id="runA", query_id=q3.id, cluster_id=3))
        s.add(QueryCluster(run_id="runA", query_id=q4.id, cluster_id=1))
        s.add(QueryCluster(run_id="runB", query_id=q5.id, cluster_id=7))
        s.commit()
    return db


def test_clusters_find_filters_by_run_and_alias(tmp_path):
    db = seed_db(tmp_path)
    chroma = FakeChroma()
    embed = FakeEmbedder()

    client = ClustersClient(db, chroma, embed, run_id="runA")
    hits = client.find(text="vector", run_id="runA", top_k=5)

    assert all(h.cluster_id in {1, 3} for h in hits)
    assert hits[0].distance <= hits[-1].distance

    hits2 = client.find(text="vector", run_id="runA", top_k=5, alias="v1")
    assert {h.cluster_id for h in hits2} == {1, 3}


def test_queries_find_within_clusters_and_run(tmp_path):
    db = seed_db(tmp_path)
    chroma = FakeChroma()
    embed = FakeEmbedder()

    qclient = QueriesClient(db, chroma, embed, run_id="runA")

    hits = qclient.find(
        text="hybrid",
        run_id="runA",
        within_clusters="vector databases",
        top_clusters=2,
        n_results=10,
        n_candidates=20,
    )

    ids = {h.query_id for h in hits}
    assert ids.issubset({1, 2, 3, 4})
    dists = [h.distance for h in hits]
    assert dists == sorted(dists)


def test_queries_count_and_facets(tmp_path):
    db = seed_db(tmp_path)
    chroma = FakeChroma()
    embed = FakeEmbedder()
    qclient = QueriesClient(db, chroma, embed, run_id="runA")

    total = qclient.count(
        text="hybrid",
        run_id="runA",
        within_clusters="vector",
        top_clusters=2,
    )
    assert isinstance(total, int)
    assert total >= 1

    by_cluster = qclient.count(
        text="hybrid",
        run_id="runA",
        within_clusters="vector",
        top_clusters=2,
        by="cluster",
    )
    assert isinstance(by_cluster, dict)
    assert all(isinstance(k, int) for k in by_cluster.keys())

    facets = qclient.facets(
        text="hybrid",
        run_id="runA",
        within_clusters="vector",
        top_clusters=2,
        facet_by=["cluster", "language", "model"],
    )
    assert set(facets.keys()) == {"cluster", "language", "model"}
    assert len(facets["cluster"]) >= 1
    assert len(facets["language"]) >= 1


def test_facets_cluster_includes_title_when_available(tmp_path):
    db = seed_db(tmp_path)
    from lmsys_query_analysis.db.models import ClusterSummary

    with db.get_session() as s:
        s.add(
            ClusterSummary(run_id="runA", cluster_id=1, title="Vector Databases", description="...")
        )
        s.add(
            ClusterSummary(run_id="runA", cluster_id=3, title="French Greetings", description="...")
        )
        s.commit()

    chroma = FakeChroma()
    embed = FakeEmbedder()
    qclient = QueriesClient(db, chroma, embed, run_id="runA")

    facets = qclient.facets(
        text="hybrid",
        run_id="runA",
        within_clusters="vector",
        top_clusters=2,
        facet_by=["cluster"],
    )
    clusters = facets["cluster"]
    meta_titles = {b.key: b.meta.get("title") for b in clusters if b.meta}
    assert 1 in meta_titles and meta_titles[1] == "Vector Databases"


def test_from_run_resolves_vector_space(tmp_path):
    db_path = tmp_path / "db.sqlite"
    db = Database(db_path)
    db.create_tables()
    with db.get_session() as s:
        s.add(
            ClusteringRun(
                run_id="runC",
                algorithm="kmeans-minibatch",
                parameters={
                    "embedding_provider": "sentence-transformers",
                    "embedding_model": "all-MiniLM-L6-v2",
                },
                num_clusters=5,
            )
        )
        s.commit()

    cc = ClustersClient.from_run(db, "runC", persist_dir=tmp_path / "chroma")
    space = cc.resolve_space()
    assert space.embedding_provider == "sentence-transformers"
    assert space.embedding_model == "all-MiniLM-L6-v2"
    assert space.embedding_dimension is None
    assert space.run_id == "runC"


def test_clusters_count_without_text_uses_chroma_count(tmp_path):
    db = seed_db(tmp_path)
    chroma = FakeChroma()
    embed = FakeEmbedder()

    client = ClustersClient(db, chroma, embed, run_id="runA")
    count = client.count(run_id="runA", text=None)
    assert count == 2


def test_clusters_find_filters_by_alias_and_summary_run_id(tmp_path):
    db = seed_db(tmp_path)
    chroma = FakeChroma()
    embed = FakeEmbedder()

    client = ClustersClient(db, chroma, embed, run_id="runA")
    hits_alias = client.find(text="vector", run_id="runA", alias="v1", top_k=10)
    assert {h.cluster_id for h in hits_alias} == {1, 3}

    hits_bad = client.find(text="vector", run_id="runA", summary_run_id="s999", top_k=10)
    assert hits_bad == []


def test_queries_find_with_explicit_cluster_ids(tmp_path):
    db = seed_db(tmp_path)
    chroma = FakeChroma()
    embed = FakeEmbedder()
    qclient = QueriesClient(db, chroma, embed, run_id="runA")

    hits = qclient.find(
        text="any",
        run_id="runA",
        cluster_ids=[3],
        n_results=10,
        n_candidates=20,
    )
    ids = {h.query_id for h in hits}
    assert ids == {3}
