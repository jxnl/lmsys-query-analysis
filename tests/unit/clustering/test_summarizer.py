"""Smoke tests for the summarizer.

These tests mock out LLM calls and embeddings so we can verify prompt assembly
and output shape without network access.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lmsys_query_analysis.clustering.summarizer import ClusterData, ClusterSummarizer


class FakeAsyncChat:
    def __init__(self, prompts: list[str]):
        self._prompts = prompts

    class _Completions:
        def __init__(self, prompts_ref: list[str]):
            self._prompts = prompts_ref

        async def create(self, response_model=None, messages=None, context=None):
            content = messages[0]["content"] if messages and len(messages) > 0 else ""
            self._prompts.append(content)
            return SimpleNamespace(
                title="Test Title",
                description="Test Description",
            )

    @property
    def completions(self):
        return self._Completions(self._prompts)


class FakeAsyncClient:
    def __init__(self, prompts: list[str]):
        self.chat = SimpleNamespace(completions=FakeAsyncChat(prompts).completions)


def patch_instructor(monkeypatch, prompts: list[str]):
    """Patch instructor.from_provider to avoid real client init and capture prompts."""
    import instructor

    class _FakeSyncChat:
        class _Completions:
            def __init__(self, prompts_ref: list[str]):
                self._prompts = prompts_ref

            def create(self, response_model=None, messages=None):
                content = messages[1]["content"] if messages and len(messages) > 1 else ""
                self._prompts.append(content)
                return SimpleNamespace(title="Test Title", description="Test Description")

        def __init__(self, prompts_ref: list[str]):
            self.completions = self._Completions(prompts_ref)

    class _FakeSyncClient:
        def __init__(self, prompts_ref: list[str]):
            self.chat = _FakeSyncChat(prompts_ref)

    def _fake_from_provider(model: str, api_key=None, async_client: bool = False, **kwargs):
        if async_client:
            return FakeAsyncClient(prompts)
        return _FakeSyncClient(prompts)

    monkeypatch.setattr(instructor, "from_provider", _fake_from_provider, raising=True)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Patch EmbeddingGenerator.generate_embeddings to a deterministic matrix."""

    def _fake_generate_embeddings(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = True
    ):
        n = len(texts)
        if n == 3:
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.99, 0.01, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        eye = np.eye(max(1, n))
        return eye[:n]

    import lmsys_query_analysis.clustering.embeddings as emb

    monkeypatch.setattr(
        emb.EmbeddingGenerator,
        "generate_embeddings",
        _fake_generate_embeddings,
        raising=True,
    )


def test_summarizer_prompt_includes_contrast_neighbors(fake_embeddings, monkeypatch):
    prompts: list[str] = []
    patch_instructor(monkeypatch, prompts)
    s = ClusterSummarizer(model="openai/gpt-5", concurrency=1)

    clusters_data = [
        ClusterData(
            cluster_id=0, queries=["python pandas dataframe indexing", "numpy vectorize loop"]
        ),
        ClusterData(
            cluster_id=1, queries=["pandas loc keyerror fix", "python typeerror add int str"]
        ),
        ClusterData(cluster_id=2, queries=["how to cook pasta", "boil water add salt"]),
    ]

    res = s.generate_batch_summaries(
        clusters_data,
        max_queries=5,
        concurrency=1,
        rpm=None,
        contrast_neighbors=2,
        contrast_examples=1,
        contrast_mode="neighbors",
    )

    assert set(res.keys()) == {0, 1, 2}
    for v in res.values():
        assert "title" in v and "description" in v and "sample_queries" in v

        assert len(prompts) == 3
        p0 = prompts[0]
        assert "<contrastive_examples>" in p0
    assert "<positive_examples>" in p0


def test_summarizer_prompt_keywords_mode(fake_embeddings, monkeypatch):
    prompts: list[str] = []
    patch_instructor(monkeypatch, prompts)
    s = ClusterSummarizer(model="openai/gpt-5", concurrency=1)

    clusters_data = [
        ClusterData(cluster_id=10, queries=["nginx reverse proxy 502", "docker image too big"]),
        ClusterData(cluster_id=11, queries=["kubernetes secret mount", "crashloopbackoff logs"]),
        ClusterData(cluster_id=12, queries=["translate text to spanish", "preserve html tags"]),
    ]

    res = s.generate_batch_summaries(
        clusters_data,
        max_queries=5,
        concurrency=1,
        rpm=None,
        contrast_neighbors=1,
        contrast_examples=0,
        contrast_mode="keywords",
    )

    assert set(res.keys()) == {10, 11, 12}
    assert len(prompts) == 3
    assert any("<contrastive_examples>" in p for p in prompts)


def test_summarizer_prompt_no_contrast(fake_embeddings, monkeypatch):
    prompts: list[str] = []
    patch_instructor(monkeypatch, prompts)
    s = ClusterSummarizer(model="openai/gpt-5", concurrency=1)

    clusters_data = [
        ClusterData(cluster_id=100, queries=["hello", "hi there"]),
        ClusterData(cluster_id=101, queries=["ciao", "hola"]),
    ]

    res = s.generate_batch_summaries(
        clusters_data,
        max_queries=5,
        concurrency=1,
        rpm=None,
        contrast_neighbors=0,
        contrast_examples=0,
        contrast_mode="neighbors",
    )

    assert set(res.keys()) == {100, 101}
    assert all("<contrastive_neighbors>" not in p for p in prompts)
