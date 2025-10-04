"""LLM-based cluster summarization using instructor.

Adds concurrent summarization using anyio + AsyncInstructor with
optional rate limiting to speed up large runs safely.
"""

from typing import List, Optional, Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import anyio
import instructor
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from jinja2 import Environment
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

console = Console()


def _create_template():
    """Create Jinja2 template."""
    env = Environment()
    return env.from_string("""
<summary_task>
  <cluster id="{{ cluster_id }}">
    <stats total_queries="{{ total_queries }}" sample_count="{{ sample_count }}" />
    <target_queries>
{% for query in queries -%}
      <query idx="{{ loop.index }}">{{ query }}</query>
{% endfor -%}
    </target_queries>
  </cluster>
{% if contrast_neighbors -%}
  <contrastive_neighbors>
{%- for neighbor in contrast_neighbors %}
    <neighbor cluster_id="{{ neighbor.cluster_id }}" size="{{ neighbor.size }}">
{%- if neighbor.mode == "neighbors" %}
{%- for example in neighbor.examples %}
      <example>{{ example }}</example>
{%- endfor %}
{%- elif neighbor.mode == "keywords" %}
      <keywords>{{ neighbor.keywords }}</keywords>
{%- endif %}
    </neighbor>
{%- endfor %}
  </contrastive_neighbors>
{% endif -%}
</summary_task>
""")


class ClusterSummaryResponse(BaseModel):
    """Structured response from LLM for cluster summarization."""

    title: str = Field(
        ..., description="Short title (5-10 words) capturing the main topic of the cluster"
    )
    description: str = Field(
        ..., description="2-3 sentences explaining the cluster theme and what the queries are about"
    )


class ClusterSummarizer:
    """Generate titles and descriptions for query clusters using LLMs via instructor."""

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        concurrency: int = 4,
        rpm: Optional[int] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
    ):
        """Initialize the summarizer.

        Args:
            model: Model in format "provider/model_name" (e.g.,
                   "anthropic/claude-sonnet-4-5-20250929", "openai/gpt-4o",
                   "groq/llama-3.1-8b-instant")
            api_key: API key for the provider (or set env var ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
            concurrency: Number of concurrent LLM requests
            rpm: Optional requests-per-minute rate limit (global across tasks)
            embedding_model: Embedding model for query selection (should match clustering model)
            embedding_provider: Provider for embeddings (sentence-transformers, openai, etc.)
        """
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.rpm = rpm if (rpm is None or rpm > 0) else None
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider

        # Initialize Jinja2 template
        self.template = _create_template()

        # Initialize rate limiter if rpm is specified
        # Convert rpm to requests per second with 1-second time period
        self.rate_limiter = AsyncLimiter(rpm, 60.0) if rpm else None

        # Initialize instructor client with full model string
        if api_key:
            self.client = instructor.from_provider(model, api_key=api_key)
            self.async_client = instructor.from_provider(
                model, api_key=api_key, async_client=True
            )
        else:
            self.client = instructor.from_provider(model)
            self.async_client = instructor.from_provider(model, async_client=True)

    def generate_cluster_summary(
        self,
        cluster_queries: List[str],
        cluster_id: int,
        max_queries: int = 50,
        contrast_neighbors: Optional[List[dict]] = None,
    ) -> dict:
        """Generate title and description for a cluster.

        Args:
            cluster_queries: All query texts in the cluster
            cluster_id: Cluster ID for reference
            max_queries: Maximum queries to include in prompt (sample if more)
            contrast_neighbors: Optional list of neighbor cluster data for contrast

        Returns:
            Dict with keys: title, description, sample_queries
        """
        # Select a representative and diverse sample of queries
        sampled = self._select_representative_queries(cluster_queries, max_queries)

        # Render prompt using Jinja2 template
        prompt = self.template.render(
            cluster_id=cluster_id,
            total_queries=len(cluster_queries),
            sample_count=len(sampled),
            queries=sampled,
            contrast_neighbors=contrast_neighbors or [],
        )

        # Call LLM with structured output - instructor handles model internally
        response = self.client.chat.completions.create(
            response_model=ClusterSummaryResponse,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert data analyst specializing in query pattern analysis. Your task is to analyze clusters of user queries submitted to LLM systems and provide insightful summaries.

When analyzing a cluster:
1. Identify the core theme or use case that unites the queries
2. Note any patterns in how users phrase requests (technical vs casual, specific vs general)
3. Highlight key characteristics that make this cluster distinct
4. If contrastive neighbors are provided, emphasize what makes THIS cluster unique compared to similar clusters

Provide:
- **Title**: A clear, specific 5-10 word description of the cluster's main theme
- **Description**: 2-3 sentences explaining:
  - What users in this cluster are trying to accomplish
  - Common patterns or notable characteristics
  - What makes this cluster distinct (especially vs neighbors if provided)""",
                },
                {"role": "user", "content": prompt},
            ],
        )

        return {
            "title": response.title,
            "description": response.description,
            "sample_queries": sampled[:10],  # Store top 10 for reference
        }

    def generate_batch_summaries(
        self,
        clusters_data: List[tuple[int, List[str]]],
        max_queries: int = 50,
        concurrency: Optional[int] = None,
        rpm: Optional[int] = None,
        contrast_neighbors: int = 2,
        contrast_examples: int = 2,
        contrast_mode: str = "neighbors",
    ) -> dict[int, dict]:
        """Generate summaries for multiple clusters (concurrently).

        Args:
            clusters_data: List of (cluster_id, queries) tuples
            max_queries: Max queries per cluster to send to LLM
            concurrency: Override default concurrency
            rpm: Override default requests-per-minute limit

        Returns:
            Dict mapping cluster_id to summary dict
        """
        use_concurrency = (
            self.concurrency if concurrency is None else max(1, int(concurrency))
        )
        use_rpm = self.rpm if rpm is None else (rpm if rpm and rpm > 0 else None)

        return anyio.run(
            self._async_generate_batch_summaries,
            clusters_data,
            max_queries,
            use_concurrency,
            use_rpm,
            contrast_neighbors,
            contrast_examples,
            contrast_mode,
        )

    # -------- Async path below --------

    async def _async_generate_batch_summaries(
        self,
        clusters_data: List[Tuple[int, List[str]]],
        max_queries: int,
        concurrency: int,
        rpm: Optional[int],
        contrast_neighbors: int,
        contrast_examples: int,
        contrast_mode: str,
    ) -> Dict[int, dict]:
        """Async concurrent generation using anyio with semaphore-based rate limiting."""

        total = len(clusters_data)
        results: Dict[int, dict] = {}
        semaphore = anyio.Semaphore(concurrency)

        console.print(
            f"[cyan]Generating LLM summaries for {total} clusters using {self.model} (concurrency={concurrency}{', rpm=' + str(rpm) if rpm else ''})...[/cyan]"
        )

        # Pre-select representative queries for each cluster
        sampled_map: Dict[int, List[str]] = {}
        for cid, qtexts in clusters_data:
            sampled_map[cid] = self._select_representative_queries(qtexts, max_queries)

        # Prepare neighbor context per cluster by randomly selecting other clusters
        id_list = [cid for cid, _ in clusters_data]
        sizes_map: Dict[int, int] = {cid: len(qs) for cid, qs in clusters_data}
        neighbor_context: Dict[int, str] = {}

        if contrast_neighbors > 0 and contrast_examples > 0 and len(id_list) > 1:
            import random

            for i, cid in enumerate(id_list):
                # Randomly select n other clusters (excluding self)
                other_ids = [other_id for j, other_id in enumerate(id_list) if j != i]
                if not other_ids:
                    neighbor_context[cid] = ""
                    continue

                n_neighbors = min(contrast_neighbors, len(other_ids))
                neighbor_ids = random.sample(other_ids, n_neighbors)

                block_lines = ["<contrastive_neighbors>"]
                for nid in neighbor_ids:
                    nsize = sizes_map.get(nid, 0)
                    exs = sampled_map.get(nid, [])[:contrast_examples]
                    exs_fmt = [e.splitlines()[0].strip()[:180] for e in exs]

                    block_lines.append(
                        f'  <neighbor cluster_id="{nid}" size="{nsize}">'
                    )
                    for ex in exs_fmt:
                        block_lines.append(f"    <example><![CDATA[{ex}]]></example>")
                    block_lines.append("  </neighbor>")

                block_lines.append("</contrastive_neighbors>")
                neighbor_context[cid] = "\n".join(block_lines)
        else:
            for cid in id_list:
                neighbor_context[cid] = ""

        async def worker(
            idx: int, cid: int, queries: List[str], progress_task, progress
        ):
            # Build per-cluster summary (retries handled by tenacity decorator)
            async with semaphore:
                summary = await self._async_generate_cluster_summary(
                    cluster_queries=queries,
                    cluster_id=cid,
                    max_queries=max_queries,
                    contrast_block=neighbor_context.get(cid, ""),
                )
            results[cid] = summary

            if progress_task is not None:
                progress.update(progress_task, advance=1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Summarizing clusters with {self.model}...",
                total=total,
            )
            async with anyio.create_task_group() as tg:
                for j, (cid, qtexts) in enumerate(clusters_data):
                    tg.start_soon(worker, j, cid, qtexts, task, progress)

        return results

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def _async_generate_cluster_summary(
        self,
        cluster_queries: List[str],
        cluster_id: int,
        max_queries: int = 50,
        contrast_block: str = "",
    ) -> dict:
        """Async variant of single-cluster summarization using AsyncInstructor.

        Automatically retries up to 5 times with exponential backoff on errors.

        Args:
            cluster_queries: All queries in the cluster
            cluster_id: Cluster ID
            max_queries: Max queries to send to LLM
            contrast_block: Pre-rendered contrast block XML string
        """
        # Select a representative and diverse sample of queries
        sampled = self._select_representative_queries(cluster_queries, max_queries)

        # Build prompt - just the core cluster data since contrast_block is pre-rendered
        core_prompt = f"""<summary_task>
  <cluster id="{cluster_id}">
    <stats total_queries="{len(cluster_queries)}" sample_count="{len(sampled)}" />
    <target_queries>
"""
        for idx, query in enumerate(sampled, 1):
            core_prompt += f'      <query idx="{idx}">{query}</query>\n'
        core_prompt += "    </target_queries>\n  </cluster>\n"

        if contrast_block:
            core_prompt += contrast_block + "\n"

        core_prompt += "</summary_task>"

        prompt = core_prompt

        messages = [
            {
                "role": "system",
                "content": """You are an expert data analyst specializing in query pattern analysis. Your task is to analyze clusters of user queries submitted to LLM systems and provide insightful summaries.

When analyzing a cluster:
1. Identify the core theme or use case that unites the queries
2. Note any patterns in how users phrase requests (technical vs casual, specific vs general)
3. Highlight key characteristics that make this cluster distinct
4. If contrastive neighbors are provided, emphasize what makes THIS cluster unique compared to similar clusters

Provide:
- **Title**: A clear, specific 5-10 word description of the cluster's main theme
- **Description**: 2-3 sentences explaining:
  - What users in this cluster are trying to accomplish
  - Common patterns or notable characteristics
  - What makes this cluster distinct (especially vs neighbors if provided)""",
            },
            {"role": "user", "content": prompt},
        ]

        # Apply rate limiting if configured
        if self.rate_limiter:
            async with self.rate_limiter:
                response = await self.async_client.chat.completions.create(
                    response_model=ClusterSummaryResponse,
                    messages=messages,
                )
        else:
            response = await self.async_client.chat.completions.create(
                response_model=ClusterSummaryResponse,
                messages=messages,
            )

        return {
            "title": response.title,
            "description": response.description,
            "sample_queries": sampled[:10],
        }

    # -------- Helper methods --------
    def _select_representative_queries(
        self, cluster_queries: List[str], max_queries: int
    ) -> List[str]:
        """Select a diverse, representative subset of queries using embeddings + MMR.

        - Deduplicate queries (case-insensitive, trimmed)
        - Compute embeddings for a capped subset of queries
        - Select k via centroid relevance + MMR for diversity
        """
        if not cluster_queries:
            return []

        # Normalize and deduplicate while preserving original text
        seen = set()
        originals: List[str] = []
        for q in cluster_queries:
            q = (q or "").strip()
            key = " ".join(q.lower().split())
            if key and key not in seen:
                seen.add(key)
                originals.append(q)

        k = min(max_queries, len(originals))
        if k <= 0:
            return []
        if k >= len(originals):
            return originals

        # Cap number of items to embed for efficiency
        cap = max(5 * k, 300)
        if len(originals) > cap:
            originals = originals[:cap]

        # Embed using configured embedding model (matches clustering model)
        try:
            from .embeddings import EmbeddingGenerator

            eg = EmbeddingGenerator(
                model_name=self.embedding_model,
                provider=self.embedding_provider,
                concurrency=self.concurrency,
            )
            E = eg.generate_embeddings(originals, show_progress=False)
        except Exception:
            # As a last resort, fall back to a simple first-k selection
            return originals[:k]

        # Compute centroid relevance
        centroid = E.mean(axis=0, keepdims=True)
        rel = cosine_similarity(E, centroid).ravel()

        # MMR selection in embedding space
        lambda_param = 0.7
        selected: List[int] = []
        candidates = set(range(E.shape[0]))

        first = int(np.argmax(rel))
        selected.append(first)
        candidates.remove(first)

        while len(selected) < k and candidates:
            sims_to_selected = cosine_similarity(E[list(candidates)], E[selected]).max(
                axis=1
            )
            candidate_list = list(candidates)
            scores = (
                lambda_param * rel[candidate_list]
                - (1 - lambda_param) * sims_to_selected
            )
            best_idx = int(np.argmax(scores))
            chosen = candidate_list[best_idx]
            selected.append(chosen)
            candidates.remove(chosen)

        selected.sort()
        return [originals[i] for i in selected]
