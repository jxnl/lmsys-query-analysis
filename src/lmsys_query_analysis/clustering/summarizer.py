"""LLM-based cluster summarization using instructor.

Adds concurrent summarization using anyio + AsyncInstructor with
optional rate limiting to speed up large runs safely.
"""

from typing import List, Optional, Dict, Tuple
import logging

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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

console = Console()

# Constants
MAX_NEIGHBOR_EXAMPLE_LENGTH = 180
RETRY_MAX_ATTEMPTS = 5
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 8


class ClusterData(BaseModel):
    """Simple data object for cluster information."""
    cluster_id: int
    queries: List[str]
    
    def __iter__(self):
        """Allow tuple unpacking for backward compatibility."""
        return iter((self.cluster_id, self.queries))
    
    @classmethod
    def from_tuple(cls, data: Tuple[int, List[str]]) -> "ClusterData":
        """Create from tuple for easy migration."""
        return cls(cluster_id=data[0], queries=data[1])
    
    def to_tuple(self) -> Tuple[int, List[str]]:
        """Convert to tuple for backward compatibility."""
        return (self.cluster_id, self.queries)


# Concise cluster prompt
DEFAULT_CLUSTER_PROMPT = """
Summarize these queries into a 2-sentence description and a short name (≤10 words, imperative sentence).

Format:
<title>Short imperative name like "Help with Python debugging" or "Write creative stories"</title>
<description>Two sentences describing what users were trying to accomplish</description>

Be specific and descriptive. Include harmful/sensitive topics explicitly when relevant.

Positive examples:
<positive_examples>
{% for item in positive_examples %}{{ item }}
{% endfor %}
</positive_examples>

Contrastive examples:
<contrastive_examples>
{% for item in contrastive_examples %}{{ item }}
{% endfor %}
</contrastive_examples>
"""



class ClusterSummaryResponse(BaseModel):
    """Structured response from LLM for cluster summarization."""

    title: str = Field(
        ...,
        description="Short imperative name (≤10 words) like 'Help with Python debugging' or 'Write creative stories'. Be specific, avoid vague terms like 'Various' or 'General'."
    )
    description: str = Field(
        ...,
        description="Two sentences in past tense describing what users were trying to accomplish and key patterns. Focus on dominant behavior (60%+ of queries)."
    )


class ClusterSummarizer:
    """Generate titles and descriptions for query clusters using LLMs via instructor.

    Note: This class no longer requires embedding models. Embeddings are only needed
    for ChromaDB updates, which are handled separately in the CLI layer.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        concurrency: int = 40,
        rpm: Optional[int] = None,
    ):
        """Initialize the summarizer.

        Args:
            model: Model in format "provider/model_name" (e.g.,
                   "openai/gpt-4o-mini", "openai/gpt-4o",
                   "anthropic/claude-sonnet-4-5-20250929", "groq/llama-3.1-8b-instant")
            api_key: API key for the provider (or set env var ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
            concurrency: Number of concurrent LLM requests
            rpm: Optional requests-per-minute rate limit (global across tasks)
        """
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.rpm = rpm if (rpm is None or rpm > 0) else None
        self.logger = logging.getLogger("lmsys")


        # Initialize async instructor client with full model string
        if api_key:
            self.async_client = instructor.from_provider(
                model, api_key=api_key, async_client=True
            )
        else:
            self.async_client = instructor.from_provider(model, async_client=True)

    def generate_batch_summaries(
        self,
        clusters_data: List[ClusterData],
        max_queries: int = 100,
        concurrency: Optional[int] = None,
        rpm: Optional[int] = None,
        contrast_neighbors: int = 5,
        contrast_examples: int = 10,
        contrast_mode: str = "neighbors",
    ) -> dict[int, dict]:
        """Generate summaries for multiple clusters (concurrently).

        Args:
            clusters_data: List of ClusterData objects
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
        clusters_data: List[ClusterData],
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
        self.logger.info(
            "Starting async batch summarization: total=%d, max_queries=%d, concurrency=%d, rpm=%s",
            total,
            max_queries,
            concurrency,
            rpm,
        )

        # Pre-select representative queries for each cluster
        sampled_map: Dict[int, List[str]] = {}
        for cluster_data in clusters_data:
            sampled_map[cluster_data.cluster_id] = self._select_representative_queries(cluster_data.queries, max_queries)

        # Build neighbor context for contrastive learning
        neighbor_context = self._build_neighbor_context(
            clusters_data,
            sampled_map,
            contrast_neighbors,
            contrast_examples,
            contrast_mode,
        )

        async def worker(
            idx: int, cluster_data: ClusterData, progress_task, progress
        ):
            # Build per-cluster summary (retries handled by tenacity decorator)
            self.logger.debug(
                "Worker %d starting for cluster %d (%d queries)", idx, cluster_data.cluster_id, len(cluster_data.queries)
            )
            async with semaphore:
                summary = await self._async_generate_cluster_summary(
                    cluster_queries=cluster_data.queries,
                    cluster_id=cluster_data.cluster_id,
                    max_queries=max_queries,
                    contrast_neighbors=neighbor_context.get(cluster_data.cluster_id, []),
                )
            results[cluster_data.cluster_id] = summary
            self.logger.info("Completed summary for cluster %d: %s", cluster_data.cluster_id, summary["title"])

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
                for j, cluster_data in enumerate(clusters_data):
                    tg.start_soon(worker, j, cluster_data, task, progress)

        return results

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(Exception),
    )
    async def _async_generate_cluster_summary(
        self,
        cluster_queries: List[str],
        cluster_id: int,
        max_queries: int = 50,
        contrast_neighbors: Optional[List[Dict]] = None,
    ) -> dict:
        """Async variant of single-cluster summarization using AsyncInstructor.

        Automatically retries up to 5 times with exponential backoff on errors.

        Args:
            cluster_queries: All queries in the cluster
            cluster_id: Cluster ID
            max_queries: Max queries to send to LLM
            contrast_neighbors: List of neighbor cluster data for contrastive learning
        """
        # Select a representative and diverse sample of queries
        self.logger.debug(
            "Selecting representative queries for cluster %d from %d total",
            cluster_id,
            len(cluster_queries),
        )
        sampled = self._select_representative_queries(cluster_queries, max_queries)
        self.logger.debug(
            "Selected %d representative queries for cluster %d", len(sampled), cluster_id
        )

        # Build contrastive examples from neighbors
        contrastive_examples = []
        if contrast_neighbors:
            for neighbor in contrast_neighbors:
                if neighbor.get("examples"):
                    contrastive_examples.extend(neighbor["examples"])

        messages = [
            {
                "role": "system",
                "content": DEFAULT_CLUSTER_PROMPT,
            },
        ]

        # Use Instructor's built-in Jinja templating with context
        response = await self.async_client.chat.completions.create(
            response_model=ClusterSummaryResponse,
            messages=messages,
            context={
                "positive_examples": sampled,
                "contrastive_examples": contrastive_examples,
            },
        )

        return {
            "title": response.title,
            "description": response.description,
            "sample_queries": sampled,
        }

    # -------- Helper methods --------

    def _build_neighbor_context(
        self,
        clusters_data: List[ClusterData],
        sampled_map: Dict[int, List[str]],
        contrast_neighbors: int,
        contrast_examples: int,
        contrast_mode: str,
    ) -> Dict[int, List[Dict]]:
        """Build neighbor context for contrastive learning with improved selection."""
        id_list = [cluster_data.cluster_id for cluster_data in clusters_data]
        sizes_map = {cluster_data.cluster_id: len(cluster_data.queries) for cluster_data in clusters_data}

        # Skip if no neighbors requested or only one cluster
        if contrast_neighbors <= 0 or len(id_list) <= 1:
            return {cid: [] for cid in id_list}

        neighbor_context = {}
        for i, cid in enumerate(id_list):
            other_ids = [other_id for j, other_id in enumerate(id_list) if j != i]
            if not other_ids:
                neighbor_context[cid] = []
                continue

            # Improved neighbor selection: prioritize clusters with different sizes
            # to get better contrastive examples
            other_clusters_with_size = [(oid, sizes_map.get(oid, 0)) for oid in other_ids]
            other_clusters_with_size.sort(key=lambda x: abs(x[1] - sizes_map.get(cid, 0)), reverse=True)
            
            n_neighbors = min(contrast_neighbors, len(other_ids))
            neighbor_ids = [oid for oid, _ in other_clusters_with_size[:n_neighbors]]

            neighbors_data = []
            for nid in neighbor_ids:
                # Get examples if contrast_examples > 0
                examples = []
                if contrast_examples > 0:
                    examples = sampled_map.get(nid, [])[:contrast_examples]
                    examples = [
                        e.splitlines()[0].strip()[:MAX_NEIGHBOR_EXAMPLE_LENGTH]
                        for e in examples
                    ]

                neighbors_data.append(
                    {
                        "cluster_id": nid,
                        "size": sizes_map.get(nid, 0),
                        "mode": contrast_mode,
                        "examples": examples,
                        "keywords": "",
                    }
                )

            neighbor_context[cid] = neighbors_data

        return neighbor_context

    def _random_sample(self, items: List[str], k: int) -> List[str]:
        """Sample k items using random sampling."""
        import random

        if len(items) <= k:
            return items

        return random.sample(items, k)

    def _select_representative_queries(
        self, cluster_queries: List[str], max_queries: int
    ) -> List[str]:
        """Select a representative subset of queries using random sampling.

        - Deduplicate queries (case-insensitive, trimmed)
        - Use random sampling for simplicity

        Args:
            cluster_queries: List of query texts
            max_queries: Maximum number to select
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

        # Use random sampling
        return self._random_sample(originals, k)
