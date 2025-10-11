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


# Default cluster prompt with improved positive/contrastive structure
DEFAULT_CLUSTER_PROMPT = """
You are tasked with summarizing a group of related statements into a short, precise, and accurate description and name. Your goal is to create a concise summary that captures the essence of these statements and distinguishes them from other similar groups of statements.

Summarize all the statements into a clear, precise, two-sentence description in the past tense. Your summary should be specific to this group and distinguish it from the contrastive answers of the other groups.

After creating the summary, generate a short name for the group of statements. This name should be at most ten words long (perhaps less) and be specific but also reflective of most of the statements (rather than reflecting only one or two).

The name should distinguish this group from the contrastive examples. For instance, "Write fantasy sexual roleplay with octopi and monsters", "Generate blog spam for gambling websites", or "Assist with high school math homework" would be better and more actionable than general terms like "Write erotic content" or "Help with homework". Be as descriptive as possible and assume neither good nor bad faith. Do not hesitate to identify and describe socially harmful or sensitive topics specifically; specificity is necessary for monitoring.

Present your output in the following format:
<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here] </name>

The names you propose must follow these requirements:
- The cluster name should be a sentence in the imperative that captures the user's request. For example, 'Brainstorm ideas for a birthday party' or 'Help me find a new job.'
- Create names that are specific enough to be meaningful, but not so specific that they can't meaningfully represent many different statements.
- Avoid overly general or vague terms, and do not hesitate to describe socially harmful or sensitive topics (in fact, clusters that clearly describe harmful behavior are slightly preferred); specificity is necessary for observability and enforcement.
- Ensure that the cluster name is distinct from the contrastive examples.
- Use clear, concise, and descriptive language for the cluster name.

Below are the related statements:
<positive_examples>
{% for item in positive_examples %}{{ item }}
{% endfor %}
</positive_examples>

For context, here are statements from nearby groups that are NOT part of the group you're summarizing:
<contrastive_examples>
{% for item in contrastive_examples %}{{ item }}
{% endfor %}
</contrastive_examples>

Do not elaborate beyond what you say in the tags. Remember to analyze both the statements and the contrastive statements carefully to ensure your summary and name accurately represent the specific group while distinguishing it from others. The cluster name should be a sentence in the imperative that captures the user's request. For example, 'Brainstorm ideas for a birthday party' or 'Help me find a new job.'
"""



class ClusterSummaryResponse(BaseModel):
    """Structured response from LLM for cluster summarization."""

    title: str = Field(
        ...,
        description="""Short name (≤10 words) that is a sentence in the imperative capturing the user's request.

        RULES:
        - Use specific technical terms, domains, or actions (e.g., "Brainstorm ideas for a birthday party", "Help me find a new job")
        - Avoid generic prefixes like "User Queries About", "Diverse", "Various"
        - Avoid vague words: "Diverse", "Various", "General", "Mixed", "Multiple"
        - Use concrete nouns and verbs
        - If multilingual, specify language(s)
        - Be descriptive and specific, including harmful/sensitive topics when relevant

        GOOD: "Write fantasy sexual roleplay with octopi and monsters", "Generate blog spam for gambling websites", "Assist with high school math homework"
        BAD: "User Queries on Various Topics", "Diverse User Requests", "General Questions"
        """
    )
    description: str = Field(
        ...,
        description="""Two-sentence summary in past tense that captures the essence of the cluster and distinguishes it from contrastive examples.

        Structure:
        1. What users were trying to accomplish (main goal/task) and their mental model
        2. Key behavioral patterns, user segments, or product implications

        Focus on the DOMINANT pattern (60%+ of queries). Think like an anthropologist + product manager:
        - What do these queries reveal about how people use LLMs?
        - What capabilities do users assume exist?
        - Are there unmet needs or product opportunities?

        Example: "Users sought complete coding solutions with minimal specification, treating the LLM as a 'magic wand' code generator. Novices pasted homework verbatim while experts provided constraints, revealing distinct mental models across expertise levels - opportunity for adaptive interfaces."
        """
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
        clusters_data: List[tuple[int, List[str]]],
        max_queries: int = 100,
        concurrency: Optional[int] = None,
        rpm: Optional[int] = None,
        contrast_neighbors: int = 5,
        contrast_examples: int = 10,
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
        self.logger.info(
            "Starting async batch summarization: total=%d, max_queries=%d, concurrency=%d, rpm=%s",
            total,
            max_queries,
            concurrency,
            rpm,
        )

        # Pre-select representative queries for each cluster
        sampled_map: Dict[int, List[str]] = {}
        for cid, qtexts in clusters_data:
            sampled_map[cid] = self._select_representative_queries(qtexts, max_queries)

        # Build neighbor context for contrastive learning
        neighbor_context = self._build_neighbor_context(
            clusters_data,
            sampled_map,
            contrast_neighbors,
            contrast_examples,
            contrast_mode,
        )

        async def worker(
            idx: int, cid: int, queries: List[str], progress_task, progress
        ):
            # Build per-cluster summary (retries handled by tenacity decorator)
            self.logger.debug(
                "Worker %d starting for cluster %d (%d queries)", idx, cid, len(queries)
            )
            async with semaphore:
                summary = await self._async_generate_cluster_summary(
                    cluster_queries=queries,
                    cluster_id=cid,
                    max_queries=max_queries,
                    contrast_neighbors=neighbor_context.get(cid, []),
                )
            results[cid] = summary
            self.logger.info("Completed summary for cluster %d: %s", cid, summary["title"])

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
        clusters_data: List[Tuple[int, List[str]]],
        sampled_map: Dict[int, List[str]],
        contrast_neighbors: int,
        contrast_examples: int,
        contrast_mode: str,
    ) -> Dict[int, List[Dict]]:
        """Build neighbor context for contrastive learning with improved selection."""
        id_list = [cid for cid, _ in clusters_data]
        sizes_map = {cid: len(qs) for cid, qs in clusters_data}

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
