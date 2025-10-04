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
from jinja2 import Environment
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

console = Console()

# Constants
RATE_LIMITER_WINDOW_SECONDS = 60.0
STRATIFIED_SAMPLE_FIRST_RATIO = 1 / 3
STRATIFIED_SAMPLE_MIDDLE_RATIO = 1 / 3
MAX_NEIGHBOR_EXAMPLE_LENGTH = 180
RETRY_MAX_ATTEMPTS = 5
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 8


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
        ...,
        description="""Concise, specific title (3-7 words) that captures the PRIMARY theme.

        RULES:
        - Use specific technical terms, domains, or actions (e.g., "Python Flask API Development", "German Language Capability Checks")
        - Avoid generic prefixes like "User Queries About", "Diverse", "Various"
        - Avoid vague words: "Diverse", "Various", "General", "Mixed", "Multiple"
        - Use concrete nouns and verbs
        - If multilingual, specify language(s)

        GOOD: "Stable Diffusion Image Prompts", "SQL Query Generation", "Spanish Greetings"
        BAD: "User Queries on Various Topics", "Diverse User Requests", "General Questions"
        """
    )
    description: str = Field(
        ...,
        description="""2-3 concise sentences explaining the cluster's PRIMARY purpose and key patterns.

        Structure:
        1. What users are trying to accomplish (main goal/task)
        2. Common characteristics (technical level, phrasing style, specific subtopics)
        3. What distinguishes this cluster from similar ones (if contrastive neighbors provided)

        Focus on the DOMINANT pattern (60%+ of queries), not every variation.
        """
    )


class ClusterSummarizer:
    """Generate titles and descriptions for query clusters using LLMs via instructor.

    Note: This class no longer requires embedding models. Embeddings are only needed
    for ChromaDB updates, which are handled separately in the CLI layer.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        concurrency: int = 40,
        rpm: Optional[int] = None,
    ):
        """Initialize the summarizer.

        Args:
            model: Model in format "provider/model_name" (e.g.,
                   "anthropic/claude-sonnet-4-5-20250929", "openai/gpt-4o",
                   "groq/llama-3.1-8b-instant")
            api_key: API key for the provider (or set env var ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
            concurrency: Number of concurrent LLM requests
            rpm: Optional requests-per-minute rate limit (global across tasks)
        """
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.rpm = rpm if (rpm is None or rpm > 0) else None
        self.logger = logging.getLogger("lmsys")

        # Initialize Jinja2 template
        self.template = _create_template()

        # Initialize rate limiter if rpm is specified
        self.rate_limiter = (
            AsyncLimiter(rpm, RATE_LIMITER_WINDOW_SECONDS) if rpm else None
        )

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

        # Build prompt using Jinja template
        prompt = self.template.render(
            cluster_id=cluster_id,
            total_queries=len(cluster_queries),
            sample_count=len(sampled),
            queries=sampled,
            contrast_neighbors=contrast_neighbors or [],
        )

        messages = [
            {
                "role": "system",
                "content": """You are an expert taxonomist specializing in categorizing user interactions with LLM systems. Your goal is to create PRECISE, SPECIFIC cluster labels that enable quick understanding.

CRITICAL INSTRUCTIONS:

1. IDENTIFY THE DOMINANT PATTERN (what 60-80% of queries share)
   - Ignore outliers and edge cases
   - Focus on the PRIMARY use case or theme
   - If truly mixed, identify the 2-3 main subtypes

2. TITLE REQUIREMENTS:
   - 3-7 words maximum
   - Use SPECIFIC technical terms, domains, or action verbs
   - NEVER use: "User Queries", "Diverse", "Various", "General", "Mixed", "Multiple", "Different"
   - ALWAYS be concrete: use programming languages, specific topics, named tools/frameworks
   - Examples of GOOD titles:
     * "Python Web Scraping Code"
     * "German Language Capability Tests"
     * "SQL Database Query Generation"
     * "Stable Diffusion Art Prompts"
   - Examples of BAD titles (NEVER DO THIS):
     * "User Queries About Programming" ❌
     * "Diverse Technical Requests" ❌
     * "Various Creative Writing Tasks" ❌

3. DESCRIPTION REQUIREMENTS:
   - Sentence 1: State the PRIMARY goal/task (what users want to accomplish)
   - Sentence 2: Key patterns (technical level, common subtopics, phrasing style)
   - Sentence 3: (Optional) What distinguishes from neighbors if provided
   - Be SPECIFIC: mention actual examples, technical terms, languages, frameworks
   - Focus on DOMINANT pattern, not every variation

4. MULTILINGUAL CLUSTERS:
   - Always specify the language(s) in the title
   - Example: "Portuguese Business Writing", "Arabic General Knowledge"

Follow the Pydantic schema rules exactly.""",
            },
            {"role": "user", "content": prompt},
        ]

        # Apply rate limiting if configured
        response = await self._call_llm_with_rate_limit(
            response_model=ClusterSummaryResponse,
            messages=messages,
        )

        self.logger.info("Generated summary for cluster %d: %s", cluster_id, response.title)

        return {
            "title": response.title,
            "description": response.description,
            "sample_queries": sampled,
        }

    # -------- Helper methods --------

    async def _call_llm_with_rate_limit(self, response_model, messages):
        """Make LLM call with optional rate limiting."""
        if self.rate_limiter:
            async with self.rate_limiter:
                return await self.async_client.chat.completions.create(
                    response_model=response_model,
                    messages=messages,
                )
        return await self.async_client.chat.completions.create(
            response_model=response_model,
            messages=messages,
        )

    def _build_neighbor_context(
        self,
        clusters_data: List[Tuple[int, List[str]]],
        sampled_map: Dict[int, List[str]],
        contrast_neighbors: int,
        contrast_examples: int,
        contrast_mode: str,
    ) -> Dict[int, List[Dict]]:
        """Build neighbor context for contrastive learning."""
        import random

        id_list = [cid for cid, _ in clusters_data]
        sizes_map = {cid: len(qs) for cid, qs in clusters_data}

        # Skip if no neighbors requested or only one cluster
        # Note: contrast_examples can be 0 for keywords mode
        if contrast_neighbors <= 0 or len(id_list) <= 1:
            return {cid: [] for cid in id_list}

        neighbor_context = {}
        for i, cid in enumerate(id_list):
            other_ids = [other_id for j, other_id in enumerate(id_list) if j != i]
            if not other_ids:
                neighbor_context[cid] = []
                continue

            n_neighbors = min(contrast_neighbors, len(other_ids))
            neighbor_ids = random.sample(other_ids, n_neighbors)

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

    def _stratified_sample(self, items: List[str], k: int) -> List[str]:
        """Sample k items using stratified approach (beginning, middle, end)."""
        import random

        if len(items) <= k:
            return items

        # Calculate split points
        n_first = int(k * STRATIFIED_SAMPLE_FIRST_RATIO)
        n_middle = int(k * STRATIFIED_SAMPLE_MIDDLE_RATIO)
        n_last = k - n_first - n_middle

        # Split into thirds
        third = len(items) // 3
        first_third = items[:third]
        middle_third = items[third : 2 * third]
        last_third = items[2 * third :]

        # Sample from each section
        samples = []
        samples.extend(first_third[:n_first])
        if middle_third and n_middle > 0:
            samples.extend(random.sample(middle_third, min(n_middle, len(middle_third))))
        if last_third and n_last > 0:
            samples.extend(random.sample(last_third, min(n_last, len(last_third))))

        return samples[:k]

    def _select_representative_queries(
        self, cluster_queries: List[str], max_queries: int
    ) -> List[str]:
        """Select a diverse, representative subset of queries using stratified random sampling.

        - Deduplicate queries (case-insensitive, trimmed)
        - Use stratified sampling (beginning, middle, end) for diversity

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

        # Use stratified sampling to ensure diversity
        return self._stratified_sample(originals, k)
