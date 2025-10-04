"""LLM-based cluster summarization using instructor.

Adds concurrent summarization using anyio + AsyncInstructor with
optional rate limiting to speed up large runs safely.
"""

from typing import List, Optional, Dict, Tuple
from math import ceil
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
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

console = Console()

EXAMPLES_PROMPT = """
EXAMPLES (format and specificity you should follow):

Example A
Sample queries:
- Why does pandas .loc throw KeyError on column subset?
- Fix TypeError: unsupported operand type(s) for +: 'int' and 'str' in Python
- Vectorizing loops in NumPy for performance
Title: Python debugging and data wrangling: pandas, NumPy errors
Description: Users troubleshoot Python code with a focus on pandas indexing, NumPy array operations, and common Type/Key errors. Guidance emphasizes minimal examples, error interpretation, and idiomatic fixes.

Example B
Sample queries:
- Fetch data from a REST API with bearer token using fetch()
- Axios POST returns 401 — how to send JWT header?
- Handle 429 rate limit responses and retries in JavaScript
Title: JavaScript REST API usage and authentication
Description: Requests cover calling APIs from the browser/node, attaching JWT/Bearer auth, handling status codes (401/429), and implementing retry/backoff. Code examples use fetch and Axios patterns.

Example C
Sample queries:
- SQL: Count distinct users per day with LEFT JOIN
- GROUP BY with conditional SUM over multiple categories
- Window function to find top N products per region
Title: SQL analytics with joins, aggregates, and windows
Description: Analytical SQL tasks combining joins, GROUP BY, conditional aggregation, and window functions (RANK/ROW_NUMBER). Solutions emphasize readable CTEs and correct grouping semantics.

Example D
Sample queries:
- Fine-tune a small Llama model on custom Q&A data
- Why is my training loss NaN in PyTorch mixed precision?
- Compare F1 and accuracy for imbalanced classification
Title: ML model training and evaluation: PyTorch, metrics, fine-tuning
Description: Practical ML workflows including dataset prep, training stability (overflow/AMP), and metric selection for imbalanced tasks. Advice centers on troubleshooting training loops, schedulers, and evaluation protocols.

Example E
Sample queries:
- Docker build ignores .dockerignore; how to reduce image size?
- Nginx reverse proxy 502 when forwarding to Gunicorn
- Kubernetes CrashLoopBackOff reading config from secrets
Title: DevOps containerization and deployment: Docker, Nginx, Kubernetes
Description: Infra topics around container images, reverse proxies, and k8s deployments. Solutions cover Docker layering, proper proxy buffers/timeouts, health checks, and Secret/ConfigMap mounts.

Example F
Sample queries:
- Detect and prevent jailbreak attempts in system prompts
- Red-team prompts to bypass safety filters — show examples
- Classify prompts by risk category (self-harm, malware, PII)
Title: LLM safety and prompt hardening: jailbreak detection and taxonomy
Description: Safety and governance tasks including jailbreak detection, red-teaming, and content risk classification. Emphasis on policies, refusal guidelines, and robust prompt templates.

Example G
Sample queries:
- Create a Seaborn heatmap with annotations and custom colorbar
- Altair interactive chart: filter by dropdown, highlight selection
- Plotly Express facet grid with per-facet y-axis
Title: Data visualization recipes: Seaborn, Altair, Plotly
Description: Visualization tasks across common Python libraries, focusing on annotated heatmaps, interactive selections, and faceting. Guidance prioritizes clear legends, accessibility, and aesthetics.

Example H
Sample queries:
- Translate product descriptions to Spanish and preserve HTML tags
- Localize date/number formats for German users
- Glossary-enforced translation for brand terms
Title: Multilingual translation and localization with formatting constraints
Description: Translation requests with constraints on placeholders/HTML and locale-aware formatting. Recommendations include tag-aware translation and glossary/terminology control.
"""


class ClusterSummaryResponse(BaseModel):
    """Structured response from LLM for cluster summarization."""

    title: str = Field(
        ..., description="Short title (5-10 words) capturing the main topic"
    )
    description: str = Field(
        ..., description="2-3 sentences explaining the cluster theme"
    )


class ClusterSummarizer:
    """Generate titles and descriptions for query clusters using LLMs via instructor."""

    def __init__(
        self,
        model: str = "openai/gpt-5",
        api_key: Optional[str] = None,
        concurrency: int = 4,
        rpm: Optional[int] = None,
    ):
        """Initialize the summarizer.

        Args:
            model: Model in format "provider/model_name" (e.g., "openai/gpt-5",
                   "openai/gpt-4", "groq/llama-3.1-8b-instant")
            api_key: API key for the provider (or set env var ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
            concurrency: Number of concurrent LLM requests
            rpm: Optional requests-per-minute rate limit (global across tasks)
        """
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.rpm = rpm if (rpm is None or rpm > 0) else None

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
    ) -> dict:
        """Generate title and description for a cluster.

        Args:
            cluster_queries: All query texts in the cluster
            cluster_id: Cluster ID for reference
            max_queries: Maximum queries to include in prompt (sample if more)

        Returns:
            Dict with keys: title, description, sample_queries
        """
        # Select a representative and diverse sample of queries
        sampled = self._select_representative_queries(cluster_queries, max_queries)

        # Build prompt with clearer, more specific guidance
        queries_text = "\n".join(f"{i + 1}. {q[:220]}" for i, q in enumerate(sampled))

        prompt = f"""You are analyzing a cluster of user queries from a conversational AI dataset.

Cluster ID: {cluster_id}
Total queries in cluster: {len(cluster_queries)}
Sample queries shown: {len(sampled)}

{EXAMPLES_PROMPT}

QUERIES:
{queries_text}

Analyze these queries and provide:
1) SHORT TITLE (5-10 words):
   - Include domain + action + subject (e.g., "Python bug debugging: pandas indexing errors").
   - Prefer concrete nouns/verbs over generic words. Avoid "general", "misc", "various".
   - If primarily code-related, prefix with the language or framework.
2) DESCRIPTION (2-3 sentences):
   - Summarize the most common intents and tasks.
   - Mention notable constraints: error fixing, homework help, API usage, jailbreak attempts, evaluation prompts, etc.
   - Call out specific tools, libraries, datasets, file types, or languages if prominent.
   - If the cluster mixes subtopics, name the 1-2 most frequent subthemes succinctly.
   - If queries are dominated by a single template or boilerplate prefix, name that pattern.
   - If placeholders/redactions (e.g., NAME_1/NAME_2, <URL>, [MASK]) are prevalent, mention this trait.
   - If the cluster is greetings-only or meta/test prompts, state that explicitly.

Write concise, specific, and informative outputs focused on what makes this cluster distinct. Keep the title in English."""

        try:
            # Call LLM with structured output - instructor handles model internally
            response = self.client.chat.completions.create(
                response_model=ClusterSummaryResponse,
                messages=[{"role": "user", "content": prompt}],
            )

            return {
                "title": response.title,
                "description": response.description,
                "sample_queries": sampled[:10],  # Store top 10 for reference
            }

        except Exception as e:
            console.print(
                f"[yellow]Warning: Failed to generate summary for cluster {cluster_id}: {e}[/yellow]"
            )
            # Fallback to basic summary
            return {
                "title": f"Cluster {cluster_id}",
                "description": f"Contains {len(cluster_queries)} queries. Sample: {cluster_queries[0][:100]}...",
                "sample_queries": sampled[:10],
            }

    def generate_batch_summaries(
        self,
        clusters_data: List[tuple[int, List[str]]],
        max_queries: int = 50,
        concurrency: Optional[int] = None,
        rpm: Optional[int] = None,
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
        )

    # -------- Async path below --------

    async def _async_generate_batch_summaries(
        self,
        clusters_data: List[Tuple[int, List[str]]],
        max_queries: int,
        concurrency: int,
        rpm: Optional[int],
    ) -> Dict[int, dict]:
        """Async concurrent generation using anyio with optional rate limiting."""

        class RateLimiter:
            def __init__(self, rate_per_min: Optional[int]):
                self.rate_per_sec = (
                    None if rate_per_min is None else float(rate_per_min) / 60.0
                )
                self._lock = anyio.Lock()
                self._next_time = time.monotonic()

            async def wait(self):
                if self.rate_per_sec is None or self.rate_per_sec <= 0:
                    return
                delay = 0.0
                async with self._lock:
                    now = time.monotonic()
                    slot = max(self._next_time, now)
                    delay = max(0.0, slot - now)
                    # schedule next slot
                    self._next_time = slot + (1.0 / self.rate_per_sec)
                if delay > 0:
                    await anyio.sleep(delay)

        limiter = RateLimiter(rpm)
        total = len(clusters_data)
        results: Dict[int, dict] = {}
        semaphore = anyio.Semaphore(concurrency)

        console.print(
            f"[cyan]Generating LLM summaries for {total} clusters using {self.model} (concurrency={concurrency}{', rpm=' + str(rpm) if rpm else ''})...[/cyan]"
        )

        async def worker(
            idx: int, cid: int, queries: List[str], progress_task, progress
        ):
            # Build per-cluster summary with retries and rate limit
            backoff = 1.0
            for attempt in range(5):
                try:
                    async with semaphore:
                        await limiter.wait()
                        summary = await self._async_generate_cluster_summary(
                            cluster_queries=queries,
                            cluster_id=cid,
                            max_queries=max_queries,
                        )
                    results[cid] = summary
                    if progress_task is not None:
                        progress.update(progress_task, advance=1)
                    return
                except Exception:
                    await anyio.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
            # Exhausted retries; fallback to basic
            results[cid] = {
                "title": f"Cluster {cid}",
                "description": f"Contains {len(queries)} queries.",
                "sample_queries": (queries[:10] if queries else []),
            }
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

    async def _async_generate_cluster_summary(
        self,
        cluster_queries: List[str],
        cluster_id: int,
        max_queries: int = 50,
    ) -> dict:
        """Async variant of single-cluster summarization using AsyncInstructor."""
        # Select a representative and diverse sample of queries
        sampled = self._select_representative_queries(cluster_queries, max_queries)

        queries_text = "\n".join(f"{i + 1}. {q[:220]}" for i, q in enumerate(sampled))
        prompt = f"""You are analyzing a cluster of user queries from a conversational AI dataset.

Cluster ID: {cluster_id}
Total queries in cluster: {len(cluster_queries)}
Sample queries shown: {len(sampled)}

{EXAMPLES_PROMPT}

QUERIES:
{queries_text}

Analyze these queries and provide:
1) SHORT TITLE (5-10 words):
   - Include domain + action + subject (e.g., "Python bug debugging: pandas indexing errors").
   - Prefer concrete nouns/verbs over generic words. Avoid "general", "misc", "various".
   - If primarily code-related, prefix with the language or framework.
2) DESCRIPTION (2-3 sentences):
   - Summarize the most common intents and tasks.
   - Mention notable constraints: error fixing, homework help, API usage, jailbreak attempts, evaluation prompts, etc.
   - Call out specific tools, libraries, datasets, file types, or languages if prominent.
   - If the cluster mixes subtopics, name the 1-2 most frequent subthemes succinctly.
   - If queries are dominated by a single template or boilerplate prefix, name that pattern.
   - If placeholders/redactions (e.g., NAME_1/NAME_2, <URL>, [MASK]) are prevalent, mention this trait.
   - If the cluster is greetings-only or meta/test prompts, state that explicitly.

Write concise, specific, and informative outputs focused on what makes this cluster distinct. Keep the title in English."""

        response = await self.async_client.chat.completions.create(
            response_model=ClusterSummaryResponse,
            messages=[{"role": "user", "content": prompt}],
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
        """Select a diverse, representative subset of queries using TF-IDF + MMR.

        - Deduplicates queries (case-insensitive, trimmed)
        - Uses TF-IDF to compute a centroid of the cluster
        - Applies Maximal Marginal Relevance (MMR) to balance relevance and diversity
        """
        if not cluster_queries:
            return []

        # Normalize and deduplicate while preserving original text
        seen = set()
        normalized: List[Tuple[str, str]] = []  # (original, key)
        for q in cluster_queries:
            q = q.strip()
            key = " ".join(q.lower().split())
            if key and key not in seen:
                seen.add(key)
                normalized.append((q, key))

        originals = [o for o, _ in normalized]
        k = min(max_queries, len(originals))
        if k <= 0:
            return []
        if k >= len(originals):
            return originals

        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
        )
        X = vectorizer.fit_transform(originals)

        # Centroid relevance
        centroid = X.mean(axis=0)
        rel = cosine_similarity(X, centroid).ravel()

        # MMR selection
        lambda_param = 0.7
        selected: List[int] = []
        candidates = set(range(X.shape[0]))

        # Start with most relevant
        first = int(np.argmax(rel))
        selected.append(first)
        candidates.remove(first)

        # Precompute pairwise similarities lazily as needed
        while len(selected) < k and candidates:
            # Compute diversity term: max similarity to any selected
            sims_to_selected = cosine_similarity(X[list(candidates)], X[selected]).max(
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

        # Preserve original order lightly by sorting chosen by original index
        selected.sort()
        return [originals[i] for i in selected]
