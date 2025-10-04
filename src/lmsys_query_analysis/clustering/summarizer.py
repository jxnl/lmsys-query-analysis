"""LLM-based cluster summarization using instructor.

Adds concurrent summarization using anyio + AsyncInstructor with
optional rate limiting to speed up large runs safely.
"""

from typing import List, Optional, Dict, Tuple
 
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
        model: str = "anthropic/sonnet-4.5-latest",
        api_key: Optional[str] = None,
        concurrency: int = 4,
        rpm: Optional[int] = None,
    ):
        """Initialize the summarizer.

        Args:
            model: Model in format "provider/model_name" (e.g.,
                   "anthropic/sonnet-4.5-latest", "openai/gpt-5",
                   "groq/llama-3.1-8b-instant")
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
        contrast_block: str = "",
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

{EXAMPLES_PROMPT}

TARGET CLUSTER QUERIES:
<queries>
{queries_text}
</queries>

<contrast_block>
{contrast_block}
</contrast_block>

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
        """Async concurrent generation using anyio with optional rate limiting.

        Also computes contrastive neighbor context to include in prompts.
        """

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

        # Pre-select representative queries for each cluster
        sampled_map: Dict[int, List[str]] = {}
        for cid, qtexts in clusters_data:
            sampled_map[cid] = self._select_representative_queries(qtexts, max_queries)

        # Compute neighbor similarity using embeddings; optionally TF-IDF for keywords mode
        id_list = [cid for cid, _ in clusters_data]
        docs = ["\n".join(sampled_map[cid]) for cid in id_list]
        sizes_map: Dict[int, int] = {cid: len(qs) for cid, qs in clusters_data}
        sim_matrix = None
        Xc = None
        vec = None
        if len(docs) >= 2 and contrast_neighbors > 0:
            # Prefer embeddings for neighbor selection
            try:
                import os
                from .embeddings import EmbeddingGenerator

                provider = "openai" if os.getenv("OPENAI_API_KEY") else "sentence-transformers"
                model_name = (
                    "text-embedding-3-small" if provider == "openai" else "all-MiniLM-L6-v2"
                )
                eg = EmbeddingGenerator(
                    model_name=model_name,
                    provider=provider,
                    concurrency=self.concurrency,
                )
                emb = eg.generate_embeddings(docs, show_progress=False)
                sim_matrix = cosine_similarity(emb, emb)
            except Exception:
                # Fallback to TF-IDF similarity if embeddings fail
                vec = TfidfVectorizer(
                    max_features=8000, ngram_range=(1, 2), min_df=1, max_df=0.95
                )
                Xc = vec.fit_transform(docs)
                sim_matrix = cosine_similarity(Xc, Xc)

            # If keywords mode requested, build TF-IDF matrix for keyword extraction
            if vec is None and Xc is None and contrast_mode == "keywords":
                vec = TfidfVectorizer(
                    max_features=8000, ngram_range=(1, 2), min_df=1, max_df=0.95
                )
                Xc = vec.fit_transform(docs)

        # Prepare neighbor context per cluster
        neighbor_context: Dict[int, str] = {}
        use_neighbors = (contrast_neighbors or 0) > 0 and (contrast_examples or 0) > 0 and contrast_mode == "neighbors"
        use_keywords = (contrast_neighbors or 0) > 0 and contrast_mode == "keywords"
        for i, cid in enumerate(id_list):
            block_lines: List[str] = []
            if sim_matrix is None:
                neighbor_context[cid] = ""
                continue
            sims = sim_matrix[i].copy()
            sims[i] = -1.0  # exclude self
            nbr_idx = sims.argsort()[::-1][: max(0, int(contrast_neighbors))]
            if len(nbr_idx) == 0:
                neighbor_context[cid] = ""
                continue
            block_lines.append("CONTRASTIVE NEIGHBORS (do not summarize these; use only to clarify distinctions):")
            for j in nbr_idx:
                nid = id_list[j]
                nsize = sizes_map.get(nid, 0)
                if use_neighbors:
                    exs = sampled_map.get(nid, [])[: max(0, int(contrast_examples))]
                    exs_fmt = []
                    for e in exs:
                        line = e.splitlines()[0].strip()
                        if len(line) > 180:
                            line = line[:177] + "..."
                        exs_fmt.append(line)
                    block_lines.append(f"Neighbor (Cluster {nid}, size≈{nsize}):")
                    for ex in exs_fmt:
                        block_lines.append(f"- {ex}")
                elif use_keywords and Xc is not None and vec is not None:
                    row = Xc[j]
                    if row.nnz:
                        nz = row.nonzero()[1]
                        weights = row.data
                        order = weights.argsort()[::-1]
                        top_idx = [nz[k] for k in order[:7]]
                        vocab = vec.get_feature_names_out()
                        kws = [vocab[t] for t in top_idx]
                        block_lines.append(
                            f"Neighbor (Cluster {nid}, size≈{nsize}) keywords: " + ", ".join(kws)
                        )
            neighbor_context[cid] = "\n".join(block_lines) if block_lines else ""

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
                            contrast_block=neighbor_context.get(cid, ""),
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
        contrast_block: str = "",
    ) -> dict:
        """Async variant of single-cluster summarization using AsyncInstructor."""
        # Select a representative and diverse sample of queries
        sampled = self._select_representative_queries(cluster_queries, max_queries)

        queries_text = "\n".join(f"{i + 1}. {q[:220]}" for i, q in enumerate(sampled))
        prompt = f"""You are analyzing a cluster of user queries from a conversational AI dataset.

<queries>
{queries_text}
</queries>

<contrast_block>
{contrast_block}
</contrast_block>

{EXAMPLES_PROMPT}

TARGET CLUSTER QUERIES:
{queries_text}

{contrast_block}

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

        # Embed using available provider (OpenAI if key, else sentence-transformers)
        try:
            import os
            from .embeddings import EmbeddingGenerator

            provider = "openai" if os.getenv("OPENAI_API_KEY") else "sentence-transformers"
            model_name = (
                "text-embedding-3-small" if provider == "openai" else "all-MiniLM-L6-v2"
            )
            eg = EmbeddingGenerator(
                model_name=model_name, provider=provider, concurrency=self.concurrency
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
            scores = lambda_param * rel[candidate_list] - (1 - lambda_param) * sims_to_selected
            best_idx = int(np.argmax(scores))
            chosen = candidate_list[best_idx]
            selected.append(chosen)
            candidates.remove(chosen)

        selected.sort()
        return [originals[i] for i in selected]
