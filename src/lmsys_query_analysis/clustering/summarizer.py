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
<few_shot_examples>
  <example label="A">
    <sample_queries>
      <item>Why does pandas .loc throw KeyError on column subset?</item>
      <item>Fix TypeError: unsupported operand type(s) for +: 'int' and 'str' in Python</item>
      <item>Vectorizing loops in NumPy for performance</item>
    </sample_queries>
    <title>Python debugging and data wrangling: pandas, NumPy errors</title>
    <description>Users troubleshoot Python code with a focus on pandas indexing, NumPy array operations, and common Type/Key errors. Guidance emphasizes minimal examples, error interpretation, and idiomatic fixes.</description>
  </example>
  <example label="B">
    <sample_queries>
      <item>Fetch data from a REST API with bearer token using fetch()</item>
      <item>Axios POST returns 401 — how to send JWT header?</item>
      <item>Handle 429 rate limit responses and retries in JavaScript</item>
    </sample_queries>
    <title>JavaScript REST API usage and authentication</title>
    <description>Requests cover calling APIs from the browser/node, attaching JWT/Bearer auth, handling status codes (401/429), and implementing retry/backoff. Code examples use fetch and Axios patterns.</description>
  </example>
  <example label="C">
    <sample_queries>
      <item>SQL: Count distinct users per day with LEFT JOIN</item>
      <item>GROUP BY with conditional SUM over multiple categories</item>
      <item>Window function to find top N products per region</item>
    </sample_queries>
    <title>SQL analytics with joins, aggregates, and windows</title>
    <description>Analytical SQL tasks combining joins, GROUP BY, conditional aggregation, and window functions (RANK/ROW_NUMBER). Solutions emphasize readable CTEs and correct grouping semantics.</description>
  </example>
  <example label="D">
    <sample_queries>
      <item>Fine-tune a small Llama model on custom Q&amp;A data</item>
      <item>Why is my training loss NaN in PyTorch mixed precision?</item>
      <item>Compare F1 and accuracy for imbalanced classification</item>
    </sample_queries>
    <title>ML model training and evaluation: PyTorch, metrics, fine-tuning</title>
    <description>Practical ML workflows including dataset prep, training stability (overflow/AMP), and metric selection for imbalanced tasks. Advice centers on troubleshooting training loops, schedulers, and evaluation protocols.</description>
  </example>
  <example label="E">
    <sample_queries>
      <item>Docker build ignores .dockerignore; how to reduce image size?</item>
      <item>Nginx reverse proxy 502 when forwarding to Gunicorn</item>
      <item>Kubernetes CrashLoopBackOff reading config from secrets</item>
    </sample_queries>
    <title>DevOps containerization and deployment: Docker, Nginx, Kubernetes</title>
    <description>Infra topics around container images, reverse proxies, and k8s deployments. Solutions cover Docker layering, proper proxy buffers/timeouts, health checks, and Secret/ConfigMap mounts.</description>
  </example>
  <example label="F">
    <sample_queries>
      <item>Detect and prevent jailbreak attempts in system prompts</item>
      <item>Red-team prompts to bypass safety filters — show examples</item>
      <item>Classify prompts by risk category (self-harm, malware, PII)</item>
    </sample_queries>
    <title>LLM safety and prompt hardening: jailbreak detection and taxonomy</title>
    <description>Safety and governance tasks including jailbreak detection, red-teaming, and content risk classification. Emphasis on policies, refusal guidelines, and robust prompt templates.</description>
  </example>
  <example label="G">
    <sample_queries>
      <item>Create a Seaborn heatmap with annotations and custom colorbar</item>
      <item>Altair interactive chart: filter by dropdown, highlight selection</item>
      <item>Plotly Express facet grid with per-facet y-axis</item>
    </sample_queries>
    <title>Data visualization recipes: Seaborn, Altair, Plotly</title>
    <description>Visualization tasks across common Python libraries, focusing on annotated heatmaps, interactive selections, and faceting. Guidance prioritizes clear legends, accessibility, and aesthetics.</description>
  </example>
  <example label="H">
    <sample_queries>
      <item>Translate product descriptions to Spanish and preserve HTML tags</item>
      <item>Localize date/number formats for German users</item>
      <item>Glossary-enforced translation for brand terms</item>
    </sample_queries>
    <title>Multilingual translation and localization with formatting constraints</title>
    <description>Translation requests with constraints on placeholders/HTML and locale-aware formatting. Recommendations include tag-aware translation and glossary/terminology control.</description>
  </example>
</few_shot_examples>
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
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
    ):
        """Initialize the summarizer.

        Args:
            model: Model in format "provider/model_name" (e.g.,
                   "anthropic/sonnet-4.5-latest", "openai/gpt-5",
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

        # Build XML-formatted prompt
        queries_xml = "\n".join(
            f"      <query idx=\"{i + 1}\"><![CDATA[{q[:220]}]]></query>" for i, q in enumerate(sampled)
        )

        prompt = f"""
<summary_task>
  <cluster id="{cluster_id}">
    <stats total_queries="{len(cluster_queries)}" sample_count="{len(sampled)}" />
    <target_queries>
{queries_xml}
    </target_queries>
  </cluster>
  <instructions>
    <title_guidelines>
      <rule>5-10 words; include domain + action + subject.</rule>
      <rule>Prefer concrete nouns/verbs; avoid generic terms ("general", "misc").</rule>
      <rule>If code-heavy, prefix with language or framework.</rule>
      <rule>Keep the title in English.</rule>
    </title_guidelines>
    <description_guidelines>
      <rule>Summarize common intents and tasks.</rule>
      <rule>Mention constraints: error fixing, homework help, API usage, jailbreak attempts, evaluation prompts.</rule>
      <rule>Call out tools/libraries/datasets/languages if prominent.</rule>
      <rule>If mixed topics, name top 1–2 subthemes.</rule>
      <rule>Name dominant template/boilerplate patterns.</rule>
      <rule>Mention placeholders/redactions (NAME_1/NAME_2, &lt;URL&gt;, [MASK]) if prevalent.</rule>
      <rule>State explicitly if greetings-only or meta/test prompts.</rule>
    </description_guidelines>
  </instructions>
  <output>Return fields: title, description.</output>
</summary_task>
"""

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
            block_lines.append("<contrastive_neighbors>")
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
                    block_lines.append(f"  <neighbor cluster_id=\"{nid}\" size=\"{nsize}\">")
                    for ex in exs_fmt:
                        block_lines.append(f"    <example><![CDATA[{ex}]]></example>")
                    block_lines.append("  </neighbor>")
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
                            f"  <neighbor cluster_id=\"{nid}\" size=\"{nsize}\"><keywords>"
                            + ", ".join(kws)
                            + "</keywords></neighbor>"
                        )
            block_lines.append("</contrastive_neighbors>")
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

        # Build XML-formatted prompt
        queries_xml = "\n".join(
            f"      <query idx=\"{i + 1}\"><![CDATA[{q[:220]}]]></query>" for i, q in enumerate(sampled)
        )
        prompt = f"""
<summary_task>
  <cluster id="{cluster_id}">
    <stats total_queries="{len(cluster_queries)}" sample_count="{len(sampled)}" />
    <target_queries>
{queries_xml}
    </target_queries>
  </cluster>
  {contrast_block}
  <instructions>
    <title_guidelines>
      <rule>5-10 words; include domain + action + subject.</rule>
      <rule>Prefer concrete nouns/verbs; avoid generic terms ("general", "misc").</rule>
      <rule>If code-heavy, prefix with language or framework.</rule>
      <rule>Keep the title in English.</rule>
    </title_guidelines>
    <description_guidelines>
      <rule>Summarize common intents and tasks.</rule>
      <rule>Mention constraints: error fixing, homework help, API usage, jailbreak attempts, evaluation prompts.</rule>
      <rule>Call out tools/libraries/datasets/languages if prominent.</rule>
      <rule>If mixed topics, name top 1–2 subthemes.</rule>
      <rule>Name dominant template/boilerplate patterns.</rule>
      <rule>Mention placeholders/redactions (NAME_1/NAME_2, &lt;URL&gt;, [MASK]) if prevalent.</rule>
      <rule>State explicitly if greetings-only or meta/test prompts.</rule>
    </description_guidelines>
  </instructions>
  <output>Return fields: title, description.</output>
</summary_task>
"""

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

        # Embed using configured embedding model (matches clustering model)
        try:
            from .embeddings import EmbeddingGenerator

            eg = EmbeddingGenerator(
                model_name=self.embedding_model,
                provider=self.embedding_provider,
                concurrency=self.concurrency
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
