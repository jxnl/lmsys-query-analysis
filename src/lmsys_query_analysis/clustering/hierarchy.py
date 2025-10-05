"""Hierarchical cluster merging using LLM-driven categorization.

Implements Anthropic's Clio-style hierarchical clustering approach:
1. Start with fine-grained clusters (200+)
2. Group into neighborhoods for manageable context
3. LLM generates higher-level categories
4. Deduplicate and assign children to parents
5. Refine parent names based on children
6. Repeat for multiple hierarchy levels
"""

from typing import List, Optional, Dict, Tuple
import logging
from datetime import datetime

import numpy as np
import anyio
import instructor
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential

console = Console()
logger = logging.getLogger(__name__)


# ==============================================================================
# Pydantic Models for Structured LLM Responses
# ==============================================================================

class NeighborhoodCategories(BaseModel):
    """Response for generating higher-level categories from a neighborhood of clusters."""

    scratchpad: str = Field(
        description="""Brief analysis of common themes and patterns across the clusters.
        Identify 2-4 main themes. Keep to 1-2 paragraphs maximum."""
    )
    categories: List[str] = Field(
        description="""List of broader category names that could encompass multiple clusters.

        Requirements:
        - Generate 8-15 category names
        - Be specific enough to be meaningful (avoid "General Queries", "Various Topics")
        - Use concrete terminology (technical terms, domains, specific actions)
        - Describe harmful/sensitive topics explicitly for observability
        - Each category should potentially cover 2-5 of the input clusters

        Examples:
        - "Python Web Development and API Implementation"
        - "Creative Writing and Storytelling Prompts"
        - "Toxic Content and Jailbreak Attempts"
        - "Multilingual Translation Requests (Spanish/Portuguese)"
        """,
        min_length=8,
        max_length=15
    )


class DeduplicatedClusters(BaseModel):
    """Response for deduplicating similar cluster names globally."""

    clusters: List[str] = Field(
        description="""Deduplicated list of distinct cluster names.

        Merge similar or overlapping names while preserving diversity.
        When merging, choose the most specific and descriptive name.
        Ensure remaining clusters are clearly distinct from each other.
        """,
        min_length=1
    )


class ClusterAssignment(BaseModel):
    """Response for assigning a cluster to its best-fit parent category."""

    scratchpad: str = Field(
        description="""Step-by-step reasoning (2-4 sentences):
        1. Key characteristics of the cluster being assigned
        2. Which parent categories could potentially fit
        3. Why the chosen parent is the best match
        """
    )
    assigned_cluster: str = Field(
        description="""Exact name of the chosen parent cluster.

        CRITICAL: Must match exactly one of the provided parent cluster names.
        Copy the full name without modification.
        """
    )


class RefinedClusterSummary(BaseModel):
    """Response for refining a parent cluster based on its assigned children."""

    summary: str = Field(
        description="""Two-sentence summary of the cluster in past tense.

        Sentence 1: What users were trying to accomplish
        Sentence 2: Common characteristics or patterns across children

        Be specific and avoid generic language.
        """
    )
    title: str = Field(
        description="""Concise, specific title (≤10 words) for the parent cluster.

        Requirements:
        - Accurately reflect ALL assigned children
        - Use specific technical terms, domains, or actions
        - Avoid generic words: "Diverse", "Various", "General", "Multiple"
        - Be actionable and precise for monitoring/observability

        Examples:
        - "Python Flask and Django Web Development"
        - "Harmful Content Generation and Jailbreak Attempts"
        - "Spanish and Portuguese Business Translation"
        """,
        max_length=100
    )


# ==============================================================================
# Neighborhood Clustering Helper
# ==============================================================================

def create_neighborhoods(
    embeddings: np.ndarray,
    n_neighborhoods: int,
    random_state: int = 42
) -> np.ndarray:
    """Cluster embeddings into neighborhoods using MiniBatchKMeans.

    Args:
        embeddings: Array of embeddings to cluster (N x D)
        n_neighborhoods: Target number of neighborhoods
        random_state: Random seed for reproducibility

    Returns:
        Array of neighborhood labels (N,)
    """
    clusterer = MiniBatchKMeans(
        n_clusters=n_neighborhoods,
        random_state=random_state,
        batch_size=1000
    )
    return clusterer.fit_predict(embeddings)


# ==============================================================================
# LLM Prompt Functions (Using Instructor)
# ==============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def generate_neighborhood_categories(
    client: instructor.AsyncInstructor,
    clusters: List[Dict[str, str]],
    target_count: int = 10
) -> NeighborhoodCategories:
    """Generate higher-level category names for a neighborhood of clusters.

    Args:
        client: Async instructor client
        clusters: List of dicts with 'title' and 'description'
        target_count: Target number of categories to generate

    Returns:
        NeighborhoodCategories with scratchpad and category list
    """
    cluster_str = "\n".join([
        f"<cluster>{c['title']}: {c['description']}</cluster>"
        for c in clusters
    ])

    system_prompt = """You are a taxonomist organizing user query patterns for AI safety monitoring and observability."""

    user_prompt = f"""Review these clusters and create {target_count} higher-level categories:

<cluster_list>
{cluster_str}
</cluster_list>

Your task is to create roughly {target_count} higher-level cluster names that could potentially include one or more of the provided clusters.
These higher-level clusters should represent broader categories or themes that emerge from the given clusters, while remaining as specific as possible.

Guidelines:
1. Analyze themes, topics, or characteristics common to multiple clusters
2. Create names that are specific enough to be meaningful, but broad enough to encompass 2-5 clusters
3. Avoid overly general or vague terms (no "General Queries", "Various Topics", "Diverse Requests")
4. Do not hesitate to describe harmful or sensitive topics explicitly - specificity is necessary for observability
5. Ensure category names are distinct from one another
6. Use clear, concise, descriptive language

You should output roughly {target_count} names (acceptable range: 8-15).

First, use a <scratchpad> to analyze themes (keep brief, 1-2 paragraphs).
Then provide your category names."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # Will be overridden by AutoInstructor
        response_model=NeighborhoodCategories,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def deduplicate_cluster_names(
    client: instructor.AsyncInstructor,
    candidate_names: List[str],
    target_count: int
) -> DeduplicatedClusters:
    """Deduplicate similar cluster names to create distinct categories.

    Args:
        client: Async instructor client
        candidate_names: List of candidate cluster names from all neighborhoods
        target_count: Target number of final clusters

    Returns:
        DeduplicatedClusters with unique cluster list
    """
    names_str = "\n".join([f"- {name}" for name in candidate_names])

    system_prompt = """You are a taxonomist creating a clear, distinct taxonomy from potentially overlapping category names."""

    user_prompt = f"""Deduplicate these cluster names to create ~{target_count} distinct categories:

<candidate_names>
{names_str}
</candidate_names>

Task:
1. Identify similar or overlapping names
2. Merge them, choosing the most specific and descriptive name
3. Ensure remaining clusters are clearly distinct
4. Preserve important diversity - don't over-merge

Output roughly {target_count} final cluster names."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=DeduplicatedClusters,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def assign_to_parent_cluster(
    client: instructor.AsyncInstructor,
    child_cluster: Dict[str, str],
    parent_candidates: List[str]
) -> ClusterAssignment:
    """Assign a child cluster to the best-fit parent category.

    Args:
        client: Async instructor client
        child_cluster: Dict with 'title' and 'description' of cluster to assign
        parent_candidates: List of parent cluster names

    Returns:
        ClusterAssignment with reasoning and chosen parent
    """
    parents_str = "\n".join([f"<cluster>{name}</cluster>" for name in parent_candidates])

    system_prompt = """You are categorizing clusters for observability, monitoring, and content moderation."""

    user_prompt = f"""Categorize this specific cluster into one of the provided higher-level clusters:

<specific_cluster>
Title: {child_cluster['title']}
Description: {child_cluster['description']}
</specific_cluster>

<higher_level_clusters>
{parents_str}
</higher_level_clusters>

Steps:
1. Analyze the key characteristics of the specific cluster
2. Consider which higher-level clusters could potentially fit
3. Determine the BEST match (you MUST choose one)
4. Be sensible - don't force poor fits

Use <scratchpad> for reasoning (2-4 sentences), then provide the exact parent cluster name."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ClusterAssignment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def refine_parent_cluster(
    client: instructor.AsyncInstructor,
    child_clusters: List[str]
) -> RefinedClusterSummary:
    """Refine a parent cluster's title and description based on its children.

    Args:
        client: Async instructor client
        child_clusters: List of child cluster titles assigned to this parent

    Returns:
        RefinedClusterSummary with title and summary
    """
    children_str = "\n".join([f"<cluster>{title}</cluster>" for title in child_clusters])

    system_prompt = """You are summarizing a group of related clusters into a precise, actionable description."""

    user_prompt = f"""Summarize this group of related cluster names:

<child_clusters>
{children_str}
</child_clusters>

Requirements:
1. Create a two-sentence summary in past tense:
   - Sentence 1: What users were trying to accomplish
   - Sentence 2: Common characteristics or patterns

2. Generate a title (≤10 words):
   - Accurately reflect ALL assigned children
   - Use specific technical terms, domains, or actions
   - Avoid generic words: "Diverse", "Various", "General"
   - Be actionable for monitoring/observability

Example good titles:
- "Python Flask and Django Web Development"
- "Harmful Content Generation and Jailbreak Attempts"
- "Spanish and Portuguese Business Translation"

Example bad titles:
- "Various Programming Queries"
- "General User Requests"
- "Diverse Technical Topics"
"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=RefinedClusterSummary,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response


# ==============================================================================
# Core Hierarchical Merging Algorithm
# ==============================================================================

async def merge_clusters_hierarchical(
    base_clusters: List[Dict],
    run_id: str,
    embedding_model: str = "all-mpnet-base-v2",
    llm_provider: str = "anthropic",
    llm_model: str = "claude-sonnet-4-5-20250929",
    target_levels: int = 3,
    merge_ratio: float = 0.2,
    neighborhood_size: int = 40,
    concurrency: int = 8,
    rpm: Optional[int] = None
) -> Tuple[str, List[Dict]]:
    """Perform hierarchical merging of clusters using LLM-driven categorization.

    Implements Clio-style algorithm:
    1. Embed cluster summaries
    2. Create neighborhoods for manageable context
    3. Generate higher-level categories per neighborhood
    4. Deduplicate globally
    5. Assign children to parents
    6. Refine parent names
    7. Repeat for multiple levels

    Args:
        base_clusters: List of dicts with 'cluster_id', 'title', 'description'
        run_id: Clustering run ID
        embedding_model: Sentence transformer model for embeddings
        llm_provider: LLM provider (anthropic/openai/groq)
        llm_model: LLM model name
        target_levels: Number of hierarchy levels to create
        merge_ratio: Target ratio for merging (0.2 = 200->40->8)
        neighborhood_size: Average clusters per neighborhood
        concurrency: Max concurrent LLM requests
        rpm: Optional rate limit (requests per minute)

    Returns:
        Tuple of (hierarchy_run_id, hierarchy_list)
        where hierarchy_list contains dicts with hierarchy metadata
    """
    logger.info(f"Starting hierarchical merging: {len(base_clusters)} base clusters")

    # Create hierarchy run ID
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    hierarchy_run_id = f"hier-{run_id}-{timestamp}"

    # Initialize embedding model
    logger.info(f"Loading embedding model: {embedding_model}")
    embedder = SentenceTransformer(embedding_model)

    # Initialize LLM client
    logger.info(f"Initializing LLM: {llm_provider}/{llm_model}")

    # Build full model string for instructor (e.g., "openai/gpt-4o-mini")
    full_model = f"{llm_provider}/{llm_model}"
    client = instructor.from_provider(full_model, async_client=True)

    # Rate limiter
    limiter = None
    if rpm:
        limiter = AsyncLimiter(rpm, 60.0)

    # Initialize hierarchy storage
    hierarchy = []

    # Level 0: Add base clusters as leaves
    for cluster in base_clusters:
        hierarchy.append({
            "hierarchy_run_id": hierarchy_run_id,
            "run_id": run_id,
            "cluster_id": cluster["cluster_id"],
            "parent_cluster_id": None,
            "level": 0,
            "children_ids": [],
            "title": cluster["title"],
            "description": cluster["description"]
        })

    # Current level clusters (start with base)
    current_clusters = base_clusters
    current_level = 0
    next_cluster_id = max(c["cluster_id"] for c in base_clusters) + 1

    # Build hierarchy iteratively
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        while current_level < target_levels - 1:
            n_current = len(current_clusters)
            n_target = max(int(n_current * merge_ratio), 2)

            logger.info(f"Level {current_level} -> {current_level + 1}: {n_current} -> {n_target} clusters")

            task = progress.add_task(
                f"Merging level {current_level} -> {current_level + 1}",
                total=n_current
            )

            # Step 1: Embed cluster summaries
            texts = [f"{c['title']}: {c['description']}" for c in current_clusters]
            embeddings = embedder.encode(texts, show_progress_bar=False)

            # Step 2: Create neighborhoods
            n_neighborhoods = max(n_current // neighborhood_size, 1)
            neighborhood_labels = create_neighborhoods(embeddings, n_neighborhoods)

            # Step 3: Generate higher-level categories per neighborhood
            all_candidates = []

            for neigh_id in range(n_neighborhoods):
                neigh_clusters = [
                    current_clusters[i]
                    for i in range(len(current_clusters))
                    if neighborhood_labels[i] == neigh_id
                ]

                if limiter:
                    async with limiter:
                        categories = await generate_neighborhood_categories(
                            client, neigh_clusters, target_count=int(len(neigh_clusters) * merge_ratio)
                        )
                else:
                    categories = await generate_neighborhood_categories(
                        client, neigh_clusters, target_count=int(len(neigh_clusters) * merge_ratio)
                    )

                all_candidates.extend(categories.categories)
                logger.info(f"Neighborhood {neigh_id}: Generated {len(categories.categories)} candidates")

            # Step 4: Deduplicate globally
            if limiter:
                async with limiter:
                    dedup_result = await deduplicate_cluster_names(
                        client, all_candidates, n_target
                    )
            else:
                dedup_result = await deduplicate_cluster_names(
                    client, all_candidates, n_target
                )

            parent_names = dedup_result.clusters
            logger.info(f"Deduplicated to {len(parent_names)} parent clusters")

            # Step 5: Assign children to parents (sequential for now - TODO: parallelize properly)
            parent_children = {name: [] for name in parent_names}

            for cluster in current_clusters:
                if limiter:
                    async with limiter:
                        assignment = await assign_to_parent_cluster(client, cluster, parent_names)
                else:
                    assignment = await assign_to_parent_cluster(client, cluster, parent_names)

                if assignment.assigned_cluster in parent_children:
                    parent_children[assignment.assigned_cluster].append(cluster["cluster_id"])

                progress.update(task, advance=1)

            # Step 6: Refine parent names based on children
            next_level_clusters = []

            for parent_name in parent_names:
                child_ids = parent_children[parent_name]
                if not child_ids:
                    logger.warning(f"Parent '{parent_name}' has no children, skipping")
                    continue

                child_titles = [
                    c["title"] for c in current_clusters if c["cluster_id"] in child_ids
                ]

                if limiter:
                    async with limiter:
                        refined = await refine_parent_cluster(client, child_titles)
                else:
                    refined = await refine_parent_cluster(client, child_titles)

                parent_cluster_id = next_cluster_id
                next_cluster_id += 1

                # Add to hierarchy
                hierarchy.append({
                    "hierarchy_run_id": hierarchy_run_id,
                    "run_id": run_id,
                    "cluster_id": parent_cluster_id,
                    "parent_cluster_id": None,  # Will be set in next iteration
                    "level": current_level + 1,
                    "children_ids": child_ids,
                    "title": refined.title,
                    "description": refined.summary
                })

                # Update children's parent_id
                for child_id in child_ids:
                    for h in hierarchy:
                        if h["cluster_id"] == child_id and h["level"] == current_level:
                            h["parent_cluster_id"] = parent_cluster_id
                            break

                # Add to next level
                next_level_clusters.append({
                    "cluster_id": parent_cluster_id,
                    "title": refined.title,
                    "description": refined.summary
                })

                logger.info(f"Created parent cluster {parent_cluster_id}: '{refined.title}' with {len(child_ids)} children")

            # Move to next level
            current_clusters = next_level_clusters
            current_level += 1

            if len(current_clusters) <= 1:
                logger.info("Reached single top-level cluster, stopping")
                break

    logger.info(f"Hierarchy complete: {len(hierarchy)} total nodes across {current_level + 1} levels")
    return hierarchy_run_id, hierarchy
