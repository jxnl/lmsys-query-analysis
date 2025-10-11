"""Hierarchical cluster merging using LLM-driven categorization.

Implements Anthropic's Clio-style hierarchical clustering approach:
1. Start with fine-grained clusters (200+)
2. Group into neighborhoods for manageable context
3. LLM generates higher-level categories
4. Deduplicate and assign children to parents
5. Refine parent names based on children
6. Repeat for multiple hierarchy levels
"""

from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
import logging
from datetime import datetime
import random
import time

from cohere.types import usage
import numpy as np
import anyio
import asyncio
import instructor
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sklearn.cluster import MiniBatchKMeans
from jinja2 import Environment
from .embeddings import EmbeddingGenerator
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
from ..runner import BaseCluster
from ..utils.logging import trace

if TYPE_CHECKING:
    from ..db.connection import Database

console = Console()
logger = logging.getLogger(__name__)


# ==============================================================================
# Jinja2 Templates for LLM Prompts
# ==============================================================================

def _create_neighborhood_categories_template():
    """Create Jinja2 template for neighborhood category generation."""
    env = Environment()
    return env.from_string("""Review these clusters and create a behavioral taxonomy with roughly {{ target_count }} categories:

<cluster_list>
{% for cluster in clusters -%}
<cluster>{{ cluster.title }}: {{ cluster.description }}
{% if cluster.representative_queries -%}
Examples:
{% for query in cluster.representative_queries[:3] -%}
- {{ query }}
{% endfor -%}
{% endif -%}
</cluster>
{% endfor -%}
</cluster_list>

Your task is to identify USER BEHAVIORS and MENTAL MODELS, then create {{ target_count }} higher-level category names.

Think like the Anthropic Education Report:
1. Identify behavioral patterns (how do users phrase requests? What do they assume?)
2. Find user segments (novices vs experts, different mental models)
3. Map along taxonomic dimensions (expertise level, autonomy/agency, task type, output expectations)
4. Look for product insights (expectation gaps, unmet needs, emerging use cases)

Guidelines:
1. FOCUS ON BEHAVIORS, not just topics (e.g., "Novice Programmers Seeking Complete Solutions" not "Programming")
2. Create names specific enough for product decisions, broad enough to encompass 2-5 clusters
3. Avoid generic terms ("General Queries", "Various Topics", "Diverse Requests", "Technical Tasks", "Professional Use")
4. Explicitly describe harmful/sensitive patterns for observability (e.g., "Jailbreak via Roleplay Prompts")
5. Ensure categories represent distinct user segments or interaction patterns
6. Think: what would a PM or UX researcher want to know?
7. PREFER SPECIFICITY: Include concrete domains, technologies, or use cases in category names
8. Each category should be DISTINCT and NON-OVERLAPPING with others

CRITICAL: Prioritize SPECIFICITY over hitting target count exactly.
- Minimum output: {{ target_count }} categories
- Target output: {{ int(target_count * 1.3) }} categories (30% more is GOOD)
- Maximum output: {{ int(target_count * 1.5) }} categories

It's better to have {{ int(target_count * 1.5) }} specific categories than {{ target_count }} generic ones.
When in doubt, CREATE MORE CATEGORIES rather than merging into generic buckets.

Provide your category names focusing on user behavior and mental models.""")



class BaseCluster(BaseModel):
    """Base cluster model for hierarchical merging."""
    cluster_id: int
    title: str
    description: str
    representative_queries: Optional[List[str]] = None

# ==============================================================================
# Pydantic Models for Structured LLM Responses
# ==============================================================================

class NeighborhoodCategories(BaseModel):
    """Response for generating higher-level categories from a neighborhood of clusters."""

    categories: List[str] = Field(
        description="""List of broader category names based on behavioral patterns and user segments.

        Requirements:
        - Generate 8-20 category names (prefer MORE specific over FEWER generic)
        - Focus on USER BEHAVIORS and MENTAL MODELS, not just topics
        - Use concrete terminology (technical terms, domains, specific actions)
        - Think taxonomically: identify axes like "expertise level", "autonomy", "output type"
        - Describe harmful/sensitive topics explicitly for observability
        - Each category should represent a distinct user segment or interaction pattern
        - AVOID generic terms: "General", "Various", "Diverse", "Technical Tasks", "Professional Use"

        Examples (behavioral/segment-focused - GOOD):
        - "Novice Programmers Seeking Complete Code Solutions"
        - "Creative Writers Requesting Story Prompts and Continuations"
        - "Jailbreak Attempts via Roleplay Prompts"
        - "Non-Native Speakers Using LLMs for Professional Translation"
        - "Expert Developers Seeking Architectural Advice"
        - "Data Scientists Requesting Python Pandas Analysis Code"
        - "Web Developers Debugging React Component Errors"
        - "Python Flask API Development Assistance"
        - "Chemical Engineers Generating Technical Documentation"
        - "SQL Database Query Optimization Requests"

        Examples (BAD - too generic/topic-focused):
        - "Python Programming" ❌
        - "Creative Writing" ❌
        - "Translation" ❌
        - "Technical Task Automation" ❌
        - "Industry Professionals" ❌
        - "Professional Development" ❌
        - "Coding Assistance" ❌
        - "Users Automating Tasks" ❌
        """,
        min_length=8,
        max_length=100
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

    assigned_cluster: str = Field(
        description="""Exact name of the chosen parent cluster.

        CRITICAL: Must match exactly one of the provided parent cluster names.
        Copy the full name without modification.
        """
    )


class RefinedClusterSummary(BaseModel):
    """Response for refining a parent cluster based on its assigned children."""

    summary: str = Field(
        description="""Two-sentence behavioral insight summary in past tense.

        Sentence 1: What users were trying to accomplish and their mental model/approach
        Sentence 2: Key behavioral patterns, user segments, or product implications across children

        Focus on INSIGHTS, not just description. What does this reveal about how people use LLMs?

        Example GOOD:
        "Users sought complete coding solutions with minimal specification, treating the LLM as a 'magic wand'
        code generator. Novices pasted homework problems verbatim while experts provided architectural constraints,
        revealing distinct mental models of AI capability across expertise levels."

        Example BAD:
        "Users asked programming questions in various languages. Questions covered web development, data analysis,
        and algorithms." ❌
        """
    )
    title: str = Field(
        description="""Concise, behavior-focused title (≤10 words) for the parent cluster.

        Requirements:
        - Accurately reflect ALL assigned children
        - Emphasize USER BEHAVIOR or SEGMENT when possible
        - Use specific technical terms, domains, or actions
        - Avoid generic words: "Diverse", "Various", "General", "Multiple"
        - Be actionable for product decisions and observability

        Examples (GOOD - behavioral focus):
        - "Novice Programmers Expecting Complete Code Generation"
        - "Jailbreak Attempts via Roleplay and Prompt Injection"
        - "Non-Native Business Communication and Translation"
        - "Expert Developers Seeking Architectural Consultation"

        Examples (OK - topic focus):
        - "Python Flask and Django Web Development"
        - "Spanish and Portuguese Business Translation"

        Examples (BAD):
        - "Various Programming Topics" ❌
        - "General Development Queries" ❌
        """,
        max_length=100
    )


# ==============================================================================
# Representative Query Selection Helper
# ==============================================================================

@trace
def select_representative_queries_from_db(
    db: "Database",
    run_id: str,
    child_cluster_ids: List[int],
    max_queries: int = 10
) -> List[str]:
    """Select representative queries from child clusters by querying the database.
    
    Args:
        db: Database connection
        run_id: Clustering run ID
        child_cluster_ids: List of child cluster IDs
        max_queries: Maximum number of queries to select
        
    Returns:
        List of representative query texts
    """
    from sqlmodel import select
    from ..db.models import Query, QueryCluster
    
    if not child_cluster_ids:
        return []
    
    # Query the database for queries in these clusters
    with db.get_session() as session:
        # Get queries from child clusters
        statement = (
            select(Query.query_text)
            .join(QueryCluster, Query.id == QueryCluster.query_id)
            .where(QueryCluster.run_id == run_id)
            .where(QueryCluster.cluster_id.in_(child_cluster_ids))
        )
        
        query_texts = session.exec(statement).all()
    
    if not query_texts:
        return []
    
    # Deduplicate queries (case-insensitive, trimmed)
    seen = set()
    unique_queries = []
    for q in query_texts:
        q = (q or "").strip()
        key = " ".join(q.lower().split())
        if key and key not in seen:
            seen.add(key)
            unique_queries.append(q)
    
    # Select up to max_queries using random sampling
    k = min(max_queries, len(unique_queries))
    if k <= 0:
        return []
    
    # Randomly sample k queries
    return random.sample(unique_queries, k)


# ==============================================================================
# Neighborhood Clustering Helper
# ==============================================================================

@trace
def create_neighborhoods(
    ids: List[int],
    embeddings: np.ndarray,
    n_neighborhoods: int,
    random_state: int = 42
) -> List[tuple[int, int]]:
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
    return list(zip(ids, clusterer.fit_predict(embeddings)))


# ==============================================================================
# LLM Prompt Functions (Using Instructor)
# ==============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
@trace
async def generate_neighborhood_categories(
    client: instructor.AsyncInstructor,
    clusters: List[BaseCluster],
    target_count: int = 10
) -> NeighborhoodCategories:
    """Generate higher-level category names for a neighborhood of clusters.

    Args:
        client: Async instructor client
        clusters: List of BaseCluster objects with title and description
        target_count: Target number of categories to generate

    Returns:
        NeighborhoodCategories with category list
    """
    user_prompt = """Review these clusters and create a behavioral taxonomy with roughly {{ target_count }} categories:

    <cluster_list>
    {% for cluster in clusters -%}
    <cluster>{{ cluster.title }}: {{ cluster.description }}
    {% if cluster.representative_queries -%}
    Examples:
    {% for query in cluster.representative_queries[:3] -%}
    - {{ query }}
    {% endfor -%}
    {% endif -%}
    </cluster>
    {% endfor -%}
    </cluster_list>

    Your task is to identify USER BEHAVIORS and MENTAL MODELS, then create {{ target_count }} higher-level category names.

    Think like the Anthropic Education Report:
    1. Identify behavioral patterns (how do users phrase requests? What do they assume?)
    2. Find user segments (novices vs experts, different mental models)
    3. Map along taxonomic dimensions (expertise level, autonomy/agency, task type, output expectations)
    4. Look for product insights (expectation gaps, unmet needs, emerging use cases)

    Guidelines:
    1. FOCUS ON BEHAVIORS, not just topics (e.g., "Novice Programmers Seeking Complete Solutions" not "Programming")
    2. Create names specific enough for product decisions, broad enough to encompass 2-5 clusters
    3. Avoid generic terms ("General Queries", "Various Topics", "Diverse Requests", "Technical Tasks", "Professional Use")
    4. Explicitly describe harmful/sensitive patterns for observability (e.g., "Jailbreak via Roleplay Prompts")
    5. Ensure categories represent distinct user segments or interaction patterns
    6. Think: what would a PM or UX researcher want to know?
    7. PREFER SPECIFICITY: Include concrete domains, technologies, or use cases in category names
    8. Each category should be DISTINCT and NON-OVERLAPPING with others

    CRITICAL: Prioritize SPECIFICITY over hitting target count exactly.
    - Minimum output: {{ target_count }} categories
    - Target output: {{ int(target_count * 1.3) }} categories (30% more is GOOD)
    - Maximum output: {{ int(target_count * 1.5) }} categories

    It's better to have {{ int(target_count * 1.5) }} specific categories than {{ target_count }} generic ones.
    When in doubt, CREATE MORE CATEGORIES rather than merging into generic buckets.

    Provide your category names focusing on user behavior and mental models."""

    system_prompt = """You are a behavioral researcher and taxonomist analyzing how people interact with LLMs. Your goal is to create a framework that reveals user mental models, behaviors, and product opportunities - not just topic categories."""

    response = await client.chat.completions.create(
        model="gpt-5", # this needs a smarter model
        reasoning_effort="medium",
        response_model=NeighborhoodCategories,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        context={
            "clusters": clusters,
            "target_count": target_count
        }
    )
    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
@trace
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
    user_prompt = """Deduplicate these cluster names to create ~{{ target_count }} distinct categories:

    <candidate_names>
    {% for name in candidate_names -%}
    - {{ name }}
    {% endfor -%}
    </candidate_names>

    Task:
    1. Identify ONLY near-duplicate names (>90% semantic overlap - essentially the same category)
    2. DEFAULT ACTION: KEEP categories separate unless they are obvious duplicates
    3. Merge ONLY when:
    - Names are near-identical (e.g., "Python Coding" vs "Python Code Generation" → merge)
    - One is a clear subset of another (e.g., "React Debugging" under "React Development" → keep separate if both have substance)
    4. DO NOT merge based on:
    - Same domain but different use cases (e.g., "SQL Query Writing" vs "SQL Performance Optimization" → KEEP BOTH)
    - Same technology but different user segments (e.g., "Novice Python Help" vs "Expert Python Architecture" → KEEP BOTH)
    - Similar topics but different behaviors (e.g., "Homework Help" vs "Professional Code Review" → KEEP BOTH)

    CRITICAL RULES:
    - Better to have {{ int(target_count * 1.4) }} specific categories than {{ target_count }} generic ones
    - When unsure, DO NOT MERGE
    - Preserve domain-specific and behavior-specific distinctions
    - Only merge if you can't distinguish the categories meaningfully

    Output {{ int(target_count * 0.9) }} to {{ int(target_count * 1.5) }} final cluster names.
    Acceptable range is wide - QUALITY (specificity) matters more than QUANTITY (hitting exact target)."""

    system_prompt = """You are a taxonomist creating a clear, distinct taxonomy from potentially overlapping category names."""

    response = await client.chat.completions.create(
        model="gpt-5-mini",
        reasoning_effort="medium",
        response_model=DeduplicatedClusters,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        context={
            "candidate_names": candidate_names,
            "target_count": target_count
        }
    )

    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
@trace
async def assign_to_parent_cluster(
    client: instructor.AsyncInstructor,
    child_cluster: BaseCluster,
    parent_candidates: List[str]
) -> ClusterAssignment:
    """Assign a child cluster to the best-fit parent category.

    Args:
        client: Async instructor client
        child_cluster: BaseCluster object with title and description
        parent_candidates: List of parent cluster names

    Returns:
        ClusterAssignment with reasoning and chosen parent
    """
    user_prompt = """Categorize this specific cluster into one of the provided higher-level clusters:

    <specific_cluster>
    Title: {{ child_cluster.title }}
    Description: {{ child_cluster.description }}
    {% if child_cluster.representative_queries -%}
    Examples:
    {% for query in child_cluster.representative_queries[:3] -%}
    - {{ query }}
    {% endfor -%}
    {% endif -%}
    </specific_cluster>

    <higher_level_clusters>
    {% for parent in parent_candidates -%}
    <cluster>{{ parent }}</cluster>
    {% endfor -%}
    </higher_level_clusters>

    Steps:
    1. Analyze the key characteristics of the specific cluster
    2. Consider which higher-level clusters could potentially fit
    3. Determine the BEST match (you MUST choose one)
    4. Be sensible - don't force poor fits

    Provide the exact parent cluster name."""

    system_prompt = """You are categorizing clusters for observability, monitoring, and content moderation."""

    logger.debug(f"Assigning '{child_cluster.title[:60]}...' to one of {len(parent_candidates)} parent options")
    response = await client.chat.completions.create(
        reasoning_effort="medium",
        response_model=ClusterAssignment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        context={
            "child_cluster": child_cluster,
            "parent_candidates": parent_candidates
        }
    )
    logger.debug(f"  → Assigned to: '{response.assigned_cluster}'")
    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
@trace
async def refine_parent_cluster(
    client: instructor.AsyncInstructor,
    child_clusters: List[BaseCluster]
) -> RefinedClusterSummary:
    """Refine a parent cluster's title and description based on its children.

    Args:
        client: Async instructor client
        child_clusters: List of BaseCluster objects assigned to this parent

    Returns:
        RefinedClusterSummary with title and summary
    """

    system_prompt = """You are a behavioral researcher creating insight-driven cluster summaries. Focus on what these patterns reveal about user behavior, mental models, and product opportunities."""

    # Create Jinja2 template for child clusters
    user_prompt = """Analyze this group of related clusters and create a behavioral insight summary:

<child_clusters>
{% for cluster in child_clusters -%}
<cluster>{{ cluster.title }}: {{ cluster.description }}
{% if cluster.representative_queries -%}
Examples:
{% for query in cluster.representative_queries[:2] -%}
- {{ query }}
{% endfor -%}
{% endif -%}
</cluster>
{% endfor -%}
</child_clusters>

Requirements:
1. Create a two-sentence BEHAVIORAL INSIGHT summary in past tense:
   - Sentence 1: What users were trying to accomplish AND their mental model/approach to AI
   - Sentence 2: Key behavioral patterns, user segments, or product implications

   Focus on INSIGHTS not description. What does this reveal?

   Example GOOD:
   "Users sought complete coding solutions with minimal specification, treating the LLM as a 'magic wand'
   code generator. Novices pasted homework verbatim while experts provided constraints, revealing distinct
   mental models across expertise levels - opportunity for adaptive interfaces."

   Example BAD:
   "Users asked programming questions. Questions covered multiple languages and frameworks." ❌

2. Generate a behavior-focused title (≤10 words):
   - Accurately reflect ALL children
   - EMPHASIZE USER BEHAVIOR or SEGMENT when possible
   - Use specific terms (technologies, domains, actions)
   - FORBIDDEN WORDS: "Diverse", "Various", "General", "Multiple", "Different", "Mixed", "Broad", "Wide-ranging", "Miscellaneous", "Assorted"
   - FORBIDDEN PHRASES: "Technical Tasks", "Professional Use", "Industry Professionals", "General Queries"
   - Include concrete details: specific languages, frameworks, domains, or user types
   - Be actionable for product/UX decisions

Example GOOD titles (behavioral focus):
- "Novice Programmers Expecting Complete Code Generation"
- "Jailbreak Attempts via Roleplay and Prompt Injection"
- "Non-Native Speakers Using LLMs for Professional Translation"
- "Data Scientists Requesting Python Pandas Analysis Code"
- "Web Developers Debugging React Component Errors"

Example OK titles (topic focus, but specific):
- "Python Flask and Django Web Development"
- "Spanish and Portuguese Business Translation"
- "SQL Database Query Optimization"

Example BAD (too generic):
- "Various Programming Queries" ❌
- "General Development Questions" ❌
- "Technical Task Automation" ❌
- "Professional Development" ❌
- "Industry Professionals Utilizing LLMs" ❌
"""

    response = await client.chat.completions.create(
        reasoning_effort="medium",
        response_model=RefinedClusterSummary,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        context={
            "child_clusters": child_clusters
        }
    )

    return response


# ==============================================================================
# Core Hierarchical Merging Algorithm
# ==============================================================================

@trace
async def merge_clusters_hierarchical(
    base_clusters: List[BaseCluster],
    run_id: str,
    db: "Database",
    embedding_model: str = "text-embedding-3-small",
    embedding_provider: str = "openai",
    llm_provider: str = "openai",
    category_model: str = "gpt-5-mini",      # For generate_neighborhood_categories (quality-critical)
    dedup_model: str = "gpt-5-mini",         # For deduplicate_cluster_names (quality-critical)
    assignment_model: str = "gpt-4o-mini",    # For assign_to_parent_cluster (high-volume, cost-sensitive)
    refinement_model: str = "gpt-4o-mini",   # For refine_parent_cluster (high-volume, cost-sensitive)
    target_levels: int = 3,
    merge_ratio: float = 0.35,  # Changed from 0.2 to 0.35 to reduce over-merging
    neighborhood_size: int = 20,  # Changed from 40 to 20 for more specific categories
    concurrency: int = 100,
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
        base_clusters: List of BaseCluster objects with cluster_id, title, and description
        run_id: Clustering run ID
        db: Database connection for querying representative queries
        embedding_model: Sentence transformer model for embeddings
        embedding_provider: Embedding provider (openai/cohere)
        llm_provider: LLM provider (anthropic/openai/groq)
        category_model: Model for generate_neighborhood_categories (quality-critical)
        dedup_model: Model for deduplicate_cluster_names (quality-critical)
        assignment_model: Model for assign_to_parent_cluster (high-volume, cost-sensitive)
        refinement_model: Model for refine_parent_cluster (high-volume, cost-sensitive)
        target_levels: Number of hierarchy levels to create
        merge_ratio: Target ratio for merging (0.2 = 200->40->8)
        neighborhood_size: Average clusters per neighborhood
        concurrency: Max concurrent LLM requests
        rpm: Optional rate limit (requests per minute)

    Returns:
        Tuple of (hierarchy_run_id, hierarchy_list)
        where hierarchy_list contains dicts with hierarchy metadata
    """
    
    logger.info(
        f"Starting hierarchical merging: {len(base_clusters)} base clusters "
        f"→ {target_levels} levels with {merge_ratio:.1%} merge ratio per level"
    )
    logger.info(
        f"Configuration: neighborhood_size={neighborhood_size}, "
        f"concurrency={concurrency}, rpm={rpm or 'unlimited'}"
    )

    # Create hierarchy run ID
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    hierarchy_run_id = f"hier-{run_id}-{timestamp}"

    # Initialize embedding model
    logger.info(f"Loading embedding model: {embedding_provider}/{embedding_model}")
    embedder = EmbeddingGenerator(
        model_name=embedding_model,
        provider=embedding_provider,
        concurrency=100
    )

    # Initialize LLM clients for different tasks
    logger.info(f"Initializing LLM clients:")
    logger.info(f"  Category generation: {llm_provider}/{category_model}")
    logger.info(f"  Deduplication: {llm_provider}/{dedup_model}")
    logger.info(f"  Assignment: {llm_provider}/{assignment_model}")
    logger.info(f"  Refinement: {llm_provider}/{refinement_model}")

    # Build full model strings for instructor
    category_client = instructor.from_provider(f"{llm_provider}/{category_model}", async_client=True)
    dedup_client = instructor.from_provider(f"{llm_provider}/{dedup_model}", async_client=True)
    assignment_client = instructor.from_provider(f"{llm_provider}/{assignment_model}", async_client=True)
    refinement_client = instructor.from_provider(f"{llm_provider}/{refinement_model}", async_client=True)


    # Initialize hierarchy storage
    # Level 0: Add base clusters as leaves
    hierarchy = [
        {
            "hierarchy_run_id": hierarchy_run_id,
            "run_id": run_id,
            "cluster_id": cluster.cluster_id,
            "parent_cluster_id": None,
            "level": 0,
            "children_ids": [],
            "title": cluster.title,
            "description": cluster.description,
        } for cluster in base_clusters
    ]
    logger.info(f"Initialized hierarchy: {hierarchy_run_id}")
    
    # Current level clusters (start with base)
    current_clusters = base_clusters
    current_level = 0
    next_cluster_id = max(c.cluster_id for c in base_clusters) + 1
    logger.debug(f"Next available cluster ID: {next_cluster_id}")

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
            actual_reduction = (1 - n_target / n_current) * 100

            logger.info(
                f"Level {current_level} -> {current_level + 1}: "
                f"{n_current} -> {n_target} clusters ({actual_reduction:.1f}% reduction)"
            )

            task = progress.add_task(
                f"Merging level {current_level} -> {current_level + 1}",
                total=n_current
            )

            # Step 1: Embed cluster representative queries
            logger.info(f"Step 1/4: Embedding {n_current} cluster representative queries...")
            ids = [c.cluster_id for c in current_clusters]
            
            # Get representative queries for each cluster
            texts = []
            for cluster in current_clusters:
                cluster_reps = select_representative_queries_from_db(db, run_id, [cluster.cluster_id], max_queries=5)
                if cluster_reps:
                    # Use representative queries for embedding
                    cluster_text = " ".join(cluster_reps)
                else:
                    # Fallback to title and description if no representative queries
                    cluster_text = f"{cluster.title}: {cluster.description}"
                texts.append(cluster_text)
            
            time = time.time()
            embeddings = await embedder.generate_embeddings_async(texts, show_progress=False)
            time_end = time.time()
            logger.info(f"Time taken to embed {n_current} clusters at level {current_level}: {time_end - time:.2f} seconds")

            # Step 2: Create neighborhoods
            n_neighborhoods = max(n_current // neighborhood_size, 1)
            avg_neighborhood_size = n_current / n_neighborhoods
            logger.info(
                f"Step 2/4: Creating {n_neighborhoods} neighborhoods "
                f"(avg size: {avg_neighborhood_size:.1f} clusters)"
            )
            
            neighborhood_labels = create_neighborhoods(ids, embeddings, n_neighborhoods) 

            # Step 3: Generate higher-level categories per neighborhood (parallelized)
            logger.info(f"Step 3/4: Generating categories from {n_neighborhoods} neighborhoods...")
            all_candidates: List[str] = []

            # Prepare neighborhood data
            neighborhood_data = []
            for neigh_id in range(n_neighborhoods):
                neigh_clusters = [
                    current_clusters[i]
                    for i, (_, label) in enumerate(neighborhood_labels)
                    if label == neigh_id
                ]
                target_cat = int(len(neigh_clusters) * merge_ratio)
                neighborhood_data.append((neigh_id, neigh_clusters, target_cat))
                logger.debug(f"  Neighborhood {neigh_id + 1}/{n_neighborhoods}: {len(neigh_clusters)} clusters → {target_cat} categories")

            # Worker function for parallel neighborhood processing
            async def process_neighborhood(neigh_id: int, neigh_clusters: List[BaseCluster], target_cat: int):
                """Process a single neighborhood and return results."""
                async with semaphore:
                    try:
                        logger.info(f"  Neighborhood {neigh_id + 1}: Attempting to generate {target_cat} categories for {len(neigh_clusters)} clusters")
                        categories = await generate_neighborhood_categories(
                            category_client, neigh_clusters, target_count=target_cat
                        )
                        logger.info(f"  Neighborhood {neigh_id + 1}: Generated {len(categories.categories)} categories")
                        for cat_name in categories.categories:
                            logger.info(f"    • {cat_name}")
                        
                        return neigh_id, categories.categories
                    except Exception as e:
                        logger.error(f"  Neighborhood {neigh_id + 1}: Failed to process: {e}")
                        raise

            # Run all neighborhoods in parallel with concurrency control
            semaphore = anyio.Semaphore(concurrency)
            
            # Execute all neighborhoods in parallel
            tasks = [
                process_neighborhood(neigh_id, neigh_clusters, target_cat)
                for neigh_id, neigh_clusters, target_cat in neighborhood_data
            ]
            
            logger.info(f"Running {len(tasks)} neighborhoods in parallel")
            results = await asyncio.gather(*tasks)
            
            # Sort results by neighborhood ID to maintain order
            results.sort(key=lambda x: x[0])
            
            # Collect all candidates
            for neigh_id, categories in results:
                all_candidates.extend(categories)

            logger.info(
                f"  Generated {len(all_candidates)} total candidates "
                f"({len(all_candidates) / n_neighborhoods:.1f} per neighborhood)"
            )
            # Step 4: Deduplicate globally (with validation)
            logger.info(
                f"Step 4/4: Deduplicating {len(all_candidates)} candidates "
                f"→ ~{n_target} parent clusters..."
            )

            # Calculate min/max bounds for parent count to prevent over-merging
            # We want roughly n_target, but allow up to 50% more to preserve specificity
            # Changed from 1.3x to 1.5x to allow more specific categories
            min_parents = max(n_target, int(n_current * merge_ratio * 0.85))
            max_parents = int(n_target * 1.5)  # Increased from 1.3 to 1.5
            logger.debug(f"  Target parent range: {min_parents}-{max_parents} (aiming for {n_target}, prefer higher end)")

            dedup_result = await deduplicate_cluster_names(
                dedup_client, all_candidates, n_target
            )

            parent_names = dedup_result.clusters

            # Validate deduplication didn't over-merge
            if len(parent_names) < min_parents:
                error_msg = (
                    f"OVER-MERGING DETECTED: Deduplication created {len(parent_names)} parents "
                    f"from {len(all_candidates)} candidates (expected {min_parents}-{max_parents}). "
                    f"This indicates overly aggressive merging that will create generic categories.\n\n"
                    f"Recommendations:\n"
                    f"  1. Reduce merge_ratio (current: {merge_ratio}, try: {merge_ratio + 0.05:.2f})\n"
                    f"  2. Reduce neighborhood_size (current: {neighborhood_size}, try: {max(15, neighborhood_size - 10)})\n"
                    f"  3. Use better LLM (current: {category_model}, try: claude-sonnet-4-5-20250929)\n\n"
                    f"Re-run with: --merge-ratio {merge_ratio + 0.05:.2f} --neighborhood-size {max(15, neighborhood_size - 10)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(
                f"  Deduplicated: {len(all_candidates)} candidates → {len(parent_names)} unique parents "
                f"(removed {len(all_candidates) - len(parent_names)} duplicates)"
            )
            # Show parent names at debug level (can be a long list)
            logger.debug("  Final parent cluster names:")
            for i, parent_name in enumerate(parent_names, 1):
                logger.debug(f"    {i}. {parent_name}")

            # Step 5: Assign children to parents (parallelized)
            # Use maximum concurrency for assignment step (embarrassingly parallel)
            assignment_concurrency = min(concurrency * 5, 50)  # 5x higher, max 50
            logger.info(f"Assigning {n_current} children to {len(parent_names)} parents (concurrency={assignment_concurrency})...")
            parent_children = {name: [] for name in parent_names}
            assignment_errors = []
            semaphore = anyio.Semaphore(assignment_concurrency)
            assignments = []

            # Worker function for parallel assignments
            async def assign_worker(cluster):
                async with semaphore:
                    try:
                        # Remove rate limiting for assignment step to maximize throughput
                        assignment = await assign_to_parent_cluster(assignment_client, cluster, parent_names)
                        assignments.append((cluster, assignment))
                    except Exception as e:
                        logger.error(
                            f"Failed to assign cluster {cluster.cluster_id} "
                            f"('{cluster.title[:60]}...'): {e}"
                        )
                        raise
                    finally:
                        progress.update(task, advance=1)

            # Run all assignments in parallel
            async with anyio.create_task_group() as tg:
                for cluster in current_clusters:
                    tg.start_soon(assign_worker, cluster)

            # Validate and organize assignments
            for cluster, assignment in assignments:
                if assignment.assigned_cluster not in parent_children:
                    error_msg = (
                        f"Invalid assignment for cluster {cluster.cluster_id} ('{cluster.title}'): "
                        f"LLM returned '{assignment.assigned_cluster}' which is not in parent list. "
                        f"Valid parents: {parent_names}"
                    )
                    logger.error(error_msg)
                    assignment_errors.append({
                        "cluster_id": cluster.cluster_id,
                        "cluster_title": cluster.title,
                        "invalid_parent": assignment.assigned_cluster,
                        "valid_parents": parent_names
                    })
                    # Assign to first parent as fallback
                    parent_children[parent_names[0]].append(cluster.cluster_id)
                    logger.warning(f"Falling back to parent '{parent_names[0]}' for cluster {cluster.cluster_id}")
                else:
                    parent_children[assignment.assigned_cluster].append(cluster.cluster_id)

            # Report assignment validation results
            valid_count = len(current_clusters) - len(assignment_errors)
            if assignment_errors:
                error_rate = len(assignment_errors) / len(current_clusters) * 100
                logger.warning(
                    f"Assignment validation: {valid_count}/{len(current_clusters)} valid "
                    f"({error_rate:.1f}% error rate)"
                )
                logger.warning("Sample errors (showing first 3):")
                for err in assignment_errors[:3]:
                    logger.warning(f"  Cluster {err['cluster_id']}: invalid parent '{err['invalid_parent']}'")
            else:
                logger.info(f"All {valid_count} cluster assignments validated successfully")
            
            # Show assignment distribution
            avg_children = sum(len(children) for children in parent_children.values()) / len(parent_names)
            min_children = min(len(children) for children in parent_children.values())
            max_children = max(len(children) for children in parent_children.values())
            logger.info(
                f"  Assignment distribution: avg={avg_children:.1f}, min={min_children}, max={max_children}"
            )
            
            # Show which parents got the most/least assignments
            sorted_parents = sorted(parent_children.items(), key=lambda x: len(x[1]), reverse=True)
            logger.debug("  Top 3 largest parent clusters:")
            for name, children in sorted_parents[:3]:
                logger.debug(f"    '{name}': {len(children)} children")
            if len(sorted_parents) > 3 and min_children < avg_children * 0.5:
                logger.debug("  Smallest parent clusters:")
                for name, children in sorted_parents[-3:]:
                    if len(children) > 0:
                        logger.debug(f"    '{name}': {len(children)} children")

            # Validate that no parent has too many children (indicates overly generic category)
            max_reasonable_children = int(avg_children * 2.0)  # Changed from 2.5x to 2.0x (stricter)
            oversized_parents = [
                (name, len(children))
                for name, children in parent_children.items()
                if len(children) > max_reasonable_children
            ]
            if oversized_parents:
                # Changed from warning to error if too many oversized parents
                oversized_ratio = len(oversized_parents) / len(parent_names)
                severity = "ERROR" if oversized_ratio > 0.2 else "WARNING"

                log_func = logger.error if severity == "ERROR" else logger.warning
                log_func(
                    f"  {severity}: Found {len(oversized_parents)} oversized parent clusters "
                    f"(>{max_reasonable_children} children, avg is {avg_children:.1f}, {oversized_ratio:.1%} of parents):"
                )
                for name, count in oversized_parents[:10]:  # Show top 10
                    log_func(f"    '{name}': {count} children - likely too generic")

                if severity == "ERROR":
                    error_msg = (
                        f"\nOVER-MERGING DETECTED: {oversized_ratio:.1%} of parent clusters are oversized.\n"
                        f"This indicates the hierarchy is creating overly generic catch-all categories.\n\n"
                        f"Recommendations:\n"
                        f"  1. Increase merge_ratio from {merge_ratio:.2f} to {min(0.5, merge_ratio + 0.1):.2f}\n"
                        f"  2. Reduce neighborhood_size from {neighborhood_size} to {max(15, neighborhood_size - 5)}\n"
                        f"  3. Use better LLM: claude-sonnet-4-5-20250929 (better at avoiding generic terms)\n\n"
                        f"Re-run with: --merge-ratio {min(0.5, merge_ratio + 0.1):.2f} --neighborhood-size {max(15, neighborhood_size - 5)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.warning(
                        f"  Consider: (1) increasing merge_ratio to {min(0.5, merge_ratio + 0.05):.2f}, "
                        f"(2) reducing neighborhood_size to {max(15, neighborhood_size - 5)}, "
                        f"or (3) using claude-sonnet-4-5-20250929 for better specificity"
                    )

            # Step 6: Refine parent names based on children (parallelized)
            # Use maximum concurrency for refinement step as well
            refinement_concurrency = min(concurrency * 3, 30)  # 3x higher, max 30
            logger.info(f"Refining {len(parent_names)} parent clusters based on their children (concurrency={refinement_concurrency})...")
            next_level_clusters = []
            refinement_results = []
            refine_semaphore = anyio.Semaphore(refinement_concurrency)

            # Worker function for parallel refinements
            async def refine_worker(parent_name, child_ids):
                if not child_ids:
                    logger.warning(f"Skipping parent '{parent_name}' - has no children assigned")
                    return None

                child_clusters = [
                    c for c in current_clusters if c.cluster_id in child_ids
                ]
                logger.debug(f"Refining '{parent_name[:50]}...' from {len(child_clusters)} children")

                try:
                    async with refine_semaphore:
                        # Remove rate limiting for refinement step to maximize throughput
                        refined = await refine_parent_cluster(refinement_client, child_clusters)

                    logger.info(f"  Refined: {len(child_clusters)} clusters → '{refined.title}'")
                    # Show all child titles at debug level (can be verbose)
                    logger.debug(f"    Children of '{refined.title}':")
                    for i, child_cluster in enumerate(child_clusters, 1):
                        logger.debug(f"      {i}. {child_cluster.title}")

                    refinement_results.append((parent_name, child_ids, child_clusters, refined))
                except Exception as e:
                    logger.error(f"Failed to refine parent '{parent_name[:50]}...': {e}")
                    raise

            # Run all refinements in parallel
            async with anyio.create_task_group() as tg:
                for parent_name in parent_names:
                    child_ids = parent_children[parent_name]
                    tg.start_soon(refine_worker, parent_name, child_ids)

            # Process results and build hierarchy
            for parent_name, child_ids, child_clusters, refined in refinement_results:
                if refined is None:
                    continue

                parent_cluster_id = next_cluster_id
                next_cluster_id += 1

                # Select representative queries from child clusters by querying database
                representative_queries = select_representative_queries_from_db(
                    db, run_id, child_ids, max_queries=10
                )

                # Add to hierarchy
                hierarchy.append({
                    "hierarchy_run_id": hierarchy_run_id,
                    "run_id": run_id,
                    "cluster_id": parent_cluster_id,
                    "parent_cluster_id": None,  # Will be set in next iteration
                    "level": current_level + 1,
                    "children_ids": child_ids,
                    "title": refined.title,
                    "description": refined.summary,
                    "representative_queries": representative_queries
                })

                # Update children's parent_id with validation
                children_found = 0
                for child_id in child_ids:
                    child_found = False
                    for h in hierarchy:
                        if h["cluster_id"] == child_id and h["level"] == current_level:
                            h["parent_cluster_id"] = parent_cluster_id
                            children_found += 1
                            child_found = True
                            break
                    if not child_found:
                        logger.error(f"Child cluster {child_id} not found in hierarchy at level {current_level}")

                # Validate all children were found
                if children_found != len(child_ids):
                    logger.warning(
                        f"Parent cluster {parent_cluster_id} expected {len(child_ids)} children "
                        f"but only found {children_found} in hierarchy"
                    )

                # Add to next level
                next_level_clusters.append({
                    "cluster_id": parent_cluster_id,
                    "title": refined.title,
                    "description": refined.summary,
                    "representative_queries": representative_queries
                })

            # Level complete summary
            logger.info(
                f"Level {current_level} → {current_level + 1} complete: "
                f"created {len(next_level_clusters)} parent clusters"
            )

            # Move to next level - convert dicts back to BaseCluster objects
            current_clusters = [
                BaseCluster(
                    cluster_id=cluster_dict["cluster_id"],
                    title=cluster_dict["title"],
                    description=cluster_dict["description"],
                    representative_queries=cluster_dict.get("representative_queries")
                )
                for cluster_dict in next_level_clusters
            ]
            current_level += 1
            logger.debug(f"Advanced to level {current_level}, next cluster ID will be {next_cluster_id}")

            if len(current_clusters) <= 1:
                logger.info(
                    f"Stopping: reached single top-level cluster at level {current_level} "
                    f"(target was {target_levels} levels)"
                )
                break
            
            logger.info(f"Continuing to level {current_level + 1} with {len(current_clusters)} clusters")

    # Final validation: Check hierarchy integrity
    logger.info("=" * 60)
    logger.info("Validating hierarchy integrity...")
    validation_errors = []
    
    total_nodes = len(hierarchy)
    levels = sorted(set(h["level"] for h in hierarchy))
    logger.debug(f"Hierarchy summary: {total_nodes} total nodes across {len(levels)} levels: {levels}")

    # Check 1: All base clusters have parents (except top level)
    base_clusters_without_parents = [
        h for h in hierarchy
        if h["level"] == 0 and h["parent_cluster_id"] is None
    ]
    if base_clusters_without_parents and current_level > 0:
        error = f"Found {len(base_clusters_without_parents)} leaf clusters without parents"
        validation_errors.append(error)
        logger.error(f"Validation check 1 failed: {error}")

    # Check 2: All parent references are valid
    logger.debug("Check 2: Validating parent references...")
    cluster_ids = {h["cluster_id"] for h in hierarchy}
    invalid_refs = []
    for h in hierarchy:
        if h["parent_cluster_id"] is not None and h["parent_cluster_id"] not in cluster_ids:
            error = f"Cluster {h['cluster_id']} references non-existent parent {h['parent_cluster_id']}"
            validation_errors.append(error)
            invalid_refs.append(error)
    if invalid_refs:
        logger.error(f"Validation check 2 failed: {len(invalid_refs)} invalid parent references")
        for ref in invalid_refs[:3]:  # Show first 3
            logger.error(f"  {ref}")

    # Check 3: Children lists match actual parent assignments
    logger.debug("Check 3: Validating children lists...")
    mismatches = []
    for h in hierarchy:
        if h["children_ids"]:
            actual_children = {
                child["cluster_id"]
                for child in hierarchy
                if child["parent_cluster_id"] == h["cluster_id"]
            }
            expected_children = set(h["children_ids"])
            if actual_children != expected_children:
                error = (
                    f"Cluster {h['cluster_id']} children mismatch: "
                    f"expected {len(expected_children)}, found {len(actual_children)}"
                )
                validation_errors.append(error)
                mismatches.append(error)
    if mismatches:
        logger.error(f"Validation check 3 failed: {len(mismatches)} children list mismatches")
        for mm in mismatches[:3]:  # Show first 3
            logger.error(f"  {mm}")

    # Check 4: Level consistency
    logger.debug("Check 4: Validating level consistency...")
    levels_found = {h["level"] for h in hierarchy}
    expected_levels = set(range(current_level + 1))
    if levels_found != expected_levels:
        error = f"Level gaps detected: expected {expected_levels}, found {levels_found}"
        validation_errors.append(error)
        logger.error(f"Validation check 4 failed: {error}")

    if validation_errors:
        logger.error("=" * 60)
        logger.error(f"VALIDATION FAILED: {len(validation_errors)} errors detected")
        logger.error("=" * 60)
        raise ValueError(f"Hierarchy validation failed: {validation_errors}")

    # Success! Log final statistics
    logger.info("=" * 60)
    logger.info("Hierarchy validation: ALL CHECKS PASSED")
    logger.info(f"Final hierarchy: {len(hierarchy)} nodes across {current_level + 1} levels")
    
    # Count nodes per level
    for level in range(current_level + 1):
        count = sum(1 for h in hierarchy if h["level"] == level)
        logger.info(f"  Level {level}: {count} clusters")
    
    logger.info(f"Hierarchy run ID: {hierarchy_run_id}")
    logger.info("=" * 60)
    
    # Print timing summary
    
    return hierarchy_run_id, hierarchy
