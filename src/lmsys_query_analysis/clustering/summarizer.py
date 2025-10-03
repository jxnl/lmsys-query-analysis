"""LLM-based cluster summarization using instructor."""
from typing import List, Optional
import instructor
from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class ClusterSummaryResponse(BaseModel):
    """Structured response from LLM for cluster summarization."""
    title: str = Field(..., description="Short title (5-10 words) capturing the main topic")
    description: str = Field(..., description="2-3 sentences explaining the cluster theme")


class ClusterSummarizer:
    """Generate titles and descriptions for query clusters using LLMs via instructor."""

    def __init__(
        self,
        model: str = "anthropic/claude-3-haiku-20240307",
        api_key: Optional[str] = None,
    ):
        """Initialize the summarizer.

        Args:
            model: Model in format "provider/model_name" (e.g., "anthropic/claude-3-haiku-20240307",
                   "openai/gpt-4", "groq/llama-3.1-8b-instant")
            api_key: API key for the provider (or set env var ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
        """
        self.model = model

        # Initialize instructor client with provider
        if api_key:
            self.client = instructor.from_provider(model, api_key=api_key)
        else:
            self.client = instructor.from_provider(model)

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
        # Sample queries if too many
        if len(cluster_queries) > max_queries:
            # Take some from start, some from middle, some from end for diversity
            sampled = (
                cluster_queries[:max_queries//3] +
                cluster_queries[len(cluster_queries)//2 - max_queries//6 : len(cluster_queries)//2 + max_queries//6] +
                cluster_queries[-max_queries//3:]
            )
        else:
            sampled = cluster_queries

        # Build prompt
        queries_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sampled))

        prompt = f"""You are analyzing a cluster of user queries from a conversational AI dataset.

Cluster ID: {cluster_id}
Total queries in cluster: {len(cluster_queries)}
Sample queries shown: {len(sampled)}

QUERIES:
{queries_text}

Analyze these queries and provide:
1. A SHORT TITLE (5-10 words) that captures the main topic or theme
2. A DESCRIPTION (2-3 sentences) explaining what types of questions/requests are in this cluster

Be specific and descriptive. Focus on what makes this cluster unique."""

        try:
            # Call LLM with structured output
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=ClusterSummaryResponse,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            return {
                "title": response.title,
                "description": response.description,
                "sample_queries": sampled[:10],  # Store top 10 for reference
            }

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate summary for cluster {cluster_id}: {e}[/yellow]")
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
    ) -> dict[int, dict]:
        """Generate summaries for multiple clusters.

        Args:
            clusters_data: List of (cluster_id, queries) tuples
            max_queries: Max queries per cluster to send to LLM

        Returns:
            Dict mapping cluster_id to summary dict
        """
        results = {}
        total = len(clusters_data)

        console.print(f"[cyan]Generating LLM summaries for {total} clusters using {self.model}...[/cyan]")

        for i, (cluster_id, queries) in enumerate(clusters_data, 1):
            console.print(f"[yellow]Processing cluster {cluster_id} ({i}/{total})...[/yellow]")

            summary = self.generate_cluster_summary(
                cluster_queries=queries,
                cluster_id=cluster_id,
                max_queries=max_queries,
            )
            results[cluster_id] = summary

        return results
