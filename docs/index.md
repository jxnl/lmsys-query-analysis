# LMSYS Query Analysis

**Terminal-based CLI tools for agents to perform comprehensive data analysis on LMSYS queries.**

This project enables systematic investigation of how people use LLM systems through data-driven analysis workflows. Agents can explore the LMSYS-1M dataset to discover patterns, form hypotheses, and generate insights about user behavior, query complexity, and LLM usage trends.

## For Agents

This toolkit is designed for **terminal-based agents** to:

- **Investigate LLM Usage Patterns**: Discover how users interact with different LLM systems
- **Identify Query Clusters**: Group similar queries to find common use cases and patterns
- **Generate Insights**: Use LLM-powered summarization to understand what makes clusters unique
- **Form Hypotheses**: Systematically explore data to develop testable hypotheses about user behavior
- **Navigate Semantically**: Search queries and clusters using natural language
- **Export Findings**: Save analysis results for further investigation

All through a composable CLI workflow: `load → cluster → summarize → search → export`

## Features

- **Data Loading**: Extract first user queries from LMSYS-1M dataset (1M conversations)
- **SQLite Storage**: Efficient indexing and querying with SQLModel
- **Clustering**: MiniBatchKMeans (scales well) and HDBSCAN (density-based)
- **ChromaDB Integration**: Semantic search across queries and cluster summaries
- **LLM Summarization**: Generate titles and descriptions for clusters using any LLM provider
- **Contrastive Analysis**: Highlight what makes each cluster unique vs. neighbors
- **CLI Interface**: Rich terminal UI designed for agent workflows

## Quick Start

```bash
# Install dependencies
uv sync --group docs

# Load sample queries
uv run lmsys load --limit 10000 --use-chroma

# Run clustering
uv run lmsys cluster kmeans --n-clusters 100 --use-chroma

# Generate summaries
uv run lmsys summarize <RUN_ID> --use-chroma

# List clusters
uv run lmsys list-clusters <RUN_ID>
```

## Architecture

The project consists of:

- **CLI**: Typer-based command-line interface with rich output
- **Database**: SQLite with SQLModel for queries and clustering results
- **ChromaDB**: Vector database for semantic search
- **Clustering**: KMeans and HDBSCAN algorithms
- **LLM Integration**: Instructor-based summarization with multiple providers

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [CLI Reference](cli/overview.md)
- [API Documentation](api/models.md)
