# LMSYS Query Analysis

Query analysis and clustering tool for the LMSYS-1M conversational dataset with semantic search and LLM-powered summarization.

## Features

- **Data Loading**: Extract first user queries from LMSYS-1M dataset (1M conversations)
- **SQLite Storage**: Efficient indexing and querying with SQLModel
- **Clustering**: MiniBatchKMeans (scales well) and HDBSCAN (density-based)
- **ChromaDB Integration**: Semantic search across queries and cluster summaries
- **LLM Summarization**: Generate titles and descriptions for clusters using any LLM provider
- **CLI Interface**: Rich terminal UI with tables and progress bars

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
