# LMSYS Query Analysis

**Terminal-based CLI tools for agents to perform comprehensive data analysis on conversational systems.**

This project enables systematic investigation of how people use LLM systems through data-driven analysis workflows. Agents can explore the LMSYS-1M dataset to discover patterns, form hypotheses, and generate insights about user behavior, query complexity, and LLM usage trends.

## Vision: Autonomous AI Research Agent for Conversational System Analysis

**Long-term vision**: Every company building AI systems (chatbots, agents, RAG systems, voice assistants) uploads their interaction logs to our platform. An autonomous AI agent with specialized tools explores the data overnight and discovers engineering fixes, prompt improvements, and product opportunities the team didn't know existed.

**Current implementation**: This repository provides the foundational tool library and CLI interface that enables agents to perform autonomous data analysis on conversational datasets. Starting with LMSYS-1M query analysis, the tools demonstrate how agents can explore data, form hypotheses, and generate actionable insights without pre-defined workflows.

## Core Capabilities

The CLI implements specialized tools across multiple categories:

**1. Data Loading Tools**
- Load datasets from Hugging Face or custom sources
- Generate and backfill embeddings for semantic analysis
- Manage dataset lifecycle (status checks, clearing)

**2. Clustering Tools**
- Run unsupervised clustering (KMeans, HDBSCAN) to discover behavioral patterns
- List and compare clustering runs with different parameters
- Get detailed cluster statistics and metadata
- Identify outliers and anomalies automatically

**3. Summarization Tools**
- Generate LLM-powered cluster summaries and descriptions
- Create multiple summary runs to compare different models or prompts
- Extract representative queries and key themes
- Support contrastive analysis (what makes each cluster unique)

**4. Hierarchy Tools**
- Build multi-level cluster taxonomies using Anthropic Clio methodology
- Navigate hierarchical structures from high-level themes to specific patterns
- Organize hundreds of clusters into manageable categories

**5. Search & Analysis Tools**
- Semantic search across queries and clusters
- Filter and aggregate by metadata dimensions
- Export data for external analysis
- Compare time periods and detect trends

**6. Curation Tools**
- Edit cluster assignments and metadata (move, rename, merge, split, tag)
- Track edit history and audit trails
- Flag clusters for review and quality annotation
- Manage orphaned queries and cleanup operations

All through a composable CLI workflow: `load → cluster → summarize → merge-clusters → search → inspect → edit → export`

## Features

- **Data Loading**: Extract first user queries from LMSYS-1M dataset (1M conversations)
- **SQLite Storage**: Efficient indexing and querying with SQLModel
- **Clustering**: MiniBatchKMeans (scales well) and HDBSCAN (density-based)
- **Hierarchical Organization**: LLM-driven merging to create multi-level cluster hierarchies (following Anthropic's Clio methodology)
- **ChromaDB Integration**: Semantic search across queries and cluster summaries
- **LLM Summarization**: Generate titles and descriptions for clusters using any LLM provider
- **Contrastive Analysis**: Highlight what makes each cluster unique vs. neighbors
- **CLI Interface**: Rich terminal UI designed for agent workflows
- **Web Interface**: Interactive Next.js viewer for exploring clustering results
- **Curation Tools**: Edit cluster assignments, track changes, and manage metadata

## Quick Start

```bash
# Install dependencies
uv sync

# Load sample queries with embeddings
uv run lmsys load --limit 10000 --use-chroma

# Run clustering
uv run lmsys cluster kmeans --n-clusters 100 --use-chroma

# Generate summaries
uv run lmsys summarize <RUN_ID> --alias "v1" --use-chroma

# Build hierarchy (essential for navigation)
uv run lmsys merge-clusters <RUN_ID>

# List clusters
uv run lmsys list-clusters <RUN_ID>

# Start web interface
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to explore your clustering results.

## Architecture

The project consists of:

- **CLI**: Typer-based command-line interface with rich output
- **Database**: SQLite with SQLModel for queries and clustering results
- **ChromaDB**: Vector database for semantic search
- **Clustering**: KMeans and HDBSCAN algorithms
- **Hierarchy**: LLM-driven hierarchical merging using Anthropic Clio methodology
- **LLM Integration**: Instructor-based summarization with multiple providers
- **Web Interface**: Next.js viewer with Drizzle ORM and ShadCN UI
- **Curation**: Cluster editing and metadata management tools

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [CLI Reference](cli/overview.md)
- [API Documentation](api/models.md)
