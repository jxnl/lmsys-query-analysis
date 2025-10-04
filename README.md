# LMSYS Query Analysis

**Terminal-based CLI tools for agents to perform comprehensive data analysis on LMSYS queries.**

This project enables systematic investigation of how people use LLM systems through data-driven analysis workflows. Agents can explore the LMSYS-1M dataset to discover patterns, form hypotheses, and generate insights about user behavior, query complexity, and LLM usage trends.

## Goal

Enable terminal-based agents to:

- **Investigate LLM Usage Patterns**: Discover how users interact with different LLM systems
- **Identify Query Clusters**: Group similar queries to find common use cases and patterns
- **Generate Insights**: Use LLM-powered summarization to understand what makes clusters unique
- **Form Hypotheses**: Systematically explore data to develop testable hypotheses about user behavior
- **Navigate Semantically**: Search queries and clusters using natural language

All through a composable CLI workflow: `load → cluster → summarize → search → export`

## Features

- **Data Loading**: Extract first user queries from LMSYS-1M dataset (1M conversations)
- **SQLite Storage**: Efficient indexing and querying with SQLModel
- **Clustering**: MiniBatchKMeans (scales well) and HDBSCAN (density-based)
- **ChromaDB Integration**: Semantic search across queries and cluster summaries
- **LLM Summarization**: Generate titles and descriptions for clusters using any LLM provider
- **Contrastive Analysis**: Highlight what makes each cluster unique vs. neighbors
- **CLI Interface**: Rich terminal UI designed for agent workflows

## Installation

```bash
uv sync
```

## Setup

### 1. HuggingFace Authentication

The LMSYS-1M dataset is gated and requires authentication:

```bash
huggingface-cli login
# Accept terms at: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
```

### 2. LLM API Key (Optional, for summarization)

Set environment variable for your chosen provider:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # For Anthropic Claude
export OPENAI_API_KEY="sk-..."          # For OpenAI GPT
```

## Quick Start

```bash
# 1. Load 10k queries (test dataset)
uv run lmsys load --limit 10000 --use-chroma

# 2. Run clustering with 100 clusters
uv run lmsys cluster --n-clusters 100 --use-chroma

# 3. Generate LLM summaries for all clusters
uv run lmsys summarize <RUN_ID> --use-chroma

# 4. List clusters with titles
uv run lmsys list-clusters <RUN_ID>

# 5. Semantic search
uv run lmsys search "python programming" --search-type clusters
```

## Commands

### Data Loading

```bash
# Load all queries (1M records)
uv run lmsys load

# Load limited number for testing
uv run lmsys load --limit 10000

# Enable ChromaDB for semantic search
uv run lmsys load --limit 10000 --use-chroma
```

**Default database:** `~/.lmsys-query-analysis/queries.db`

### Clustering

```bash
# Run KMeans with 200 clusters (recommended for fine-grained analysis)
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma \
  --embedding-model all-MiniLM-L6-v2 \
  --embedding-provider sentence-transformers \
  --embed-batch-size 64 --mb-batch-size 8192 --chunk-size 10000 \
  --description "Fine-grained clustering"

# Faster with fewer clusters
uv run lmsys cluster kmeans --n-clusters 50 --use-chroma

# HDBSCAN (finds natural clusters; excludes noise)
uv run lmsys cluster hdbscan --use-chroma \
  --embedding-model all-MiniLM-L6-v2 --embedding-provider sentence-transformers \
  --embed-batch-size 64 --chunk-size 10000

# Notes
# - With --use-chroma, clustering reuses existing query embeddings from Chroma and backfills missing ones.
# - Tune throughput with --embed-batch-size, --mb-batch-size, and --chunk-size.
```

### LLM Summarization

The summarization system now supports **multiple summary runs** for the same clustering run, allowing you to:
- Compare different LLM models side-by-side
- Iterate on prompts without losing previous results
- Track which parameters produced the best summaries

Each summary run is uniquely identified by a `summary_run_id` (auto-generated as `summary-{model}-{timestamp}`).

```bash
# Generate titles and descriptions for all clusters
# Auto-generates summary_run_id like: summary-claude-sonnet-4-5-2025-20251004-124530
uv run lmsys summarize <RUN_ID>

# Use different LLM models (creates separate summary runs)
uv run lmsys summarize <RUN_ID> --model "openai/gpt-4"
uv run lmsys summarize <RUN_ID> --model "groq/llama-3.1-8b-instant"

# Custom summary run ID for easy reference
uv run lmsys summarize <RUN_ID> --summary-run-id "claude-v1"

# Summarize specific cluster only
uv run lmsys summarize <RUN_ID> --cluster-id 5

# Adjust number of queries sent to LLM
uv run lmsys summarize <RUN_ID> --max-queries 100

# Speed up with concurrency and optional rate limiting
uv run lmsys summarize <RUN_ID> --concurrency 8 --rpm 60

# Note: --use-chroma flag has been removed. Summaries are stored in SQLite only.
```

### Viewing Results

```bash
# List all clustering runs (ordered by newest first)
uv run lmsys runs

# Show only the most recent run
uv run lmsys runs --latest

# List clusters with LLM-generated titles (shows latest summary run by default)
uv run lmsys list-clusters <RUN_ID>

# View specific summary run
uv run lmsys list-clusters <RUN_ID> --summary-run-id "summary-claude-sonnet-4-5-2025-20251004-124530"

# Limit results
uv run lmsys list-clusters <RUN_ID> --limit 20

# Show example queries per cluster
uv run lmsys list-clusters <RUN_ID> --show-examples 3
```

### Semantic Search

```bash
# Search cluster summaries
uv run lmsys search "python programming" --search-type clusters \
  --run-id <RUN_ID> --embedding-model all-MiniLM-L6-v2

# Search within specific run
uv run lmsys search "machine learning" --search-type clusters --run-id <RUN_ID>

# Search individual queries (if loaded with --use-chroma)
uv run lmsys search "how to build neural network" --search-type queries \
  --n-results 20 --embedding-model all-MiniLM-L6-v2
```

Search uses explicit query embeddings to ensure consistency with stored vectors. Use an embedding model that matches the vectors in Chroma (the model used during load/cluster).

## Architecture

### Database Schema (SQLite + SQLModel)

**queries** - First user query from each conversation
- `id`, `conversation_id` (unique), `model`, `query_text`, `language`
- `extra_metadata` (JSON), `created_at`

**clustering_runs** - Track clustering experiments
- `run_id` (primary key), `algorithm`, `num_clusters`
- `parameters` (JSON), `description`, `created_at`

**query_clusters** - Map queries to clusters per run
- `run_id`, `query_id`, `cluster_id`, `confidence_score`

**cluster_summaries** - LLM-generated summaries (supports multiple summary runs)
- `run_id`, `cluster_id`, `summary_run_id` (unique per summarization)
- `title`, `description`, `summary`, `num_queries`
- `representative_queries` (JSON)
- `model` (LLM used), `parameters` (JSON - summarization settings)
- `generated_at`

### ChromaDB Collections

**queries** - All user queries with embeddings (optional)
- ID format: `query_{id}`
- Metadata: model, language, conversation_id

**cluster_summaries** - Cluster titles + descriptions with embeddings
- ID format: `cluster_{run_id}_{cluster_id}`
- Metadata: run_id, cluster_id, title, description, num_queries
- Document: Combined title + description for semantic search

## Development

### Run Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src/lmsys_query_analysis --cov-report=term

# Specific test file
uv run pytest tests/test_models.py -v
```

### Smoke Test

Run a minimal end-to-end check using a temporary SQLite DB and Chroma directory:

```bash
bash smoketest.sh                 # full run (requires HF + model downloads)
SMOKE_LIMIT=500 bash smoketest.sh # adjust dataset size
SMOKE_CLUSTERS=8 bash smoketest.sh
SMOKE_OFFLINE=1 bash smoketest.sh # seed minimal data if load fails (offline)
```

The smoke test will: load or seed data, run clustering, list/inspect clusters, export CSV/JSON, and try semantic search when Chroma data is available.

### Logging

All commands support verbose logging:

```bash
uv run lmsys -v cluster --n-clusters 200
```

### Project Structure

```
src/lmsys_query_analysis/
├── cli/
│   └── main.py              # Typer CLI commands
├── db/
│   ├── models.py            # SQLModel schemas
│   ├── connection.py        # Database manager
│   ├── loader.py            # LMSYS dataset loader
│   └── chroma.py            # ChromaDB manager
└── clustering/
    ├── embeddings.py        # Sentence-transformers wrapper
    ├── kmeans.py            # MiniBatchKMeans clustering (streaming)
    └── summarizer.py        # LLM summarization with instructor

tests/                       # Test suite with 20+ tests
```

Additional files
- `src/lmsys_query_analysis/clustering/hdbscan_clustering.py` — HDBSCAN clustering
- `src/lmsys_query_analysis/utils/logging.py` — Rich-backed logging setup

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY` - For Claude models
- `OPENAI_API_KEY` - For GPT models
- `GROQ_API_KEY` - For Groq models
- `HUGGINGFACE_TOKEN` - For dataset access

### Default Paths

- SQLite DB: `~/.lmsys-query-analysis/queries.db`
- ChromaDB: `~/.lmsys-query-analysis/chroma/`

Override with `--db-path` and `--chroma-path` flags.

## Examples

### Full Workflow

```bash
# 1. Load 50k queries with ChromaDB
uv run lmsys load --limit 50000 --use-chroma

# 2. Run fine-grained clustering
uv run lmsys cluster --n-clusters 200 --use-chroma \
  --description "50k queries, 200 clusters"

# 3. Get the run ID from output, then generate summaries
uv run lmsys summarize kmeans-200-20251003-123456 --use-chroma

# 4. Explore clusters
uv run lmsys list-clusters kmeans-200-20251003-123456

# 5. Find Python-related clusters
uv run lmsys search "python programming tutorials" \
  --search-type clusters --run-id kmeans-200-20251003-123456

# 6. View all runs
uv run lmsys runs
```

### Using Different LLM Providers

```bash
# Claude (Anthropic) - Fast and cheap
uv run lmsys summarize <RUN_ID> \
  --model "anthropic/claude-3-haiku-20240307" --use-chroma

# GPT-4 (OpenAI) - Highest quality
uv run lmsys summarize <RUN_ID> \
  --model "openai/gpt-4" --use-chroma

# Llama via Groq - Fast inference
uv run lmsys summarize <RUN_ID> \
  --model "groq/llama-3.1-8b-instant" --use-chroma

# Local with Ollama
uv run lmsys summarize <RUN_ID> \
  --model "ollama/llama3" --use-chroma
```

## Troubleshooting

### Schema Migration

If you get schema errors after updating:

```bash
# Option 1: Delete and recreate (loses data)
rm ~/.lmsys-query-analysis/queries.db
rm -rf ~/.lmsys-query-analysis/chroma
# Then re-run load and cluster commands

# Option 2: Check current schema
sqlite3 ~/.lmsys-query-analysis/queries.db ".schema cluster_summaries"

# Note: The cluster_summaries table now requires summary_run_id field.
# If upgrading from older version, recreate the database.
```

### ChromaDB Not Found

Make sure you used `--use-chroma` when loading data and clustering:

```bash
uv run lmsys load --limit 1000 --use-chroma
uv run lmsys cluster --n-clusters 50 --use-chroma
```

## License

MIT
