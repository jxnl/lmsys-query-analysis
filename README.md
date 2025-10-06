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

All through a composable CLI workflow: `load → cluster → summarize → merge-clusters → search → export`

## Features

- **Data Loading**: Extract first user queries from LMSYS-1M dataset (1M conversations)
- **SQLite Storage**: Efficient indexing and querying with SQLModel
- **Clustering**: MiniBatchKMeans (scales well) and HDBSCAN (density-based)
- **Hierarchical Organization**: LLM-driven merging to create multi-level cluster hierarchies (following Anthropic's Clio methodology)
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

### 2. LLM / Embedding API Keys (Optional)

Set environment variable for your chosen provider:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # For Anthropic Claude
export OPENAI_API_KEY="sk-..."          # For OpenAI GPT
export COHERE_API_KEY="sk-cohere-..."   # For Cohere Embed v4 / Chat
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

### Quick 5k End-to-End (Local, no API keys)

```bash
# Load 5k with local sentence-transformers embeddings
uv run lmsys load --limit 5000 --use-chroma \
  --embedding-provider sentence-transformers --embedding-model all-MiniLM-L6-v2

# Cluster into 50 groups and write centroids to Chroma
uv run lmsys cluster kmeans --n-clusters 50 --use-chroma \
  --embedding-provider sentence-transformers --embedding-model all-MiniLM-L6-v2

# Get latest run_id
uv run lmsys runs --latest

# Generate summaries
lmsys summarize <RUN_ID> --alias v1
--use-chroma --max-queries 80 --concurrency 6

# Merge clusters
uv run lmsys merge-clusters <RUN_ID> --target-levels 3 --merge-ratio 0.2

# Cluster discovery (JSON)
uv run lmsys search-cluster "vector databases" --run-id <RUN_ID> --json | jq .

# Query discovery conditioned on found clusters (JSON)
uv run lmsys search "hybrid search" --run-id <RUN_ID> \
  --within-clusters "vector databases" --top-clusters 5 \
  --n-results 50 --facets cluster --json | jq .

# Verify Chroma↔SQLite sync for the run
uv run lmsys verify sync <RUN_ID> --json | jq .
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

# Use Cohere embed-v4.0 for embeddings (Matryoshka 256 by default)
uv run lmsys load --limit 10000 --use-chroma \
  --embedding-provider cohere --embedding-model embed-v4.0
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma \
  --embedding-provider cohere --embedding-model embed-v4.0
```

### LLM Summarization

The summarization system now supports **multiple summary runs** for the same clustering run, allowing you to:
- Compare different LLM models side-by-side
- Iterate on prompts without losing previous results
- Track which parameters produced the best summaries

Each summary run is uniquely identified by a `summary_run_id` (auto-generated as `summary-{model}-{timestamp}`) and can have a friendly `alias` for easy reference.

```bash
# Generate titles and descriptions with a friendly alias
uv run lmsys summarize <RUN_ID> --alias "claude-v1"

# Use different LLM models (creates separate summary runs)
uv run lmsys summarize <RUN_ID> --model "openai/gpt-4" --alias "gpt4-test"
uv run lmsys summarize <RUN_ID> --model "groq/llama-3.1-8b-instant" --alias "llama-fast"

# Custom summary run ID for easy reference
uv run lmsys summarize <RUN_ID> --summary-run-id "claude-v1" --alias "my-best"

# Summarize specific cluster only
uv run lmsys summarize <RUN_ID> --cluster-id 5

# Adjust number of queries sent to LLM
uv run lmsys summarize <RUN_ID> --max-queries 100

# Speed up with concurrency and optional rate limiting
uv run lmsys summarize <RUN_ID> --concurrency 8 --rpm 60

# Notes:
# - With --use-chroma, summaries are also embedded and written to Chroma using the
#   same embedding model/provider recorded on the clustering run.
```

### Hierarchical Organization

After summarizing clusters, you can organize them into a multi-level hierarchy using LLM-driven merging (following Anthropic's Clio methodology):

```bash
# Create 3-level hierarchy from 100 base clusters
# Level 0: 100 leaf clusters → Level 1: 20 parents → Level 2: 4 top categories
uv run lmsys merge-clusters kmeans-100-20251004-170442 \
  --target-levels 3 \
  --merge-ratio 0.2

# Customize hierarchy parameters
uv run lmsys merge-clusters <RUN_ID> \
  --target-levels 2 \              # Number of hierarchy levels (2 = one merge)
  --merge-ratio 0.5 \              # Merge aggressiveness (0.5 = 100->50->25)
  --llm-provider anthropic \       # Provider for merging (anthropic/openai/groq)
  --llm-model claude-sonnet-4-5-20250929 \
  --concurrency 8 \                # Parallel LLM requests
  --neighborhood-size 40           # Clusters per LLM context (Clio default)

# Use faster/cheaper models for experimentation
uv run lmsys merge-clusters <RUN_ID> \
  --llm-provider openai --llm-model gpt-4o-mini

# Example: 200 clusters → 40 → 8 top-level categories
uv run lmsys merge-clusters kmeans-200-20251004-005043 \
  --target-levels 3 --merge-ratio 0.2
```

**How it works:**
1. **Neighborhood Formation**: Groups similar clusters using embeddings (manageable LLM context)
2. **Category Generation**: LLM proposes broader category names for each neighborhood
3. **Deduplication**: Merges similar categories globally to create distinct parents
4. **Assignment**: LLM assigns each child cluster to best-fit parent using semantic matching
5. **Refinement**: LLM refines parent names based on actual assigned children
6. **Iteration**: Repeats process for multiple levels

**Output:**
- Hierarchies stored in `cluster_hierarchies` table with parent-child relationships
- Navigate from broad categories → specific topics
- Hierarchy ID format: `hier-<run_id>-<timestamp>`

**Example Hierarchy** (20 leaf clusters → 10 parent categories):

```
Multilingual User Engagement and Assistance (4 children)
├── German Language User Engagement
├── Multilingual Assistance and Queries
├── Chinese Language Interaction
└── Spanish and Portuguese Language Interactions

Python Problem Solving and Task Descriptions (2 children)
├── Python Programming Task Descriptions
└── Technical Problem Solving Queries

Business Management and Academic Inquiry Strategies (2 children)
├── Business and Academic Inquiry Responses
└── Business Management and Consulting Practices

US Sanction Analysis and Legal Entity Recognition (2 children)
├── Detailed US Sanction Types
└── Named Entity Recognition for Legal Statements
```

**Key Benefits:**
- **Semantic grouping**: LLM identifies thematic relationships (language, domain, task type)
- **Multi-perspective navigation**: Same content organized by different dimensions
- **Content isolation**: Sensitive clusters automatically separated for moderation
- **Flexible granularity**: 1:1 mappings indicate optimal abstraction level

### Viewing Results

```bash
# List all clustering runs (ordered by newest first)
uv run lmsys runs

# Show only the most recent run
uv run lmsys runs --latest

# List clusters with LLM-generated titles (shows latest summary run by default)
uv run lmsys list-clusters <RUN_ID>

# View specific summary run by alias (easier!)
uv run lmsys list-clusters <RUN_ID> --alias "claude-v1"

# Or by summary run ID
uv run lmsys list-clusters <RUN_ID> --summary-run-id "summary-claude-sonnet-4-5-2025-20251004-124530"

# Limit results
uv run lmsys list-clusters <RUN_ID> --limit 20

# Show example queries per cluster
uv run lmsys list-clusters <RUN_ID> --show-examples 3
```

### Semantic Search (Clusters + Queries)

Two high-level commands with JSON-friendly outputs for piping into `jq`.

```bash
# Cluster discovery (summaries)
uv run lmsys search-cluster "vector databases" --run-id <RUN_ID> --top-k 20 --json | jq .

# Query discovery with semantic conditioning
uv run lmsys search "hybrid search" --run-id <RUN_ID> \
  --within-clusters "vector databases" --top-clusters 5 \
  --n-results 50 --facets cluster --json | jq .

# Grouped counts and additional facets
uv run lmsys search "vector" --run-id <RUN_ID> --by cluster --json | jq .
uv run lmsys search "rag" --run-id <RUN_ID> --facets language,model --json | jq .

# Direct cluster filters
uv run lmsys search "sql" --run-id <RUN_ID> --cluster-ids 12,27,44 --json | jq .
```

Notes
- With `--run-id`, the CLI resolves the correct embedding provider/model/dimension from the run to prevent cross‑space mix‑ups.
- Queries are always filtered by `query_clusters` in SQLite; Chroma is used for retrieval only. Cluster facets include `meta.title` when `--run-id` is set.

JSON output shapes
- search-cluster:
  - `{ text, run_id, results: [{ cluster_id, distance, title, description, num_queries }] }`
- search (queries):
  - `{ text, run_id, applied_clusters: [{ cluster_id, title, description, num_queries, distance }], results: [{ query_id, distance, snippet, model, language, cluster_id }], facets: { clusters: [{ cluster_id, count, meta: { title } }], language: [{ key, count }], model: [{ key, count }] } }`

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
- `alias` (friendly name like "claude-v1", "gpt4-best")
- `title`, `description`, `summary`, `num_queries`
- `representative_queries` (JSON)
- `model` (LLM used), `parameters` (JSON - summarization settings)
- `generated_at`

**cluster_hierarchies** - Multi-level cluster hierarchies (Clio-style organization)
- `hierarchy_run_id` (unique per hierarchy), `run_id`, `cluster_id`
- `parent_cluster_id` (null for top level), `level` (0=leaf, 1=first merge, etc.)
- `children_ids` (JSON array), `title`, `description`
- `created_at`

### IDs and Provenance

- `run_id` (Clustering Run):
  - What: Unique ID for one clustering experiment (e.g., `kmeans-200-20251004-170442`).
  - Stores: Algorithm, parameters, and the embedding space (`embedding_provider`, `embedding_model`, and dimension) in `clustering_runs.parameters`.
  - Used for: Filtering query→cluster assignments, resolving the correct Chroma collections and embedding space, listing clusters, summarizing, and searching.
  - Where: Primary key in `clustering_runs`; included in Chroma summaries metadata (`run_id`, `cluster_id`).

- `summary_run_id` (Summary Pass) and `alias`:
  - What: Identifies one LLM summarization pass over a run’s clusters; multiple can exist per run (e.g., different models or prompts). Auto-generated as `summary-<model>-<timestamp>` if not provided; `alias` is an optional human-friendly name (e.g., `claude-v1`).
  - Used for: Selecting which titles/descriptions to view or search when multiple summary passes exist.
  - Where: Columns on `cluster_summaries`; also stored in Chroma summaries metadata as `summary_run_id` and `alias` for filtering during semantic cluster search.

- `hierarchy_run_id` (Hierarchy Run):
  - What: Unique ID for one hierarchical merging run that organizes clusters into parent/child categories (e.g., `hier-<run_id>-<timestamp>`).
  - Used for: Viewing and reusing a specific multi-level hierarchy separate from the base clustering.
  - Where: Column on `cluster_hierarchies`; all nodes for a hierarchy share the same `hierarchy_run_id`.

Provenance and safety:
- A single `run_id` defines the vector space for semantic search; the CLI and SDK resolve provider/model/dimension from the run to avoid mixing spaces.
- Queries in Chroma store stable metadata (query_id, model, language, conversation_id). Cluster membership is always joined from SQLite (`query_clusters`) to ensure correctness per-run.
- Cluster summaries in Chroma store `run_id`, `cluster_id`, and summary provenance (`summary_run_id`, `alias`) so you can filter cluster search to a specific summary set.

### ChromaDB Collections

Note: Chroma collections are model/provider-specific. Names are suffixed by
`{provider}_{model}` to avoid mixing vector spaces.

**queries** - All user queries with embeddings (optional)
- ID format: `query_{id}`
- Metadata: model, language, conversation_id

**cluster_summaries** - Cluster titles + descriptions with embeddings
- ID format: `cluster_{run_id}_{cluster_id}`
- Metadata: run_id, cluster_id, title, description, num_queries
- Document: Combined title + description for semantic search

### Chroma Utilities and Verification

```bash
# List collections and vector space metadata
uv run lmsys chroma info --json | jq .

# Verify SQLite↔Chroma sync for a run and vector space alignment
uv run lmsys verify sync <RUN_ID> --json | jq .
```

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

### Embedding Smoke Tests

Quick online checks for embedding providers (requires API keys):

```bash
export OPENAI_API_KEY=... # optional
export COHERE_API_KEY=... # optional
uv run pytest -q -m smoke
```
These tests create a few embeddings and assert non-zero vectors and expected shapes.

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
    ├── hdbscan_clustering.py # HDBSCAN density-based clustering
    ├── hierarchy.py         # LLM-driven hierarchical merging (Clio-style)
    └── summarizer.py        # LLM summarization with instructor

tests/                       # Test suite with 20+ tests
utils/
    └── logging.py           # Rich-backed logging setup
```

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY` - For Claude models
- `OPENAI_API_KEY` - For GPT models
- `COHERE_API_KEY` - For Cohere embed-v4.0 models
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
