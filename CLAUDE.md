# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose & Goals

This repository provides **terminal-based CLI tools for agents to perform comprehensive data analysis on LMSYS queries**. The primary goal is to enable systematic investigation of how people use LLM systems through data-driven analysis workflows.

### Core Capabilities

- **Data Loading**: Download and process the LMSYS-1M dataset from Hugging Face
- **Clustering Analysis**: Group similar queries using embeddings and clustering algorithms (KMeans, HDBSCAN)
- **Hierarchical Organization**: Create multi-level cluster hierarchies using LLM-driven merging (Anthropic Clio methodology)
- **Cluster Investigation**: Explore and summarize query clusters to identify patterns and usage trends
- **Contrastive Analysis**: Highlight what makes each cluster unique compared to neighbors
- **Semantic Search**: Navigate queries and clusters using natural language
- **Hypothesis Generation**: Discover insights about user behavior, query patterns, and LLM interactions
- **Agentic Cluster Curation**: Interactive CLI tools (`lmsys edit`) for fixing cluster quality issues through direct CRUD operations

### Agent Workflow

Agents can use this tool to:

1. **Investigate LLM Usage Patterns**: Discover how users interact with different LLM systems
2. **Identify Common Use Cases**: Group similar queries to find patterns
3. **Generate Insights**: Use LLM-powered summarization to understand cluster characteristics
4. **Form Hypotheses**: Systematically explore data to develop testable hypotheses
5. **Export Findings**: Save analysis results for further investigation

All capabilities are accessible through the `lmsys` CLI command in a composable workflow: `load → cluster → summarize → merge-clusters → search → export`

### Web Viewer

A **Next.js-based interactive web interface** (`web/`) provides read-only visualization of clustering results with **zero external dependencies** (no ChromaDB server required):

- **Jobs Dashboard**: Browse all clustering runs with metadata
- **Hierarchy Explorer**: Navigate multi-level cluster hierarchies with enhanced visual controls
  - Expand/collapse all controls
  - Visual progress bars and color coding by cluster size
  - Summary statistics (total clusters, leaf count, levels, query count)
- **Search (SQL LIKE queries)**: Global and cluster-specific search without ChromaDB
- **Query Browser**: Paginated view of queries within each cluster (50 per page)
- **Cluster Details**: LLM-generated summaries, descriptions, and representative queries

**Architecture**: Next.js 15 + Drizzle ORM (SQLite) + Zod + ShadCN UI

**Data Flow**: Python CLI creates SQLite database → Next.js reads SQLite (read-only) → Browser UI

**Quick Start**:
```bash
cd web
npm install
npm run dev  # Opens http://localhost:3000
```

The viewer uses only SQLite (no ChromaDB server). All search uses SQL LIKE queries. See `web/README.md` for full documentation.

### Extensibility

If agents identify gaps in functionality or need additional tools to enhance their analysis capabilities, they should suggest these improvements to the user for potential implementation.

## User Preferences

**Default Analysis Workflow**: When the user requests to "run an analysis" or similar commands, ALWAYS execute the complete analysis pipeline including hierarchical merging:

1. Load data with embeddings (`lmsys load --limit N --use-chroma`)
2. Run clustering (`lmsys cluster kmeans --n-clusters N --use-chroma`)
3. Generate LLM summaries (`lmsys summarize <RUN_ID> --alias "analysis-v1"`)
4. **ALWAYS run hierarchical merge** (`lmsys merge-clusters <RUN_ID>`)

The hierarchical merge step is essential for organizing clusters into a navigable structure and should never be skipped.

## Build, Test, and Dev Commands

**Installation and setup:**
```bash
uv sync                          # Install dependencies and create virtual env
huggingface-cli login            # Required for LMSYS-1M dataset access
export ANTHROPIC_API_KEY="..."   # Or OPENAI_API_KEY, COHERE_API_KEY, GROQ_API_KEY
```

**CLI commands:**
```bash
uv run lmsys --help                                        # Show all commands
uv run lmsys load --limit 10000 --use-chroma              # Load data with embeddings
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma # Run clustering
uv run lmsys runs --latest                                # Show most recent run
uv run lmsys summarize <RUN_ID> --alias "v1"              # Generate LLM summaries
uv run lmsys merge-clusters <RUN_ID>                      # Build hierarchy
uv run lmsys list-clusters <RUN_ID>                       # View cluster titles
uv run lmsys search "python" --run-id <RUN_ID>            # Semantic search
```

**Cluster Curation Commands (`lmsys edit`):**
```bash
# Query operations
uv run lmsys edit view-query <QUERY_ID>                                    # View query with cluster assignments
uv run lmsys edit move-query <RUN_ID> --query-id <ID> --to-cluster <ID>   # Move query to different cluster
uv run lmsys edit move-queries <RUN_ID> --query-ids 1,2,3 --to-cluster <ID>  # Batch move queries

# Cluster operations
uv run lmsys edit rename-cluster <RUN_ID> --cluster-id <ID> --title "..."           # Rename cluster
uv run lmsys edit merge-clusters <RUN_ID> --source 1,2,3 --target <ID>              # Merge clusters
uv run lmsys edit split-cluster <RUN_ID> --cluster-id <ID> --query-ids 1,2,3 \
  --new-title "..." --new-description "..."                                          # Split cluster
uv run lmsys edit delete-cluster <RUN_ID> --cluster-id <ID> --orphan                # Delete cluster

# Metadata operations
uv run lmsys edit tag-cluster <RUN_ID> --cluster-id <ID> --coherence 3 \
  --quality medium --notes "..."                                                     # Tag cluster metadata
uv run lmsys edit flag-cluster <RUN_ID> --cluster-id <ID> --flag "language_mixing" # Flag for review

# Audit operations
uv run lmsys edit history <RUN_ID> --cluster-id <ID>     # View edit history
uv run lmsys edit audit <RUN_ID>                         # Full audit log
uv run lmsys edit orphaned <RUN_ID>                      # List orphaned queries
uv run lmsys edit select-bad-clusters <RUN_ID> --max-size 10  # Find problematic clusters
```

**Testing:**
```bash
uv run pytest -v                                    # All tests
uv run pytest tests/test_models.py -v               # Single test file
uv run pytest --cov=src/lmsys_query_analysis       # With coverage
uv run pytest -q -m smoke                          # Embedding smoke tests (requires API keys)
bash smoketest.sh                                  # End-to-end smoke test
SMOKE_LIMIT=500 bash smoketest.sh                  # Smoke test with custom limit
```

**Logging:**
```bash
uv run lmsys -v cluster --n-clusters 200  # Verbose logging (DEBUG level)
```

## Architecture

### High-Level Structure

The codebase follows a layered architecture:

1. **CLI Layer** (`cli/main.py`): Typer-based command interface with Rich terminal UI
2. **Business Logic** (`clustering/`, `db/loader.py`): Clustering algorithms, LLM summarization, hierarchical merging
3. **Data Layer** (`db/models.py`, `db/connection.py`, `db/chroma.py`): SQLite persistence and ChromaDB vector storage
4. **SDK Layer** (`semantic/`): Typed client interfaces for programmatic access (ClustersClient, QueriesClient)

### Database Schema (SQLite + SQLModel)

**queries** - First user query from each conversation
- `id`, `conversation_id` (unique), `model`, `query_text`, `language`
- `extra_metadata` (JSON), `created_at`

**clustering_runs** - Track clustering experiments
- `run_id` (primary key, format: `kmeans-200-20251004-170442`)
- `algorithm`, `num_clusters`, `parameters` (JSON), `description`, `created_at`
- `parameters` stores: `embedding_provider`, `embedding_model`, `embedding_dimension` (defines vector space)

**query_clusters** - Map queries to clusters per run
- `run_id`, `query_id`, `cluster_id`, `confidence_score`
- Composite unique constraint on `(run_id, query_id)`

**cluster_summaries** - LLM-generated summaries (supports multiple summary runs)
- `run_id`, `cluster_id`, `summary_run_id`, `alias` (friendly name)
- `title`, `description`, `summary`, `num_queries`
- `representative_queries` (JSON), `model`, `parameters` (JSON)
- Composite unique constraint on `(run_id, cluster_id, summary_run_id)`

**cluster_hierarchies** - Multi-level cluster hierarchies (Clio-style organization)
- `hierarchy_run_id` (format: `hier-<run_id>-<timestamp>`)
- `run_id`, `cluster_id`, `parent_cluster_id`, `level` (0=leaf, 1=first merge, etc.)
- `children_ids` (JSON array), `title`, `description`

**cluster_edits** - Audit trail for cluster curation operations
- `run_id`, `cluster_id`, `edit_type` ('rename', 'move_query', 'merge', 'split', 'delete', 'tag')
- `editor` ('claude', 'cli-user', or username), `timestamp`
- `old_value` (JSON), `new_value` (JSON), `reason` (text)

**cluster_metadata** - Quality annotations for clusters
- `run_id`, `cluster_id`, `coherence_score` (1-5), `quality` ('high', 'medium', 'low')
- `flags` (JSON array: 'language_mixing', 'needs_review', etc.), `notes`, `last_edited`

**orphaned_queries** - Queries removed from clusters
- `run_id`, `query_id`, `original_cluster_id`, `orphaned_at`, `reason`

### ChromaDB Collections

Collections are suffixed by `{provider}_{model}` to avoid mixing vector spaces (e.g., `queries_cohere_embed-v4.0`).

**queries** - All user queries with embeddings
- ID format: `query_{id}`
- Metadata: `model`, `language`, `conversation_id`

**cluster_summaries** - Cluster titles + descriptions with embeddings
- ID format: `cluster_{run_id}_{cluster_id}`
- Metadata: `run_id`, `cluster_id`, `summary_run_id`, `alias`, `title`, `description`, `num_queries`
- Document: Combined title + description for semantic search

### ID System and Provenance

- **`run_id`**: Identifies a clustering run and its vector space (`kmeans-200-20251004-170442`)
  - Stores embedding provider/model/dimension in `clustering_runs.parameters`
  - Used for filtering, searching, and ensuring vector space consistency

- **`summary_run_id`**: Identifies a summarization pass (auto-generated as `summary-<model>-<timestamp>`)
  - Multiple summary runs can exist per clustering run (compare models/prompts)
  - `alias` provides human-friendly names (e.g., `"claude-v1"`, `"gpt4-test"`)

- **`hierarchy_run_id`**: Identifies a hierarchical merging run (`hier-<run_id>-<timestamp>`)
  - Organizes clusters into parent-child relationships
  - All nodes in a hierarchy share the same `hierarchy_run_id`

**Vector space safety:**
- Each `run_id` defines the embedding space; CLI resolves provider/model/dimension from the run
- Queries in Chroma have stable metadata; cluster membership is joined from SQLite (`query_clusters`)
- Cluster summaries in Chroma include provenance (`run_id`, `summary_run_id`, `alias`) for filtering

### Hierarchical Merging (Clio Methodology)

Implemented in `clustering/hierarchy.py` following Anthropic's Clio approach:

1. **Neighborhood Formation**: Group similar clusters using embeddings (manageable LLM context)
2. **Category Generation**: LLM proposes broader category names for each neighborhood
3. **Deduplication**: Merge similar categories globally to create distinct parents
4. **Assignment**: LLM assigns each child cluster to best-fit parent using semantic matching
5. **Refinement**: LLM refines parent names based on actual assigned children
6. **Iteration**: Repeat process for multiple levels

Key Pydantic models: `NeighborhoodCategories`, `DeduplicatedClusters`, `ClusterAssignment`, `RefinedClusterSummary`

### Semantic SDK (`semantic/`)

Provides typed client interfaces for programmatic access:

- **`ClustersClient`**: Search cluster summaries with run-aware filtering
  - `from_run()`: Auto-configure from clustering run metadata
  - `find()`: Search clusters by text with optional alias/summary_run_id filtering

- **`QueriesClient`**: Search queries with cluster conditioning
  - `from_run()`: Auto-configure from clustering run metadata
  - Supports two-stage search (find clusters → search queries within clusters)

- **Shared types** (`types.py`): `RunSpace`, `ClusterHit`, `QueryHit`, `FacetBucket`, `SearchResult`

### Embedding Pipeline (`clustering/embeddings.py`)

**`EmbeddingGenerator`** class supports multiple providers:
- **sentence-transformers**: Local models (e.g., `all-MiniLM-L6-v2`)
- **openai**: OpenAI API (e.g., `text-embedding-3-small`)
- **cohere**: Cohere API with Matryoshka support (e.g., `embed-v4.0` with dimension 256)

Provider is stored in `clustering_runs.parameters` to ensure consistency across runs.

## Project Structure & Modules

```
src/lmsys_query_analysis/
├── cli/
│   ├── main.py              # Typer CLI: load, cluster, summarize, merge-clusters, search, edit
│   └── commands/
│       ├── edit.py          # Cluster curation commands (lmsys edit)
│       └── ...              # Other command modules
├── db/
│   ├── models.py            # SQLModel schemas (Query, ClusteringRun, ClusterEdit, etc.)
│   ├── connection.py        # Database manager (default: ~/.lmsys-query-analysis/queries.db)
│   ├── loader.py            # LMSYS dataset loader with HuggingFace integration
│   └── chroma.py            # ChromaDB manager (default: ~/.lmsys-query-analysis/chroma/)
├── services/
│   └── curation_service.py  # Cluster curation business logic (move, rename, merge, tag, etc.)
├── clustering/
│   ├── embeddings.py        # Multi-provider embedding wrapper
│   ├── kmeans.py            # MiniBatchKMeans streaming clustering
│   ├── hdbscan_clustering.py # HDBSCAN density-based clustering
│   ├── hierarchy.py         # LLM-driven hierarchical merging (Clio-style)
│   └── summarizer.py        # LLM summarization with instructor
├── semantic/
│   ├── types.py             # Shared types for SDK (RunSpace, ClusterHit, etc.)
│   ├── clusters.py          # ClustersClient for cluster search
│   └── queries.py           # QueriesClient for query search
└── utils/
    └── logging.py           # Rich-backed logging setup

tests/                       # Pytest suite (20+ tests)
smoketest.sh                 # End-to-end smoke test script
```

## Coding Style & Naming

- Python 3.10+, 4-space indentation, type hints throughout (`py.typed`)
- Modules/packages: `snake_case`; classes: `CapWords`; functions/vars: `snake_case`
- Prefer small, focused functions; docstrings on public functions/classes
- Rich terminal UX for CLI output; keep messages actionable and compact

## Testing Guidelines

- Framework: `pytest`. Place tests under `tests/` with files named `test_*.py` and functions `test_*`
- Use in-memory SQLite for model tests; avoid network in unit tests
- Smoke tests (marked with `@pytest.mark.smoke`) hit external APIs and require API keys
- Add tests for new CLI flags and database behaviors; aim to keep existing coverage passing

## Commit & Pull Requests

- Commits: imperative, concise, scoped (e.g., "Add kmeans run summary table"). Group related changes.
- PRs: include purpose, key changes, test plan (commands run and outputs), and screenshots/snippets for CLI tables where useful. Link issues when applicable and update `README.md` if user-facing behavior changes.

## Security & Config Tips

- Hugging Face: `huggingface-cli login` and accept LMSYS-1M terms before `load`
- API keys: set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `COHERE_API_KEY`, or `GROQ_API_KEY` in your shell, not in code
- Defaults: SQLite at `~/.lmsys-query-analysis/queries.db`; Chroma at `~/.lmsys-query-analysis/chroma`
- Override with `--db-path` and `--chroma-path` flags
