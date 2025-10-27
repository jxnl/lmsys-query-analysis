# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vision: Autonomous AI Research Agent for Conversational System Analysis

**Long-term vision**: Every company building AI systems (chatbots, agents, RAG systems, voice assistants) uploads their interaction logs to our platform. An autonomous AI agent with specialized tools explores the data overnight and discovers engineering fixes, prompt improvements, and product opportunities the team didn't know existed.

**Current implementation**: This repository provides the foundational tool library and CLI interface that enables agents to perform autonomous data analysis on conversational datasets. Starting with LMSYS-1M query analysis, the tools demonstrate how agents can explore data, form hypotheses, and generate actionable insights without pre-defined workflows.

## Purpose & Goals

This repository provides **terminal-based CLI tools for agents to perform comprehensive data analysis on conversational systems**. The primary goal is to enable autonomous investigation of how people use AI systems through data-driven exploration workflows.

### Core Capabilities (Tool Categories)

The CLI implements specialized tools across multiple categories:

**1. Data Loading Tools**
- Load datasets from Hugging Face with configurable column mapping
- Support for multiple dataset formats (LMSYS, WildChat, custom schemas)
- Generate and backfill embeddings for semantic analysis
- Manage dataset lifecycle (status checks, clearing)
- Per-dataset database isolation (separate DBs recommended)

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

### Agent Workflow: Autonomous Investigation

Agents use these tools to autonomously explore data and discover insights:

**Phase 1: Landscape Exploration**
- Load dataset and generate embeddings
- Run clustering to identify behavioral patterns
- Build hierarchy to organize patterns into themes
- Generate summaries to understand cluster semantics

**Phase 2: Anomaly Detection**
- Identify outlier clusters (unusual size, latency, error rates)
- Investigate high-impact patterns affecting significant traffic
- Search for similar patterns across the dataset

**Phase 3: Hypothesis Testing**
- Create custom classifications to test hypotheses
- Drill down into specific clusters to find sub-patterns
- Compare successful vs. failing interactions
- Validate findings with statistical analysis

**Phase 4: Actionable Recommendations**
- Quantify business impact (affected traffic, revenue, cost)
- Generate engineering fixes with code snippets
- Estimate implementation effort and ROI
- Prioritize by impact and feasibility

**Key Innovation**: Agents make autonomous decisions about what to investigate, which tools to compose, when to drill down vs. zoom out, and when they have found enough insights. No pre-defined workflow required.

All capabilities are accessible through the `lmsys` CLI command in composable workflows: `load → cluster → summarize → merge-clusters → search → inspect → edit → export`

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

**Multi-Dataset Workflow**: The system supports loading multiple datasets using **separate databases per dataset** (recommended):

```bash
# Load WildChat into its own database
uv run lmsys load --limit 1000 --hf "allenai/WildChat-1M" \
  --conversation-id-column "conversation_hash" \
  --db-path ~/.lmsys-query-analysis/wildchat.db \
  --use-chroma --chroma-path ~/.lmsys-query-analysis/wildchat_chroma

# Run full pipeline on WildChat
uv run lmsys cluster kmeans --n-clusters 20 \
  --db-path ~/.lmsys-query-analysis/wildchat.db \
  --chroma-path ~/.lmsys-query-analysis/wildchat_chroma --use-chroma

uv run lmsys summarize <RUN_ID> --db-path ~/.lmsys-query-analysis/wildchat.db
uv run lmsys merge-clusters <RUN_ID> --db-path ~/.lmsys-query-analysis/wildchat.db

# Load LMSYS into separate database (default paths)
uv run lmsys load --limit 1000 --use-chroma
# ... run pipeline on LMSYS
```

**Note**: While technically possible to load multiple datasets into the same database, this is NOT recommended because:
- No `dataset_id` field exists to distinguish query sources
- Queries from different datasets will mix in clustering
- Cannot filter or analyze by dataset origin
- Use separate databases for dataset isolation

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
uv run lmsys load --limit 10000 --use-chroma              # Load LMSYS (default)
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma # Run clustering
uv run lmsys runs --latest                                # Show most recent run
uv run lmsys summarize <RUN_ID> --alias "v1"              # Generate LLM summaries
uv run lmsys merge-clusters <RUN_ID>                      # Build hierarchy
uv run lmsys list-clusters <RUN_ID>                       # View cluster titles
uv run lmsys search "python" --run-id <RUN_ID>            # Semantic search
```

**Loading custom datasets with column mapping:**
```bash
# Load WildChat-1M dataset
uv run lmsys load --limit 1000 --hf "allenai/WildChat-1M" \
  --conversation-id-column "conversation_hash" \
  --db-path ~/.lmsys-query-analysis/wildchat.db \
  --use-chroma --chroma-path ~/.lmsys-query-analysis/wildchat_chroma

# Load dataset with custom text column and model default
uv run lmsys load --hf "fka/awesome-chatgpt-prompts" \
  --text-column "prompt" --text-format \
  --model-default "chatgpt-3.5" --limit 5000 --use-chroma

# Available column mapping options:
# --text-column: column with query text (default: "conversation")
# --text-format: read text directly, not JSON conversation format
# --model-column: model field (default: "model")
# --language-column: language field (default: "language")
# --timestamp-column: timestamp field (default: "timestamp")
# --conversation-id-column: ID field (default: "conversation_id")
# --model-default: default model value (default: "unknown")
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

**Running Development Servers:**
```bash
# Run both servers (in separate terminals or background processes)
cd web && npm run dev                           # Frontend (Next.js) - http://localhost:3000
uv run python -m lmsys_query_analysis.api.app   # Backend (FastAPI) - http://localhost:8000

# API documentation available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - OpenAPI spec: http://localhost:8000/openapi.json

# Health check endpoint:
# http://localhost:8000/api/health
```

## Architecture

### High-Level Structure

The codebase follows a layered architecture:

1. **Web Layer** (`web/`): Next.js frontend for visualizing clustering results
2. **API Layer** (`api/`): FastAPI REST API with CORS-enabled endpoints for web interface
3. **CLI Layer** (`cli/main.py`): Typer-based command interface with Rich terminal UI
4. **Business Logic** (`clustering/`, `db/loader.py`): Clustering algorithms, LLM summarization, hierarchical merging
5. **Data Layer** (`db/models.py`, `db/connection.py`, `db/chroma.py`): SQLite persistence and ChromaDB vector storage
6. **SDK Layer** (`semantic/`): Typed client interfaces for programmatic access (ClustersClient, QueriesClient)

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
├── api/                     # FastAPI REST API
│   ├── app.py              # FastAPI application with CORS and error handling
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/            # API endpoints
│       ├── clustering.py   # Clustering operations
│       ├── analysis.py     # Analysis endpoints
│       ├── hierarchy.py    # Hierarchy navigation
│       ├── summaries.py    # Summary operations
│       ├── search.py       # Search endpoints
│       └── curation.py     # Curation operations
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

web/                         # Next.js web viewer (frontend)
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
