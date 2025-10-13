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

### Web Viewer (Zero External Dependencies)

A **Next.js-based interactive web interface** (`web/`) provides read-only visualization with **no external services required** - just SQLite (no ChromaDB server needed):

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

### Agent Guidelines

For specialized agent workflows (cluster-inspector, data-analyst), see `AGENTS.md` which contains:
- Agent-specific prompts and capabilities
- Parallel execution patterns for cluster quality improvement
- Data analysis and insight generation workflows

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
3. **Service Layer** (`services/`): Business logic for cluster operations and data management
   - `curation_service.py`: CRUD operations for cluster editing (move, rename, merge, split, delete, tag)
   - `cluster_service.py`: Cluster querying and analysis
   - `query_service.py`: Query operations and retrieval
   - `export_service.py`: Data export functionality
   - `run_service.py`: Clustering run management
4. **Data Layer** (`db/models.py`, `db/connection.py`, `db/chroma.py`): SQLite persistence and ChromaDB vector storage
5. **SDK Layer** (`semantic/`): Typed client interfaces for programmatic access (ClustersClient, QueriesClient)
6. **Runner Module** (`runner.py`): High-level workflow orchestration and execution

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

### Cluster Curation Service (`services/curation_service.py`)

Comprehensive cluster quality management with full audit trail:

**Query Operations:**
- Move queries between clusters (single or batch)
- View query details with all cluster assignments

**Cluster Operations:**
- Rename clusters (update title/description)
- Merge multiple clusters into target
- Split queries from cluster into new cluster
- Delete clusters (orphan or reassign queries)

**Metadata & Quality:**
- Tag clusters with coherence scores (1-5 scale)
- Set quality levels (high/medium/low)
- Add flags (language_mixing, needs_review, etc.)
- Attach free-form notes

**Audit & Analysis:**
- Complete edit history per cluster or run
- Track orphaned queries with provenance
- Find problematic clusters by size, language mix, or quality

All operations create audit trail entries in `cluster_edits` table with editor, timestamp, old/new values, and reason.

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
│   ├── curation_service.py  # Cluster curation business logic (move, rename, merge, split, delete, tag)
│   ├── cluster_service.py   # Cluster querying and analysis
│   ├── query_service.py     # Query operations and retrieval
│   ├── export_service.py    # Data export functionality
│   └── run_service.py       # Clustering run management
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
├── utils/
│   └── logging.py           # Rich-backed logging setup
└── runner.py                # High-level workflow orchestration and execution

tests/                       # Pytest suite (20+ tests)
smoketest.sh                 # End-to-end smoke test script
web/                         # Next.js web viewer (zero external dependencies)
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

## CSV Data Loading

The CLI supports loading query data from **CSV files** in addition to HuggingFace datasets, enabling analysis of custom query datasets from any source.

### CSV File Format

**Required Columns** (exact, case-sensitive):
- `conversation_id`: Unique identifier for each conversation/query
- `query_text`: The actual query text to analyze

**Optional Columns**:
- `model`: Model name (defaults to "unknown" if not provided)
- `language`: Language of the query (e.g., "English", "Spanish")
- `timestamp`: ISO-8601 formatted timestamp (e.g., "2024-10-13T12:34:56Z")

**File Requirements**:
- UTF-8 encoded CSV file
- Column headers must match exactly (case-sensitive)
- Rows with empty `conversation_id` or `query_text` are skipped with warnings
- Invalid timestamps are set to `None` and continue processing

### CSV Example

```csv
conversation_id,query_text,model,language,timestamp
conv_1,What is machine learning?,gpt-4,English,2024-01-01T10:00:00Z
conv_2,How do I learn Python?,claude-3,English,2024-01-01T11:00:00Z
conv_3,Explain quantum computing,gpt-3.5,English,2024-01-01T12:00:00Z
```

### Usage Examples

**Load single CSV file:**
```bash
uv run lmsys load --csv data/queries.csv
```

**Load multiple CSV files** (automatically deduplicates across all files):
```bash
uv run lmsys load --csv data/dataset1.csv --csv data/dataset2.csv
```

**Load HuggingFace dataset** (explicit):
```bash
uv run lmsys load --hf-dataset lmsys/lmsys-chat-1m --limit 10000
```

**Default behavior** (loads LMSYS dataset if no source specified):
```bash
uv run lmsys load --limit 1000  # Defaults to lmsys/lmsys-chat-1m
```

**CSV with ChromaDB embeddings:**
```bash
uv run lmsys load --csv data/queries.csv --use-chroma
```

**CSV with custom embedding provider:**
```bash
uv run lmsys load --csv data/queries.csv --use-chroma \
  --embedding-provider cohere --embedding-model embed-v4.0
```

### Validation & Error Handling

**Missing required columns:**
```bash
$ uv run lmsys load --csv invalid.csv
Error: CSV file missing required column: query_text
```

**Empty required fields:**
- Rows with empty `conversation_id` or `query_text` are **skipped automatically**
- Warning logged for each skipped row
- Loading continues with valid rows

**Invalid timestamps:**
- Non-ISO-8601 timestamps set to `None`
- Warning logged, processing continues
- Example: `"2024/01/01"` → sets `timestamp=None`

**Nonexistent file:**
```bash
$ uv run lmsys load --csv missing.csv
Error: CSV file does not exist: missing.csv
```

**Mutual exclusivity:**
```bash
$ uv run lmsys load --csv data.csv --hf-dataset org/dataset
Error: Cannot specify both --csv and --hf-dataset. Choose one data source.
```

### Deduplication Behavior

- **Single source**: Deduplicates based on `conversation_id` within the file
- **Multiple sources**: Deduplicates across **all sources** in a single load command
- **Re-loading**: If you load the same CSV again, existing `conversation_id`s are **skipped** (logged in stats as "Skipped")

### Statistics Output

**Single CSV file:**
```
Loading Statistics
┌─────────────────┬───────┐
│ Metric          │ Count │
├─────────────────┼───────┤
│ Total Processed │ 100   │
│ Loaded          │ 95    │
│ Skipped         │ 5     │
│ Errors          │ 0     │
└─────────────────┴───────┘
```

**Multiple CSV files:**
```
Multi-Source Loading Statistics
┌─────────────────────┬───────────┬────────┬─────────┬────────┐
│ Source              │ Processed │ Loaded │ Skipped │ Errors │
├─────────────────────┼───────────┼────────┼─────────┼────────┤
│ csv:dataset1.csv    │ 1000      │ 1000   │ 0       │ 0      │
│ csv:dataset2.csv    │ 1000      │ 800    │ 200     │ 0      │
├─────────────────────┼───────────┼────────┼─────────┼────────┤
│ Total               │ 2000      │ 1800   │ 200     │ 0      │
└─────────────────────┴───────────┴────────┴─────────┴────────┘
```

### Phase 1 Limitations

These limitations are part of Phase 1 implementation and may be addressed in future versions:

1. **No dataset_id in schema**: All queries share the same database schema without explicit dataset tracking
2. **Unique conversation_id**: The `conversation_id` must be globally unique across all sources (collisions cause skipping)
3. **No column mapping**: CSV columns must use exact names (`conversation_id`, `query_text`); no custom mapping support
4. **Simple text format**: CSV source does not parse conversation arrays; `query_text` is used directly
5. **No streaming for CSV**: CSV files are loaded entirely into memory (HuggingFace supports streaming)

### Complete Workflow Example

```bash
# 1. Load custom CSV data with embeddings
uv run lmsys load --csv my_queries.csv --use-chroma

# 2. Cluster the data
uv run lmsys cluster kmeans --n-clusters 50 --use-chroma

# 3. Get the latest run ID
uv run lmsys runs --latest

# 4. Generate summaries
uv run lmsys summarize <RUN_ID> --alias "my-analysis"

# 5. Build hierarchy
uv run lmsys merge-clusters <RUN_ID>

# 6. Explore results
uv run lmsys list-clusters <RUN_ID>
uv run lmsys search "specific topic" --run-id <RUN_ID>
```
