# CLI Overview

The `lmsys` command-line interface provides comprehensive tools for loading, clustering, analyzing, and curating the LMSYS-1M dataset.

## Command Structure

```bash
uv run lmsys [OPTIONS] COMMAND [ARGS]...
```

## Command Groups

### 📦 Data Management

- **`load`** - Download and load queries from LMSYS-1M dataset
- **`clear`** - Clear all data (database and ChromaDB)
- **`backfill-chroma`** - Backfill missing embeddings in ChromaDB

[→ Data Loading Guide](load.md)

### 🔍 Clustering

- **`cluster kmeans`** - Run MiniBatchKMeans clustering
- **`cluster hdbscan`** - Run HDBSCAN density-based clustering

[→ Clustering Guide](clustering.md)

### 📊 Analysis & Exploration

- **`summarize`** - Generate LLM-powered summaries for clusters
- **`runs`** - List all clustering runs
- **`list-clusters`** - List clusters from a specific run
- **`inspect`** - View detailed cluster information
- **`export`** - Export clusters to CSV/JSON

[→ Analysis Commands](analysis.md)

### 🌳 Hierarchical Organization

- **`merge-clusters`** - Create multi-level cluster hierarchies using LLM-driven merging
- **`show-hierarchy`** - Display hierarchical cluster structure as a tree

[→ Hierarchy Guide](hierarchy.md)

### 🔎 Search

- **`search`** - Semantic search across queries or cluster summaries
- **`search-cluster`** - Search cluster titles and descriptions

[→ Search Guide](search.md)

### ✏️ Cluster Curation

- **`edit view-query`** - View query details with cluster assignments
- **`edit move-query`** - Move queries between clusters
- **`edit rename-cluster`** - Rename cluster titles/descriptions
- **`edit merge-clusters`** - Merge multiple clusters
- **`edit split-cluster`** - Split queries into new cluster
- **`edit delete-cluster`** - Delete clusters (orphan or reassign queries)
- **`edit tag-cluster`** - Add quality metadata (coherence, quality, notes)
- **`edit flag-cluster`** - Flag clusters for review
- **`edit history`** - View edit history
- **`edit audit`** - View full audit log
- **`edit orphaned`** - List orphaned queries
- **`edit select-bad-clusters`** - Find problematic clusters

[→ Cluster Curation Guide](edit.md)

### 🛠️ Utilities

- **`chroma`** - ChromaDB utilities
- **`verify`** - Verification and consistency checks

[→ Utilities Guide](utilities.md)

## Global Options

```bash
--db-path PATH           # Custom database path (default: ~/.lmsys-query-analysis/queries.db)
--chroma-path PATH       # Custom ChromaDB path (default: ~/.lmsys-query-analysis/chroma)
-v, --verbose           # Enable verbose logging
--help                  # Show help message
```

## Quick Start Examples

### Basic Workflow

```bash
# 1. Load data
uv run lmsys load --limit 10000 --use-chroma

# 2. Cluster
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma

# 3. Generate summaries
uv run lmsys summarize <RUN_ID> --use-chroma

# 4. Create hierarchy
uv run lmsys merge-clusters <RUN_ID>

# 5. Explore
uv run lmsys show-hierarchy <HIERARCHY_RUN_ID>
uv run lmsys list-clusters <RUN_ID>
```

### Custom Database Paths

```bash
# Use custom database path
uv run lmsys --db-path /tmp/test.db load --limit 1000

# Cluster with custom paths
uv run lmsys --db-path /tmp/test.db --chroma-path /tmp/chroma \
  cluster kmeans --n-clusters 50 --use-chroma
```

### Verbose Logging

```bash
# Enable debug logging for troubleshooting
uv run lmsys -v load --limit 1000 --use-chroma
uv run lmsys -v cluster kmeans --n-clusters 100
```

## Typical Workflows

### Analysis Workflow

1. **Load** → Load dataset with embeddings
2. **Cluster** → Group similar queries
3. **Summarize** → Generate LLM descriptions
4. **Hierarchize** → Create multi-level organization
5. **Curate** → Fix quality issues with `edit` commands
6. **Export** → Save results for further analysis

### Curation Workflow

1. **Inspect** → Review cluster quality
2. **Tag** → Mark low-quality clusters
3. **Fix** → Use `edit` commands to improve
4. **Audit** → Review changes
5. **Export** → Save curated results

## Next Steps

- **Getting Started**: [Data Loading](load.md) → [Clustering](clustering.md) → [Analysis](analysis.md)
- **Advanced**: [Hierarchies](hierarchy.md) → [Curation](edit.md)
- **Reference**: [Search](search.md) → [Utilities](utilities.md)
