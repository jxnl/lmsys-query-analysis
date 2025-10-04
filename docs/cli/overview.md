# CLI Overview

The `lmsys` command-line interface provides comprehensive tools for loading, clustering, and analyzing the LMSYS-1M dataset.

## Command Structure

```bash
uv run lmsys [OPTIONS] COMMAND [ARGS]...
```

## Available Commands

### Data Management

- **`load`** - Download and load queries from LMSYS-1M dataset
- **`clear`** - Clear all data (database and ChromaDB)
- **`backfill-chroma`** - Backfill missing embeddings in ChromaDB

### Clustering

- **`cluster kmeans`** - Run MiniBatchKMeans clustering
- **`cluster hdbscan`** - Run HDBSCAN density-based clustering

### Analysis

- **`summarize`** - Generate LLM summaries for clusters
- **`runs`** - List all clustering runs
- **`list-clusters`** - List clusters from a specific run
- **`inspect`** - Inspect a specific cluster
- **`search`** - Semantic search for queries or clusters
- **`export`** - Export clusters to CSV/JSON

## Global Options

```bash
--db-path PATH           # Custom database path (default: ~/.lmsys-query-analysis/queries.db)
--chroma-path PATH       # Custom ChromaDB path (default: ~/.lmsys-query-analysis/chroma)
-v, --verbose           # Enable verbose logging
--help                  # Show help message
```

## Examples

```bash
# Load data with verbose logging
uv run lmsys -v load --limit 10000 --use-chroma

# Use custom database path
uv run lmsys --db-path /tmp/test.db load --limit 1000

# Cluster with custom paths
uv run lmsys --db-path /tmp/test.db --chroma-path /tmp/chroma cluster kmeans --n-clusters 50
```

## Next Steps

- [Data Loading](load.md)
- [Clustering](clustering.md)
- [Analysis Commands](analysis.md)
