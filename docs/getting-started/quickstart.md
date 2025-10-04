# Quick Start

This guide will walk you through a complete workflow from loading data to analyzing clusters.

## Step 1: Load Data

Load a sample of 10,000 queries from the LMSYS-1M dataset:

```bash
uv run lmsys load --limit 10000 --use-chroma
```

The `--use-chroma` flag enables semantic search by storing embeddings in ChromaDB.

## Step 2: Run Clustering

Cluster the queries into 100 groups using KMeans:

```bash
uv run lmsys cluster kmeans --n-clusters 100 --use-chroma
```

This will output a run ID like `kmeans-100-20251003-123456`.

## Step 3: Generate Summaries

Generate LLM-powered summaries for each cluster:

```bash
uv run lmsys summarize kmeans-100-20251003-123456 --use-chroma
```

## Step 4: View Results

List clusters with their titles:

```bash
uv run lmsys list-clusters kmeans-100-20251003-123456
```

Show example queries:

```bash
uv run lmsys list-clusters kmeans-100-20251003-123456 --show-examples 3
```

## Step 5: Semantic Search

Find clusters related to specific topics:

```bash
uv run lmsys search "python programming" --search-type clusters --run-id kmeans-100-20251003-123456
```

Search individual queries:

```bash
uv run lmsys search "how to build neural network" --search-type queries --n-results 20
```

## Full Workflow Example

```bash
# Load 50k queries
uv run lmsys load --limit 50000 --use-chroma

# Run fine-grained clustering
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma --description "Fine-grained analysis"

# Generate summaries (note the run ID from previous command)
uv run lmsys summarize kmeans-200-20251003-123456 --use-chroma --concurrency 8

# Explore results
uv run lmsys list-clusters kmeans-200-20251003-123456 --limit 50
uv run lmsys search "machine learning tutorials" --search-type clusters --run-id kmeans-200-20251003-123456

# View all runs
uv run lmsys runs
```

## Next Steps

- [CLI Command Reference](../cli/overview.md)
- [Clustering Algorithms](../cli/clustering.md)
- [API Documentation](../api/models.md)
