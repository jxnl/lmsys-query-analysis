# Analysis Commands

Commands for analyzing, summarizing, and exploring clustering results.

## Summarize

Generate LLM-powered summaries for clusters with metadata persistence.

```bash
uv run lmsys summarize RUN_ID [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cluster-id` | INTEGER | None | Summarize specific cluster only |
| `--max-queries` | INTEGER | 50 | Max queries to send to LLM |
| `--model` | TEXT | anthropic/claude-3-haiku-20240307 | LLM model to use |
| `--use-chroma` | FLAG | False | Use ChromaDB for context |
| `--concurrency` | INTEGER | 4 | Concurrent requests |
| `--rpm` | INTEGER | None | Rate limit (requests per minute) |
| `--contrast-mode` | TEXT | none | Contrastive summary mode (none/neighbor/all) |
| `--summary-run-id` | TEXT | None | Custom summary run ID (auto-generated if not provided) |
| `--alias` | TEXT | None | Friendly alias for this summary run |
| `--contrast-neighbors` | INTEGER | 2 | Number of neighbor clusters for contrast |
| `--contrast-examples` | INTEGER | 2 | Examples per neighbor cluster |

### Examples

Basic summarization:

```bash
uv run lmsys summarize kmeans-100-20251003-123456 --use-chroma
```

Use GPT-4 with alias:

```bash
uv run lmsys summarize kmeans-100-20251003-123456 \
  --model "openai/gpt-4o-mini" --alias "gpt4o-test" --use-chroma
```

Single cluster:

```bash
uv run lmsys summarize kmeans-100-20251003-123456 --cluster-id 5 --use-chroma
```

High concurrency with rate limiting:

```bash
uv run lmsys summarize kmeans-100-20251003-123456 \
  --concurrency 8 --rpm 60 --use-chroma
```

Contrastive summaries (highlight unique aspects):

```bash
uv run lmsys summarize kmeans-100-20251003-123456 \
  --contrast-mode neighbors --contrast-neighbors 3 --contrast-examples 2 --use-chroma
```

Custom summary run ID:

```bash
uv run lmsys summarize kmeans-100-20251003-123456 \
  --summary-run-id "claude-v1" --alias "my-best-summaries" --use-chroma
```

### Metadata Persistence

Each summarization run creates a `SummaryRun` record that tracks:
- **LLM Configuration**: Provider and model used
- **Parameters**: `max_queries`, `concurrency`, `rpm`, contrast settings
- **Execution Metadata**: `total_clusters`, `execution_time_seconds`, `alias`
- **Provenance**: Links to the clustering run and individual cluster summaries

This enables:
- **Reproducibility**: Recreate exact runs with same parameters
- **Comparison**: Compare effectiveness of different LLM models
- **Audit Trail**: Track which configurations produce best results
- **Cost Analysis**: Analyze costs across different providers

### Supported LLM Providers

- **Anthropic**: `anthropic/claude-3-haiku-20240307`, `anthropic/claude-3-sonnet-20240229`
- **OpenAI**: `openai/gpt-4`, `openai/gpt-3.5-turbo`
- **Groq**: `groq/llama-3.1-8b-instant`
- **Ollama**: `ollama/llama3` (local)

---

## List Runs

View all clustering runs.

```bash
uv run lmsys runs [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--latest` | FLAG | False | Show only the most recent run |

### Examples

All runs:

```bash
uv run lmsys runs
```

Latest run only:

```bash
uv run lmsys runs --latest
```

---

## List Clusters

Display clusters from a specific run.

```bash
uv run lmsys list-clusters RUN_ID [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--limit` | INTEGER | None | Limit number of clusters shown |
| `--show-examples` | INTEGER | 0 | Number of example queries per cluster |

### Examples

Basic list:

```bash
uv run lmsys list-clusters kmeans-100-20251003-123456
```

Top 20 clusters with examples:

```bash
uv run lmsys list-clusters kmeans-100-20251003-123456 \
  --limit 20 --show-examples 3
```

---

## Inspect Cluster

View detailed information about a specific cluster.

```bash
uv run lmsys inspect RUN_ID CLUSTER_ID [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--limit` | INTEGER | 20 | Number of queries to display |

### Examples

```bash
uv run lmsys inspect kmeans-100-20251003-123456 5
```

```bash
uv run lmsys inspect kmeans-100-20251003-123456 5 --limit 50
```

---

## Search

Semantic search for queries or cluster summaries.

```bash
uv run lmsys search QUERY [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--search-type` | TEXT | queries | Search "queries" or "clusters" |
| `--run-id` | TEXT | None | Filter by specific run (for clusters) |
| `--n-results` | INTEGER | 10 | Number of results to return |
| `--embedding-model` | TEXT | all-MiniLM-L6-v2 | Embedding model to use |

### Examples

Search cluster summaries:

```bash
uv run lmsys search "python programming" \
  --search-type clusters --run-id kmeans-100-20251003-123456
```

Search individual queries:

```bash
uv run lmsys search "how to build neural network" \
  --search-type queries --n-results 20
```

Cross-run cluster search:

```bash
uv run lmsys search "machine learning tutorials" --search-type clusters
```

!!! tip
    Use the same embedding model that was used during `load` or `cluster` for best results.

---

## Export

Export cluster data to CSV or JSON.

```bash
uv run lmsys export RUN_ID [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | PATH | None | Output file path (required) |
| `--format` | TEXT | csv | Output format: "csv" or "json" |
| `--include-queries` | FLAG | False | Include full query texts |

### Examples

Export to CSV:

```bash
uv run lmsys export kmeans-100-20251003-123456 --output results.csv
```

Export to JSON with queries:

```bash
uv run lmsys export kmeans-100-20251003-123456 \
  --output results.json --format json --include-queries
```

---

## Clear Data

Remove all data from database and ChromaDB.

```bash
uv run lmsys clear
```

!!! warning
    This command deletes all queries, clustering runs, and embeddings. It cannot be undone.

---

## Backfill ChromaDB

Backfill missing embeddings in ChromaDB.

```bash
uv run lmsys backfill-chroma [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--embedding-model` | TEXT | all-MiniLM-L6-v2 | Embedding model |
| `--batch-size` | INTEGER | 32 | Batch size |

### Examples

```bash
uv run lmsys backfill-chroma --batch-size 64
```

Use this command if you loaded data without `--use-chroma` and want to add embeddings later.
