# Data Loading

The `load` command downloads and processes queries from the LMSYS-1M dataset.

## Basic Usage

```bash
uv run lmsys load [OPTIONS]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--limit` | INTEGER | None | Maximum number of queries to load |
| `--use-chroma` | FLAG | False | Store embeddings in ChromaDB for semantic search |
| `--embedding-model` | TEXT | all-MiniLM-L6-v2 | Sentence-transformers model for embeddings |
| `--embedding-provider` | TEXT | sentence-transformers | Embedding provider to use |
| `--batch-size` | INTEGER | 32 | Batch size for embedding generation |

## Examples

### Load All Queries

Load the entire LMSYS-1M dataset (1 million conversations):

```bash
uv run lmsys load
```

!!! warning
    Loading all 1M queries will take significant time and storage. Consider using `--limit` for testing.

### Load Sample for Testing

Load 10,000 queries for quick testing:

```bash
uv run lmsys load --limit 10000
```

### Enable Semantic Search

Load queries with ChromaDB embeddings for semantic search:

```bash
uv run lmsys load --limit 50000 --use-chroma
```

### Custom Embedding Model

Use a different embedding model:

```bash
uv run lmsys load --limit 10000 --use-chroma \
  --embedding-model all-mpnet-base-v2 \
  --batch-size 64
```

## Output

The command displays:

- Progress bar for dataset download
- Number of queries loaded
- Database location
- ChromaDB collection status (if `--use-chroma` is used)

Example output:

```
Loading LMSYS-1M dataset...
Loaded 10000 queries
Database: /Users/username/.lmsys-query-analysis/queries.db
ChromaDB: 10000 embeddings stored in 'queries' collection
```

## Storage Locations

- **SQLite Database**: `~/.lmsys-query-analysis/queries.db`
- **ChromaDB**: `~/.lmsys-query-analysis/chroma/`

Override with `--db-path` and `--chroma-path` flags.

## Requirements

Before running `load`, you must:

1. Authenticate with HuggingFace: `huggingface-cli login`
2. Accept LMSYS-1M dataset terms: [https://huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)

## Performance Tips

- Use `--limit` during development to iterate faster
- Increase `--batch-size` if you have sufficient memory
- `--use-chroma` adds overhead but enables semantic search
- Consider using `all-MiniLM-L6-v2` (default) for speed or `all-mpnet-base-v2` for quality

## Next Steps

- [Run Clustering](clustering.md)
- [Semantic Search](analysis.md#search)
