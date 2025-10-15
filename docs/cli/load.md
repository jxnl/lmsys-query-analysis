# Data Loading

The `load` command downloads and processes queries from Hugging Face datasets. By default, it loads from the LMSYS-1M dataset (`lmsys/lmsys-chat-1m`), but you can specify any compatible Hugging Face dataset using the `--hf` flag.

## Basic Usage

```bash
uv run lmsys load [OPTIONS]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--hf` | TEXT | lmsys/lmsys-chat-1m | Hugging Face dataset to load from |
| `--limit` | INTEGER | None | Maximum number of queries to load |
| `--use-chroma` | FLAG | False | Store embeddings in ChromaDB for semantic search |
| `--embedding-model` | TEXT | cohere/embed-v4.0 | Embedding model (format: provider/model) |
| `--batch-size` | INTEGER | 32 | Batch size for embedding generation |

## Examples

### Load from Default Dataset

Load from the default LMSYS-1M dataset (no `--hf` flag needed):

```bash
# Load entire dataset (1 million conversations)
uv run lmsys load

# Load sample for testing
uv run lmsys load --limit 10000

# With semantic search enabled
uv run lmsys load --limit 50000 --use-chroma
```

!!! warning
    Loading all 1M queries will take significant time and storage. Consider using `--limit` for testing.

!!! info "Backwards Compatibility"
    When `--hf` is not specified, the system automatically uses `lmsys/lmsys-chat-1m`. All existing commands work without changes.

### Load from Custom Hugging Face Dataset

Specify a custom dataset using the `--hf` flag:

```bash
# Load from a different HF dataset
uv run lmsys load --hf username/my-conversations --limit 10000

# With semantic search
uv run lmsys load --hf username/customer-support-logs --limit 50000 --use-chroma

# Explicitly specify default dataset (same as omitting --hf)
uv run lmsys load --hf lmsys/lmsys-chat-1m --limit 10000 --use-chroma
```

### Load with Different Embedding Models

Combine custom datasets with different embedding models:

```bash
# Custom dataset + OpenAI embeddings
uv run lmsys load --hf username/my-dataset --limit 10000 --use-chroma \
  --embedding-model openai/text-embedding-3-small \
  --batch-size 64

# Custom dataset + Cohere embeddings
uv run lmsys load --hf company/internal-chats --limit 50000 --use-chroma \
  --embedding-model cohere/embed-v4.0 \
  --batch-size 50
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

1. **Authenticate with HuggingFace**: `huggingface-cli login`
2. **Accept dataset terms**: For the default LMSYS-1M dataset, accept terms at [https://huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
3. **For custom datasets**: Ensure you have access to the specified Hugging Face dataset and have accepted any required terms

## Dataset Requirements

When using custom Hugging Face datasets with `--hf`, your dataset should have a structure compatible with the LMSYS format. The loader expects conversations with user queries that can be extracted and analyzed. If your dataset has a different structure, you may need to adapt it or contact the maintainers for support.

## Performance Tips

- Use `--limit` during development to iterate faster
- Increase `--batch-size` if you have sufficient memory
- `--use-chroma` adds overhead but enables semantic search
- Default `cohere/embed-v4.0` provides good quality and speed with Matryoshka dimensions
- Use `sentence-transformers/all-MiniLM-L6-v2` for faster local processing
- Use `openai/text-embedding-3-small` or `openai/text-embedding-3-large` for highest quality

## Next Steps

- [Run Clustering](clustering.md)
- [Semantic Search](analysis.md#search)
