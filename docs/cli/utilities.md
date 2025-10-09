# Utility Commands

Utilities for managing ChromaDB, verifying data consistency, and maintaining the system.

## Overview

Utility commands help with:
- **ChromaDB management**: Collection utilities and diagnostics
- **Data verification**: Consistency checks across database and vector store
- **Cleanup**: Clearing data and rebuilding indexes
- **Backfilling**: Adding embeddings to existing queries

---

## clear

Remove all data from SQLite database and ChromaDB.

```bash
uv run lmsys clear [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--db-path` | PATH | ~/.lmsys-query-analysis/queries.db | Database path |
| `--chroma-path` | PATH | ~/.lmsys-query-analysis/chroma | ChromaDB path |

### Examples

Clear default databases:

```bash
uv run lmsys clear
```

Clear custom databases:

```bash
uv run lmsys --db-path /tmp/test.db --chroma-path /tmp/chroma clear
```

### Warning

⚠️ **This command is destructive and cannot be undone.**

It deletes:
- All queries
- All clustering runs
- All cluster summaries
- All hierarchies
- All ChromaDB embeddings
- All curation metadata (edits, tags, orphaned queries)

Use this when:
- Starting fresh with new data
- Testing different clustering approaches
- Cleaning up after experiments

---

## backfill-chroma

Backfill missing embeddings into ChromaDB for queries that were loaded without embeddings.

```bash
uv run lmsys backfill-chroma [OPTIONS]
```

### Use Cases

1. **Loaded data without `--use-chroma`**: Add embeddings later
2. **Switched embedding models**: Re-embed with different model
3. **Missing embeddings**: Fix incomplete ChromaDB collections

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--embedding-model` | TEXT | all-MiniLM-L6-v2 | Embedding model (provider/model) |
| `--batch-size` | INTEGER | 32 | Batch size for embedding generation |
| `--chunk-size` | INTEGER | 5000 | Process queries in chunks |

### Examples

Basic backfill:

```bash
uv run lmsys backfill-chroma
```

Use specific model:

```bash
uv run lmsys backfill-chroma \
  --embedding-model cohere/embed-v4.0 \
  --batch-size 64
```

Large batch for speed:

```bash
uv run lmsys backfill-chroma \
  --batch-size 128 \
  --chunk-size 10000
```

### Output

```
Backfilling embeddings into ChromaDB...
Found 10000 queries in database
Checking ChromaDB for existing embeddings...
Missing: 8432 queries

Generating embeddings:
Chunk 1/2: 100%|████████████████| 5000/5000
Chunk 2/2: 100%|████████████████| 3432/3432

Added 8432 embeddings to ChromaDB
Backfill complete!
```

---

## chroma

ChromaDB collection utilities.

```bash
uv run lmsys chroma COMMAND [OPTIONS]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `list-collections` | List all ChromaDB collections |
| `collection-stats` | Show stats for a specific collection |
| `delete-collection` | Delete a ChromaDB collection |

### Examples

List all collections:

```bash
uv run lmsys chroma list-collections
```

View collection stats:

```bash
uv run lmsys chroma collection-stats queries_cohere_embed-v4.0
```

Delete a collection:

```bash
uv run lmsys chroma delete-collection queries_openai_text-embedding-3-small
```

### Collection Naming

Collections are named by pattern:

```
{type}_{provider}_{model}
```

Examples:
- `queries_cohere_embed-v4.0` - Queries with Cohere embeddings
- `cluster_summaries_openai_text-embedding-3-small` - Summaries with OpenAI embeddings

This prevents mixing embeddings from different models/providers.

---

## verify

Verification and consistency checks between SQLite and ChromaDB.

```bash
uv run lmsys verify COMMAND [OPTIONS]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `embeddings` | Check query-embedding consistency |
| `clusters` | Verify cluster assignments |
| `summaries` | Check summary embeddings |

### Examples

Verify all query embeddings exist:

```bash
uv run lmsys verify embeddings
```

Check cluster consistency:

```bash
uv run lmsys verify clusters --run-id kmeans-200-20251008
```

Verify summary embeddings:

```bash
uv run lmsys verify summaries --run-id kmeans-200-20251008
```

### Output

```
Verifying embeddings...

SQLite queries: 10000
ChromaDB embeddings: 9987
Missing: 13 queries

Missing query IDs:
  234, 567, 891, 1023, 1456, 2345, 2678, 3456, 4567, 5678, 6789, 7890, 8901

Run 'lmsys backfill-chroma' to fix missing embeddings.
```

### When to Use

Run verification when:
- Embeddings seem inconsistent
- Search results are unexpected
- After manual database modifications
- Before important clustering runs
- After system crashes or interruptions

---

## Performance Tuning

### Backfill Performance

**Speed up backfilling:**
- Increase `--batch-size` (requires more memory)
- Use faster embedding models (local models)
- Increase `--chunk-size` for larger batches

**Reduce memory:**
- Decrease `--batch-size`
- Decrease `--chunk-size`
- Process in multiple runs

### ChromaDB Optimization

**Collection Management:**
- Delete unused collections to save disk space
- Keep embedding models consistent within collections
- Use separate collections for different experiments

### Disk Space

Monitor disk usage:

```bash
# SQLite database size
du -h ~/.lmsys-query-analysis/queries.db

# ChromaDB size
du -sh ~/.lmsys-query-analysis/chroma
```

---

## Troubleshooting

### "ChromaDB collection not found"

**Cause**: Collection doesn't exist for specified embedding model.

**Solution**:
1. Check collections: `lmsys chroma list-collections`
2. Backfill if needed: `lmsys backfill-chroma --embedding-model <MODEL>`

### "Embedding dimension mismatch"

**Cause**: Trying to use embeddings from different models together.

**Solution**:
1. Stick to one embedding model per workflow
2. Or use separate ChromaDB collections
3. Clear and reload if needed

### "Out of memory during backfill"

**Cause**: Batch size too large.

**Solution**:
```bash
lmsys backfill-chroma --batch-size 16 --chunk-size 1000
```

---

## Database Maintenance

### Regular Maintenance Tasks

**Weekly**:
- Verify embeddings: `lmsys verify embeddings`
- Check disk space

**After Experiments**:
- Clear unused data: `lmsys clear` (if starting fresh)
- Delete old collections: `lmsys chroma delete-collection <NAME>`

**Before Production Runs**:
- Verify consistency: `lmsys verify embeddings`
- Backfill if needed: `lmsys backfill-chroma`

### Backup Strategies

**SQLite Database**:
```bash
cp ~/.lmsys-query-analysis/queries.db ~/backups/queries-$(date +%Y%m%d).db
```

**ChromaDB**:
```bash
tar -czf ~/backups/chroma-$(date +%Y%m%d).tar.gz ~/.lmsys-query-analysis/chroma
```

---

## Advanced Usage

### Multiple Embedding Models

Work with multiple embedding models simultaneously:

```bash
# Load with Cohere
uv run lmsys load --limit 10000 \
  --embedding-model cohere/embed-v4.0 --use-chroma

# Backfill with OpenAI (creates separate collection)
uv run lmsys backfill-chroma \
  --embedding-model openai/text-embedding-3-small

# Now have both collections:
# - queries_cohere_embed-v4.0
# - queries_openai_text-embedding-3-small
```

### Custom ChromaDB Paths

Use project-specific ChromaDB:

```bash
# Project A
uv run lmsys --chroma-path ~/projects/project-a/chroma \
  load --limit 5000 --use-chroma

# Project B
uv run lmsys --chroma-path ~/projects/project-b/chroma \
  load --limit 5000 --use-chroma
```

---

## See Also

- [Data Loading](load.md) - Initial data loading with embeddings
- [Clustering](clustering.md) - Using embeddings for clustering
- [Search](search.md) - Semantic search with embeddings
- [Overview](overview.md) - Complete command reference
