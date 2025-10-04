# Clustering

The clustering commands group similar queries together using machine learning algorithms.

## Available Algorithms

### KMeans

```bash
uv run lmsys cluster kmeans [OPTIONS]
```

MiniBatchKMeans is recommended for large datasets. It scales well and produces consistent results.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--n-clusters` | INTEGER | 100 | Number of clusters to create |
| `--use-chroma` | FLAG | False | Reuse embeddings from ChromaDB |
| `--embedding-model` | TEXT | all-MiniLM-L6-v2 | Embedding model |
| `--embedding-provider` | TEXT | sentence-transformers | Provider for embeddings |
| `--embed-batch-size` | INTEGER | 32 | Batch size for embedding |
| `--mb-batch-size` | INTEGER | 4096 | MiniBatch size for KMeans |
| `--chunk-size` | INTEGER | 5000 | Chunk size for streaming |
| `--description` | TEXT | None | Optional run description |

#### Examples

Basic clustering:

```bash
uv run lmsys cluster kmeans --n-clusters 100 --use-chroma
```

Fine-grained analysis with 200 clusters:

```bash
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma \
  --description "Fine-grained clustering"
```

Optimize for speed:

```bash
uv run lmsys cluster kmeans --n-clusters 50 --use-chroma \
  --embed-batch-size 64 --mb-batch-size 8192 --chunk-size 10000
```

### HDBSCAN

```bash
uv run lmsys cluster hdbscan [OPTIONS]
```

HDBSCAN finds natural clusters based on density. It excludes noise points and doesn't require specifying the number of clusters.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--use-chroma` | FLAG | False | Reuse embeddings from ChromaDB |
| `--embedding-model` | TEXT | all-MiniLM-L6-v2 | Embedding model |
| `--embedding-provider` | TEXT | sentence-transformers | Provider for embeddings |
| `--embed-batch-size` | INTEGER | 32 | Batch size for embedding |
| `--chunk-size` | INTEGER | 5000 | Chunk size for streaming |
| `--min-cluster-size` | INTEGER | 15 | Minimum cluster size |
| `--min-samples` | INTEGER | 5 | Minimum samples for core point |
| `--description` | TEXT | None | Optional run description |

#### Examples

Basic HDBSCAN:

```bash
uv run lmsys cluster hdbscan --use-chroma
```

Adjust density parameters:

```bash
uv run lmsys cluster hdbscan --use-chroma \
  --min-cluster-size 20 --min-samples 10
```

## Understanding Run IDs

Each clustering run generates a unique ID in the format:

```
{algorithm}-{num_clusters}-{timestamp}
```

Examples:
- `kmeans-100-20251003-123456`
- `hdbscan-20251003-123456`

Use this ID for downstream commands like `summarize`, `list-clusters`, and `search`.

## Choosing Cluster Count

For KMeans:

- **50-100 clusters**: Broad categories, good for overview
- **100-200 clusters**: Balanced granularity (recommended)
- **200-500 clusters**: Fine-grained analysis, specific patterns

For HDBSCAN:

- The algorithm determines cluster count automatically
- Adjust `--min-cluster-size` to control granularity
- Smaller values create more, smaller clusters

## Performance Optimization

### Memory

- Increase `--mb-batch-size` for faster KMeans (requires more RAM)
- Decrease `--chunk-size` if running out of memory

### Speed

- Use `--use-chroma` to reuse existing embeddings
- Increase `--embed-batch-size` with sufficient memory
- Use smaller embedding models like `all-MiniLM-L6-v2`

### Quality

- Use larger embedding models like `all-mpnet-base-v2`
- Increase `--n-clusters` for finer distinctions
- For HDBSCAN, adjust `--min-cluster-size` based on dataset size

## Output

The command displays:

- Number of queries processed
- Embedding progress
- Clustering progress
- Run ID for further analysis

Example output:

```
Clustering 10000 queries with kmeans (n_clusters=100)
Generating embeddings: 100%|████████████████| 10000/10000
Running MiniBatchKMeans...
Saving results...
Clustering complete!
Run ID: kmeans-100-20251003-123456

Use this ID with: lmsys summarize kmeans-100-20251003-123456
```

## Next Steps

- [Generate Summaries](analysis.md#summarize)
- [View Clusters](analysis.md#list-clusters)
- [Search Clusters](analysis.md#search)
