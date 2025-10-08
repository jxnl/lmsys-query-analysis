# Clustering API

Python API for clustering queries using KMeans or HDBSCAN algorithms.

## KMeans Clustering

### Usage Example

```python
from lmsys_query_analysis.clustering.kmeans import cluster_queries_kmeans
from lmsys_query_analysis.db.connection import DatabaseManager

db = DatabaseManager("~/.lmsys-query-analysis/queries.db")

run_id = cluster_queries_kmeans(
    db=db,
    n_clusters=100,
    embedding_model="embed-v4.0",
    embedding_provider="cohere",
    embed_batch_size=32,
    mb_batch_size=4096,
    chunk_size=5000,
    description="Fine-grained clustering",
    use_chroma=True,
    chroma_path="~/.lmsys-query-analysis/chroma"
)

print(f"Clustering complete: {run_id}")
```

### Parameters

- **db** (DatabaseManager): Database manager instance
- **n_clusters** (int): Number of clusters to create
- **embedding_model** (str): Model name for embeddings (e.g., "embed-v4.0", "text-embedding-3-small")
- **embedding_provider** (str): Provider ("cohere", "openai", or "sentence-transformers")
- **embed_batch_size** (int): Batch size for embedding generation
- **mb_batch_size** (int): MiniBatch size for KMeans
- **chunk_size** (int): Chunk size for streaming processing
- **description** (str | None): Optional run description
- **use_chroma** (bool): Whether to use ChromaDB
- **chroma_path** (str | None): Path to ChromaDB storage

### Returns

- **run_id** (str): Unique identifier for the clustering run

---

## HDBSCAN Clustering

### Usage Example

```python
from lmsys_query_analysis.clustering.hdbscan_clustering import cluster_queries_hdbscan
from lmsys_query_analysis.db.connection import DatabaseManager

db = DatabaseManager("~/.lmsys-query-analysis/queries.db")

run_id = cluster_queries_hdbscan(
    db=db,
    embedding_model="embed-v4.0",
    embedding_provider="cohere",
    embed_batch_size=32,
    chunk_size=5000,
    min_cluster_size=15,
    min_samples=5,
    description="Density-based clustering",
    use_chroma=True,
    chroma_path="~/.lmsys-query-analysis/chroma"
)

print(f"HDBSCAN clustering complete: {run_id}")
```

### Parameters

- **db** (DatabaseManager): Database manager instance
- **embedding_model** (str): Model name for embeddings (e.g., "embed-v4.0", "text-embedding-3-small")
- **embedding_provider** (str): Provider ("cohere", "openai", or "sentence-transformers")
- **embed_batch_size** (int): Batch size for embedding generation
- **chunk_size** (int): Chunk size for streaming processing
- **min_cluster_size** (int): Minimum cluster size for HDBSCAN
- **min_samples** (int): Minimum samples for core points
- **description** (str | None): Optional run description
- **use_chroma** (bool): Whether to use ChromaDB
- **chroma_path** (str | None): Path to ChromaDB storage

### Returns

- **run_id** (str): Unique identifier for the clustering run

---

## Embeddings

### Usage Example

```python
from lmsys_query_analysis.clustering.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(
    model_name="all-MiniLM-L6-v2",
    provider="sentence-transformers"
)

# Generate embeddings for queries
queries = ["How to center a div?", "Python list comprehension", "Git merge conflict"]
embeddings = generator.generate_batch(queries, batch_size=32)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")
```

### Supported Models

#### Sentence Transformers

- `all-MiniLM-L6-v2` (384 dims, fast, good quality)
- `all-mpnet-base-v2` (768 dims, slower, higher quality)
- `multi-qa-mpnet-base-dot-v1` (768 dims, optimized for Q&A)

### Methods

#### generate_batch

```python
def generate_batch(
    self,
    texts: list[str],
    batch_size: int = 32
) -> np.ndarray:
    """Generate embeddings for a batch of texts."""
```

---

## Summarization

### Usage Example

```python
from lmsys_query_analysis.clustering.summarizer import summarize_clusters
from lmsys_query_analysis.db.connection import DatabaseManager
from lmsys_query_analysis.db.chroma import ChromaDBManager

db = DatabaseManager("~/.lmsys-query-analysis/queries.db")
chroma = ChromaDBManager("~/.lmsys-query-analysis/chroma")

await summarize_clusters(
    db=db,
    run_id="kmeans-100-20251003-123456",
    cluster_id=None,  # Summarize all clusters
    max_queries=50,
    model="anthropic/claude-3-haiku-20240307",
    use_chroma=True,
    chroma_manager=chroma,
    concurrency=4,
    rpm=None,
    contrast_mode="none"
)
```

### Parameters

- **db** (DatabaseManager): Database manager instance
- **run_id** (str): Clustering run identifier
- **cluster_id** (int | None): Specific cluster to summarize (None for all)
- **max_queries** (int): Maximum queries to send to LLM
- **model** (str): LLM model identifier
- **use_chroma** (bool): Whether to use ChromaDB for context
- **chroma_manager** (ChromaDBManager | None): ChromaDB manager instance
- **concurrency** (int): Concurrent requests
- **rpm** (int | None): Rate limit (requests per minute)
- **contrast_mode** (str): Contrastive mode ("none", "neighbor", "all")

### Contrast Modes

- **none**: Standard summaries describing cluster content
- **neighbor**: Highlight differences from nearest clusters
- **all**: Contrast against all other clusters

---

## ChromaDB Integration

### Usage Example

```python
from lmsys_query_analysis.db.chroma import ChromaDBManager

chroma = ChromaDBManager("~/.lmsys-query-analysis/chroma")

# Add query embeddings
chroma.add_queries_batch(
    ids=["query_1", "query_2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    metadatas=[
        {"model": "gpt-3.5-turbo", "language": "en"},
        {"model": "claude-2", "language": "en"}
    ],
    documents=["How to center a div?", "Python list comprehension"]
)

# Search queries
results = chroma.search_queries(
    query_embedding=[0.15, 0.25, ...],
    n_results=10
)

# Add cluster summaries
chroma.add_cluster_summaries(
    run_id="kmeans-100-20251003-123456",
    summaries=[
        {
            "cluster_id": 0,
            "title": "CSS Layout",
            "description": "Questions about CSS positioning",
            "embedding": [0.1, 0.2, ...],
            "num_queries": 150
        }
    ]
)
```

### Methods

#### add_queries_batch

Add query embeddings to ChromaDB.

#### search_queries

Search for similar queries using vector similarity.

#### add_cluster_summaries

Add cluster summaries with embeddings.

#### search_cluster_summaries

Search cluster summaries by semantic similarity.

---

## Next Steps

- [Database Models](models.md)
- [CLI Reference](../cli/overview.md)
