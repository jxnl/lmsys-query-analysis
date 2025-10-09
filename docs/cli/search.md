# Search Commands

Semantic search capabilities for finding queries and clusters using natural language.

## Overview

The search functionality uses embeddings and vector similarity to find relevant content:

- **Query search**: Find individual queries matching your search term
- **Cluster search**: Find clusters whose summaries match your search term
- **Cross-run search**: Search across multiple clustering runs
- **Semantic matching**: Finds conceptually similar content, not just keyword matches

---

## search

General-purpose semantic search across queries or cluster summaries.

```bash
uv run lmsys search QUERY [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--search-type` | TEXT | queries | Search "queries" or "clusters" |
| `--run-id` | TEXT | None | Filter by specific run (for clusters) |
| `--n-results` | INTEGER | 10 | Number of results to return |
| `--embedding-model` | TEXT | Inherits from run | Embedding model to use |

### Examples

#### Search Individual Queries

Find queries about neural networks:

```bash
uv run lmsys search "how to build neural network" \
  --search-type queries --n-results 20
```

Find Python programming questions:

```bash
uv run lmsys search "python programming tutorials" \
  --search-type queries --n-results 15
```

#### Search Cluster Summaries

Find clusters about machine learning in a specific run:

```bash
uv run lmsys search "machine learning algorithms" \
  --search-type clusters \
  --run-id kmeans-200-20251008 \
  --n-results 10
```

Cross-run cluster search (all runs):

```bash
uv run lmsys search "web development frameworks" \
  --search-type clusters \
  --n-results 15
```

### Output Format

**Query search results:**
```
Top 10 results for "neural networks":

1. [Score: 0.89] How do I implement a simple neural network in PyTorch?
   Model: gpt-4 | Language: English

2. [Score: 0.86] What's the difference between CNN and RNN architectures?
   Model: claude-3-sonnet | Language: English

3. [Score: 0.84] Explain backpropagation in neural networks
   Model: gpt-3.5-turbo | Language: English
```

**Cluster search results:**
```
Top 5 clusters for "machine learning":

1. [Score: 0.92] Cluster 45: Machine Learning Algorithms and Techniques
   Run: kmeans-200-20251008 | 23 queries
   Description: Questions about ML algorithms, training, and optimization

2. [Score: 0.88] Cluster 67: Deep Learning and Neural Networks
   Run: kmeans-200-20251008 | 31 queries
   Description: PyTorch, TensorFlow, model architecture questions

3. [Score: 0.85] Cluster 12: Data Science Tools and Libraries
   Run: kmeans-200-20251008 | 19 queries
   Description: scikit-learn, pandas, data preprocessing
```

---

## search-cluster

Specialized search for cluster titles and descriptions.

```bash
uv run lmsys search-cluster QUERY [OPTIONS]
```

This is a convenience wrapper around `search --search-type clusters`.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | TEXT | None | Filter by specific run |
| `--alias` | TEXT | None | Filter by summary alias |
| `--n-results` | INTEGER | 10 | Number of results |

### Examples

Search within specific run:

```bash
uv run lmsys search-cluster "python programming" \
  --run-id kmeans-200-20251008 \
  --n-results 5
```

Filter by summary alias:

```bash
uv run lmsys search-cluster "database queries" \
  --alias "claude-v1" \
  --n-results 10
```

---

## How Semantic Search Works

### Embedding-Based Similarity

1. **Query embedding**: Your search term is converted to a vector
2. **Similarity calculation**: Cosine similarity with stored embeddings
3. **Ranking**: Results sorted by similarity score (0-1)
4. **Filtering**: Optional run/alias filters applied

### Advantages Over Keyword Search

- **Conceptual matching**: "ML algorithms" matches "machine learning techniques"
- **Synonym handling**: "automobile" matches "car"
- **Context awareness**: "Python web frameworks" matches "Django tutorials"
- **Multilingual support**: Some models work across languages

---

## Choosing Embedding Models

### Available Models

When using `--embedding-model`, format is `provider/model`:

**Sentence Transformers (local):**
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, lightweight (default)
- `sentence-transformers/all-mpnet-base-v2` - Better quality, slower

**OpenAI:**
- `openai/text-embedding-3-small` - Balanced speed/quality
- `openai/text-embedding-3-large` - Highest quality

**Cohere:**
- `cohere/embed-v4.0` - Excellent for semantic search

### Best Practices

1. **Use the same model as clustering**: For best results, use the embedding model from the original clustering run
2. **Local models for speed**: `all-MiniLM-L6-v2` is fast and works offline
3. **API models for quality**: OpenAI/Cohere models give better results
4. **Check run parameters**: View embedding model with `lmsys runs`

---

## Web UI Search

The web viewer provides a more interactive search experience:

**Global Search** (http://localhost:3000/search):
- Search across all runs and queries
- Filter by run with URL params: `/search?runId=kmeans-200`
- Paginated results with cluster context

**Cluster-Specific Search**:
- Navigate to a cluster detail page
- Use search bar to find queries within that cluster
- Highlights matching text

**Features**:
- Real-time search as you type (SQL LIKE queries, not semantic)
- Result highlighting
- Quick navigation to clusters
- Pagination for large result sets

---

## Performance Tips

### Speed

- Use local models: `all-MiniLM-L6-v2` is fastest
- Limit results with `--n-results`
- Index is pre-built during `load --use-chroma`

### Quality

- Match the clustering embedding model
- Use larger models like `all-mpnet-base-v2`
- Increase `--n-results` to see more matches

### Cost

- Local models are free
- API models charge per embedding:
  - OpenAI: ~$0.0001 per search
  - Cohere: ~$0.0001 per search

---

## Workflow Examples

### Explore Unknown Dataset

```bash
# 1. Search for broad topics
uv run lmsys search "programming" --search-type clusters

# 2. Search for specific languages
uv run lmsys search "python" --search-type clusters
uv run lmsys search "javascript" --search-type clusters

# 3. Find individual query examples
uv run lmsys search "how to" --search-type queries --n-results 50
```

### Find Related Clusters

```bash
# 1. Search for a topic
uv run lmsys search-cluster "machine learning" \
  --run-id kmeans-200-20251008

# 2. Inspect top matches
uv run lmsys inspect kmeans-200-20251008 45
uv run lmsys inspect kmeans-200-20251008 67

# 3. Curate if needed
uv run lmsys edit merge-clusters kmeans-200-20251008 \
  --source 45,67 --target 45
```

### Two-Stage Search

Search clusters first, then queries within:

```bash
# 1. Find relevant cluster
uv run lmsys search-cluster "web scraping" \
  --run-id kmeans-200-20251008
# Output: Cluster 82

# 2. View all queries in that cluster
uv run lmsys inspect kmeans-200-20251008 82 --limit 100
```

---

## Troubleshooting

### "No results found"

Possible causes:
1. **Typo in search term**: Try variations
2. **Wrong search type**: Try both `queries` and `clusters`
3. **Run not found**: Check `--run-id` is correct
4. **No embeddings**: Run with `--use-chroma` during `load`

### Poor quality results

Solutions:
1. **Use better embedding model**: Try `all-mpnet-base-v2` or OpenAI models
2. **Match clustering model**: Use same model as original clustering
3. **Refine search terms**: Be more specific
4. **Increase results**: Use `--n-results 50` to see more options

### Slow searches

Optimizations:
1. **Use local models**: `all-MiniLM-L6-v2` is fastest
2. **Reduce results**: Lower `--n-results`
3. **Pre-build index**: Ensure ChromaDB initialized with `--use-chroma`

---

## See Also

- [Analysis Commands](analysis.md) - Inspect and list clusters
- [Clustering Guide](clustering.md) - Create searchable clusters
- [Hierarchy Guide](hierarchy.md) - Organize clusters hierarchically
- [Overview](overview.md) - Complete command reference
