# LMSYS CLI Documentation

Complete reference for the `lmsys` command-line interface.

## Quick Links

- **[Overview](overview.md)** - Command structure, global options, quick start
- **[Getting Started](#getting-started)** - Your first workflow

---

## Documentation by Topic

### 📦 Data Management

**[Data Loading Guide](load.md)**
- Loading LMSYS-1M dataset
- Embedding generation
- ChromaDB integration
- Performance optimization

---

### 🔍 Analysis Pipeline

**[Clustering Guide](clustering.md)**
- KMeans clustering
- HDBSCAN clustering
- Algorithm comparison
- Parameter tuning

**[Analysis Commands](analysis.md)**
- Generate LLM summaries
- List and inspect clusters
- Export results
- Contrastive analysis

**[Hierarchy Guide](hierarchy.md)**
- Create multi-level hierarchies
- Clio methodology
- Visualize hierarchy trees
- Web UI integration

---

### 🔎 Exploration

**[Search Guide](search.md)**
- Semantic query search
- Cluster search
- Embedding models
- Search optimization

---

### ✏️ Curation

**[Cluster Curation Guide](edit.md)**
- Move queries between clusters
- Rename and merge clusters
- Quality tagging
- Audit trail
- Orphaned query management

---

### 🛠️ Maintenance

**[Utilities Guide](utilities.md)**
- ChromaDB management
- Data verification
- Backfilling embeddings
- System cleanup

---

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/lmsys-query-analysis
cd lmsys-query-analysis

# Install dependencies
uv sync

# Login to Hugging Face
huggingface-cli login

# Set API keys (choose one)
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export COHERE_API_KEY="your-key"
```

### Your First Analysis

```bash
# 1. Load data (10,000 queries with embeddings)
uv run lmsys load --limit 10000 --use-chroma

# 2. Cluster queries (200 clusters)
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma

# 3. View the run ID from output, then generate summaries
uv run lmsys summarize <RUN_ID> --use-chroma

# 4. Create hierarchical organization
uv run lmsys merge-clusters <RUN_ID> --use-chroma

# 5. Explore results
uv run lmsys list-clusters <RUN_ID>
uv run lmsys show-hierarchy <HIERARCHY_RUN_ID>

# 6. Launch web viewer
cd web && npm install && npm run dev
# Open http://localhost:3000
```

---

## Command Reference

### Data Commands

| Command | Description | Guide |
|---------|-------------|-------|
| `load` | Download and load LMSYS-1M dataset | [load.md](load.md) |
| `clear` | Clear all data | [utilities.md](utilities.md#clear) |
| `backfill-chroma` | Add embeddings to existing queries | [utilities.md](utilities.md#backfill-chroma) |

### Clustering Commands

| Command | Description | Guide |
|---------|-------------|-------|
| `cluster kmeans` | MiniBatchKMeans clustering | [clustering.md](clustering.md#kmeans) |
| `cluster hdbscan` | HDBSCAN density-based clustering | [clustering.md](clustering.md#hdbscan) |

### Analysis Commands

| Command | Description | Guide |
|---------|-------------|-------|
| `summarize` | Generate LLM summaries | [analysis.md](analysis.md#summarize) |
| `runs` | List clustering runs | [analysis.md](analysis.md#list-runs) |
| `list-clusters` | List clusters in run | [analysis.md](analysis.md#list-clusters) |
| `inspect` | View cluster details | [analysis.md](analysis.md#inspect-cluster) |
| `export` | Export to CSV/JSON | [analysis.md](analysis.md#export) |

### Hierarchy Commands

| Command | Description | Guide |
|---------|-------------|-------|
| `merge-clusters` | Create cluster hierarchy | [hierarchy.md](hierarchy.md#merge-clusters) |
| `show-hierarchy` | Display hierarchy tree | [hierarchy.md](hierarchy.md#show-hierarchy) |

### Search Commands

| Command | Description | Guide |
|---------|-------------|-------|
| `search` | Semantic search | [search.md](search.md#search) |
| `search-cluster` | Search cluster summaries | [search.md](search.md#search-cluster) |

### Curation Commands

| Command | Description | Guide |
|---------|-------------|-------|
| `edit view-query` | View query details | [edit.md](edit.md#view-query) |
| `edit move-query` | Move query to cluster | [edit.md](edit.md#move-query) |
| `edit move-queries` | Batch move queries | [edit.md](edit.md#move-queries) |
| `edit rename-cluster` | Rename cluster | [edit.md](edit.md#rename-cluster) |
| `edit merge-clusters` | Merge clusters | [edit.md](edit.md#merge-clusters) |
| `edit split-cluster` | Split cluster | [edit.md](edit.md#split-cluster) |
| `edit delete-cluster` | Delete cluster | [edit.md](edit.md#delete-cluster) |
| `edit tag-cluster` | Tag quality metadata | [edit.md](edit.md#tag-cluster) |
| `edit flag-cluster` | Flag for review | [edit.md](edit.md#flag-cluster) |
| `edit history` | View edit history | [edit.md](edit.md#history) |
| `edit audit` | Full audit log | [edit.md](edit.md#audit) |
| `edit orphaned` | List orphaned queries | [edit.md](edit.md#orphaned) |
| `edit select-bad-clusters` | Find problematic clusters | [edit.md](edit.md#select-bad-clusters) |

### Utility Commands

| Command | Description | Guide |
|---------|-------------|-------|
| `chroma list-collections` | List ChromaDB collections | [utilities.md](utilities.md#chroma) |
| `chroma collection-stats` | Collection statistics | [utilities.md](utilities.md#chroma) |
| `chroma delete-collection` | Delete collection | [utilities.md](utilities.md#chroma) |
| `verify embeddings` | Verify embedding consistency | [utilities.md](utilities.md#verify) |
| `verify clusters` | Verify cluster assignments | [utilities.md](utilities.md#verify) |
| `verify summaries` | Verify summary embeddings | [utilities.md](utilities.md#verify) |

---

## Workflows

### Standard Analysis Workflow

**Goal**: Load data, cluster, analyze, organize

```bash
# 1. Data Loading
uv run lmsys load --limit 10000 --use-chroma

# 2. Clustering
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma

# 3. Summarization
uv run lmsys summarize <RUN_ID> --use-chroma --alias "v1"

# 4. Hierarchy
uv run lmsys merge-clusters <RUN_ID> --use-chroma

# 5. Review
uv run lmsys show-hierarchy <HIERARCHY_RUN_ID>
uv run lmsys list-clusters <RUN_ID> --show-examples 3
```

**Guides**: [load.md](load.md) → [clustering.md](clustering.md) → [analysis.md](analysis.md) → [hierarchy.md](hierarchy.md)

---

### Cluster Quality Improvement Workflow

**Goal**: Fix incoherent or low-quality clusters

```bash
# 1. Identify problem clusters
uv run lmsys edit select-bad-clusters <RUN_ID> --max-size 10

# 2. Inspect suspicious clusters
uv run lmsys inspect <RUN_ID> <CLUSTER_ID>

# 3. Tag quality issues
uv run lmsys edit tag-cluster <RUN_ID> --cluster-id <ID> \
  --coherence 1 --quality low --notes "Incoherent mix"

# 4. Fix issues
uv run lmsys edit move-query <RUN_ID> --query-id <ID> --to-cluster <ID>
uv run lmsys edit delete-cluster <RUN_ID> --cluster-id <ID> --orphan

# 5. Review changes
uv run lmsys edit history <RUN_ID> --cluster-id <ID>
```

**Guide**: [edit.md](edit.md)

---

### Search and Exploration Workflow

**Goal**: Find specific topics and related clusters

```bash
# 1. Search for topic
uv run lmsys search "machine learning" --search-type clusters --n-results 10

# 2. Inspect top matches
uv run lmsys inspect <RUN_ID> <CLUSTER_ID>

# 3. Find related clusters
uv run lmsys search "neural networks" --search-type clusters

# 4. Search within cluster
uv run lmsys search "pytorch" --search-type queries
```

**Guide**: [search.md](search.md)

---

## Best Practices

### Data Loading

✅ **DO**:
- Use `--use-chroma` for semantic search capabilities
- Match `--embedding-model` to your clustering needs
- Start with smaller `--limit` for testing

❌ **DON'T**:
- Load without `--use-chroma` if you plan to cluster
- Mix embedding models in same workflow
- Load full dataset without testing first

### Clustering

✅ **DO**:
- Use 100-200 clusters for balanced granularity
- Reuse embeddings with `--use-chroma`
- Add `--description` to document runs

❌ **DON'T**:
- Use too few clusters (<50) - too broad
- Use too many clusters (>500) - too fragmented
- Forget to generate summaries after clustering

### Curation

✅ **DO**:
- Always provide `--reason` for edits
- Tag clusters before deleting
- Review `edit history` after complex operations

❌ **DON'T**:
- Delete clusters without reviewing queries
- Skip quality tagging
- Make bulk changes without backup

---

## Troubleshooting

### Common Issues

| Issue | Solution | Guide |
|-------|----------|-------|
| "ChromaDB collection not found" | Run with `--use-chroma` or backfill | [utilities.md](utilities.md#backfill-chroma) |
| "No summaries found" | Run `summarize` command | [analysis.md](analysis.md#summarize) |
| Slow clustering | Increase batch sizes, use local embeddings | [clustering.md](clustering.md#performance-optimization) |
| Out of memory | Reduce batch/chunk sizes | [load.md](load.md), [clustering.md](clustering.md) |
| Poor search results | Match embedding model to clustering | [search.md](search.md#choosing-embedding-models) |

---

## Additional Resources

- **Main Documentation**: [../index.md](../index.md)
- **API Reference**: [../api/](../api/)
- **Web Viewer**: `cd web && npm run dev`
- **GitHub**: [Repository](https://github.com/your-org/lmsys-query-analysis)

---

## Version Information

- **CLI Version**: Check with `uv run lmsys --help`
- **Python**: 3.10+
- **Dependencies**: See `pyproject.toml`

---

_Last updated: October 2025_
