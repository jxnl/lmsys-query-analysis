# Hierarchical Organization

Create multi-level cluster hierarchies using LLM-driven merging based on Anthropic's Clio methodology.

## Overview

Hierarchical organization groups base clusters into parent categories, creating a navigable tree structure. This enables:

- **Top-down exploration**: Start with broad categories, drill into specifics
- **Better organization**: Natural grouping of related clusters
- **Improved navigation**: Multi-level structure in web UI

The `merge-clusters` command uses LLMs to intelligently group clusters by semantic similarity.

---

## merge-clusters

Create a hierarchical organization of clusters.

```bash
uv run lmsys merge-clusters RUN_ID [OPTIONS]
```

### How It Works

Following Anthropic's **Clio methodology**:

1. **Neighborhood Formation**: Group similar clusters using embeddings
2. **Category Generation**: LLM proposes broader category names
3. **Deduplication**: Merge similar categories to create distinct parents
4. **Assignment**: LLM assigns each cluster to best-fit parent
5. **Refinement**: LLM refines parent names based on assigned children
6. **Iteration**: Repeat process for multiple levels

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | TEXT | anthropic/claude-sonnet-4-5-20250929 | LLM model for hierarchy generation |
| `--concurrency` | INTEGER | 4 | Concurrent LLM requests |
| `--alias` | TEXT | None | Filter by summary alias |
| `--use-chroma` | FLAG | False | Use ChromaDB for similarity search |

### Examples

Basic hierarchy creation:

```bash
uv run lmsys merge-clusters kmeans-200-20251008 --use-chroma
```

Use specific summary alias:

```bash
uv run lmsys merge-clusters kmeans-200-20251008 \
  --alias "claude-v1" --use-chroma
```

High concurrency:

```bash
uv run lmsys merge-clusters kmeans-200-20251008 \
  --concurrency 8 --use-chroma
```

### Output

The command generates a `hierarchy_run_id` in the format:

```
hier-{run_id}-{timestamp}
```

Example:
```
hier-kmeans-200-20251008-170442-20251008-180523
```

Use this ID with `show-hierarchy` to visualize the tree.

### Performance

- **Duration**: ~2-5 minutes for 200 clusters (depending on LLM speed)
- **Cost**: Approximately $0.10-0.30 for 200 clusters with Claude Haiku
- **Quality**: Better with larger, more powerful models

### Best Practices

1. **Always use `--use-chroma`**: Enables semantic similarity search for better groupings
2. **Generate summaries first**: Run `summarize` before creating hierarchy
3. **Use consistent aliases**: Filter by specific summary runs for reproducibility
4. **Review results**: Use `show-hierarchy` to inspect the tree structure

---

## show-hierarchy

Display hierarchical cluster structure as a tree.

```bash
uv run lmsys show-hierarchy HIERARCHY_RUN_ID [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-depth` | INTEGER | None | Limit tree depth shown |
| `--min-queries` | INTEGER | 0 | Hide clusters with fewer queries |

### Examples

Full hierarchy:

```bash
uv run lmsys show-hierarchy hier-kmeans-200-20251008-180523
```

Top 2 levels only:

```bash
uv run lmsys show-hierarchy hier-kmeans-200-20251008-180523 \
  --max-depth 2
```

Hide small clusters:

```bash
uv run lmsys show-hierarchy hier-kmeans-200-20251008-180523 \
  --min-queries 5
```

### Output Format

```
📁 Programming and Software Development (Level 2, 45 children)
├── 📂 Python Programming (Level 1, 12 children)
│   ├── 📄 Cluster 5: Data Science with Python (23 queries)
│   ├── 📄 Cluster 12: Web Development with Django (18 queries)
│   └── 📄 Cluster 45: Python Basics and Syntax (15 queries)
├── 📂 JavaScript and Web Development (Level 1, 8 children)
│   ├── 📄 Cluster 23: React and Frontend (31 queries)
│   └── 📄 Cluster 67: Node.js Backend (19 queries)
└── 📂 Database and SQL (Level 1, 6 children)
    ├── 📄 Cluster 89: PostgreSQL Queries (22 queries)
    └── 📄 Cluster 103: Database Design (17 queries)
```

### Understanding Levels

- **Level 0**: Leaf nodes (original clusters from clustering run)
- **Level 1**: First merge (parent categories)
- **Level 2**: Second merge (super-categories)
- **Level N**: N-th merge level

### Navigation

The tree uses symbols:
- `📁` - Top-level parent
- `📂` - Mid-level parent
- `📄` - Leaf cluster (original)

---

## Web UI Integration

Hierarchies are visualized in the web viewer at `http://localhost:3000`:

1. Navigate to a run's page
2. View available hierarchies
3. Click to explore the interactive tree:
   - **Expand/collapse** controls
   - **Visual progress bars** showing cluster sizes
   - **Color coding** by cluster size
   - **Summary statistics** (total clusters, levels, query count)

---

## Workflow Example

Complete workflow from clustering to hierarchy:

```bash
# 1. Load data
uv run lmsys load --limit 10000 --use-chroma

# 2. Cluster
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma

# 3. Generate summaries
uv run lmsys summarize kmeans-200-20251008 --use-chroma --alias "v1"

# 4. Create hierarchy
uv run lmsys merge-clusters kmeans-200-20251008 --alias "v1" --use-chroma
# Output: Created hierarchy: hier-kmeans-200-20251008-180523

# 5. View hierarchy
uv run lmsys show-hierarchy hier-kmeans-200-20251008-180523

# 6. Explore in web UI
cd web && npm run dev
# Navigate to http://localhost:3000
```

---

## Multi-Level Hierarchies

The Clio algorithm can create multiple hierarchy levels:

- **200 clusters** → ~40 level-1 parents → ~10 level-2 super-parents
- **500 clusters** → ~100 level-1 parents → ~25 level-2 → ~8 level-3

The algorithm automatically determines the optimal number of levels based on cluster count.

---

## Troubleshooting

### "No summaries found"

Run `summarize` first:

```bash
uv run lmsys summarize <RUN_ID> --use-chroma
```

### "ChromaDB not initialized"

Use `--use-chroma` during `load` or run `backfill-chroma`:

```bash
uv run lmsys backfill-chroma
```

### Poor quality groupings

Try:
1. Using a better LLM model (e.g., Claude Sonnet instead of Haiku)
2. Regenerating summaries with better prompts
3. Using contrastive summaries: `summarize --contrast-mode neighbor`

---

## Database Schema

Hierarchies are stored in the `cluster_hierarchies` table:

- `hierarchy_run_id` - Unique ID for this hierarchy
- `run_id` - Original clustering run
- `cluster_id` - Cluster being organized
- `parent_cluster_id` - Parent cluster (null for top level)
- `level` - Hierarchy level (0=leaf, 1=first merge, etc.)
- `children_ids` - JSON array of child cluster IDs
- `title` - LLM-generated title for merged clusters
- `description` - LLM-generated description

---

## See Also

- [Clustering Guide](clustering.md) - Create base clusters
- [Analysis Commands](analysis.md) - Generate summaries
- [Search Guide](search.md) - Search clusters semantically
- [Overview](overview.md) - Complete command reference
