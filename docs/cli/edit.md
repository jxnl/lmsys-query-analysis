# Cluster Curation Commands

The `lmsys edit` command group provides interactive tools for curating and fixing cluster quality issues. These commands enable iterative refinement of clustering results without requiring full re-runs.

## Philosophy

**Agentic Cluster Curation** gives you (and Claude) direct CRUD operations on clustering data. Instead of automated cleanup logic, these tools provide simple primitives that can be composed to achieve complex curation goals.

Key principles:
- **Simple primitives**: Basic CRUD operations, not high-level "cleanup" commands
- **Composable**: Combine primitives to achieve complex goals
- **Observable**: Tools print what they did for full transparency
- **Auditable**: All changes logged with WHO, WHAT, and WHY

## Command Groups

### Query Operations

View and move individual queries between clusters.

#### `view-query`

View detailed information about a query, including all cluster assignments.

```bash
uv run lmsys edit view-query <QUERY_ID>
```

**Example:**
```bash
uv run lmsys edit view-query 12345
```

**Output:**
- Query text, model, language, conversation ID
- All cluster assignments across different runs
- Confidence scores

#### `move-query`

Move a single query from one cluster to another within a run.

```bash
uv run lmsys edit move-query <RUN_ID> \
  --query-id <ID> \
  --to-cluster <CLUSTER_ID> \
  [--reason "explanation"]
```

**Example:**
```bash
uv run lmsys edit move-query kmeans-200-20251008 \
  --query-id 1234 \
  --to-cluster 67 \
  --reason "Query is about Python, not JavaScript"
```

#### `move-queries`

Batch move multiple queries to a cluster.

```bash
uv run lmsys edit move-queries <RUN_ID> \
  --query-ids <ID1,ID2,ID3> \
  --to-cluster <CLUSTER_ID> \
  [--reason "explanation"]
```

**Example:**
```bash
uv run lmsys edit move-queries kmeans-200-20251008 \
  --query-ids 123,456,789 \
  --to-cluster 42 \
  --reason "All queries about React hooks"
```

---

### Cluster Operations

Rename, merge, split, and delete clusters.

#### `rename-cluster`

Update the title and/or description of a cluster.

```bash
uv run lmsys edit rename-cluster <RUN_ID> \
  --cluster-id <ID> \
  [--title "New Title"] \
  [--description "New Description"]
```

**Example:**
```bash
uv run lmsys edit rename-cluster kmeans-200-20251008 \
  --cluster-id 143 \
  --title "Python Data Science Questions" \
  --description "Questions about pandas, numpy, matplotlib, and scikit-learn"
```

#### `merge-clusters`

Merge multiple source clusters into a target cluster.

```bash
uv run lmsys edit merge-clusters <RUN_ID> \
  --source <ID1,ID2,ID3> \
  --target <TARGET_ID> \
  [--new-title "Merged Title"] \
  [--new-description "Merged Description"]
```

**Example:**
```bash
uv run lmsys edit merge-clusters kmeans-200-20251008 \
  --source 44,133,295 \
  --target 44 \
  --new-title "Chemical Industry Content"
```

**Behavior:**
- All queries from source clusters moved to target
- Source clusters remain in database but have 0 queries
- Target cluster summary updated if new title/description provided

#### `split-cluster`

Split a subset of queries from a cluster into a new cluster.

```bash
uv run lmsys edit split-cluster <RUN_ID> \
  --cluster-id <ID> \
  --query-ids <ID1,ID2,ID3> \
  --new-title "New Cluster Title" \
  --new-description "New Cluster Description"
```

**Example:**
```bash
uv run lmsys edit split-cluster kmeans-200-20251008 \
  --cluster-id 20 \
  --query-ids 101,102,103,104,105 \
  --new-title "Finance and Investment Queries" \
  --new-description "Separated from general career development"
```

**Behavior:**
- Creates a new cluster with next available cluster ID
- Moves specified queries to new cluster
- Original cluster retains remaining queries

#### `delete-cluster`

Delete a cluster, either orphaning its queries or moving them to another cluster.

```bash
uv run lmsys edit delete-cluster <RUN_ID> \
  --cluster-id <ID> \
  [--orphan | --move-to <CLUSTER_ID>] \
  [--reason "explanation"]
```

**Example (orphan):**
```bash
uv run lmsys edit delete-cluster kmeans-200-20251008 \
  --cluster-id 23 \
  --orphan \
  --reason "Test pollution - greeting messages"
```

**Example (reassign):**
```bash
uv run lmsys edit delete-cluster kmeans-200-20251008 \
  --cluster-id 23 \
  --move-to 42 \
  --reason "Merging into larger category"
```

**Behavior:**
- Deletes cluster summaries
- If `--orphan`: queries moved to `orphaned_queries` table
- If `--move-to`: queries reassigned to target cluster

---

### Metadata Operations

Tag clusters with quality ratings, coherence scores, and flags.

#### `tag-cluster`

Add or update quality metadata for a cluster.

```bash
uv run lmsys edit tag-cluster <RUN_ID> \
  --cluster-id <ID> \
  [--coherence <1-5>] \
  [--quality <high|medium|low>] \
  [--notes "Free-form text"]
```

**Example:**
```bash
uv run lmsys edit tag-cluster kmeans-200-20251008 \
  --cluster-id 143 \
  --coherence 2 \
  --quality low \
  --notes "Incoherent mix of unrelated topics"
```

**Metadata Fields:**
- **coherence**: 1-5 scale (1=incoherent, 5=highly coherent)
- **quality**: `high`, `medium`, or `low`
- **notes**: Free-form annotations

#### `flag-cluster`

Add a flag to a cluster for review or categorization.

```bash
uv run lmsys edit flag-cluster <RUN_ID> \
  --cluster-id <ID> \
  --flag "flag_name"
```

**Common Flags:**
- `language_mixing`: Queries in multiple languages
- `needs_review`: Requires manual inspection
- `test_pollution`: Contains test/debugging queries
- `greeting_spam`: Generic greetings (hi, hello, test)

**Example:**
```bash
uv run lmsys edit flag-cluster kmeans-200-20251008 \
  --cluster-id 283 \
  --flag "language_mixing"
```

---

### Audit Operations

View edit history and orphaned queries.

#### `history`

Show edit history for a specific cluster or entire run.

```bash
uv run lmsys edit history <RUN_ID> [--cluster-id <ID>]
```

**Example (cluster-specific):**
```bash
uv run lmsys edit history kmeans-200-20251008 --cluster-id 143
```

**Example (all edits):**
```bash
uv run lmsys edit history kmeans-200-20251008
```

**Output:**
- Timestamp, edit type, editor, reason
- Detailed change information (old → new)

#### `audit`

Full audit log for a run with optional date filtering.

```bash
uv run lmsys edit audit <RUN_ID> [--since "YYYY-MM-DD"]
```

**Example:**
```bash
uv run lmsys edit audit kmeans-200-20251008 --since "2025-10-08"
```

#### `orphaned`

List all orphaned queries for a run.

```bash
uv run lmsys edit orphaned <RUN_ID>
```

**Example:**
```bash
uv run lmsys edit orphaned kmeans-200-20251008
```

**Output:**
- Query ID, text, original cluster, orphaned timestamp, reason

---

### Batch Operations

Find and select clusters matching quality criteria.

#### `select-bad-clusters`

Find clusters that match quality filters (useful for bulk operations).

```bash
uv run lmsys edit select-bad-clusters <RUN_ID> \
  [--max-size <N>] \
  [--min-size <N>] \
  [--min-languages <N>] \
  [--quality <high|medium|low>]
```

**Example:**
```bash
uv run lmsys edit select-bad-clusters kmeans-200-20251008 \
  --max-size 10 \
  --quality low
```

**Filters:**
- `--max-size`: Clusters with ≤ N queries
- `--min-size`: Clusters with ≥ N queries
- `--min-languages`: Clusters with ≥ N distinct languages
- `--quality`: Only clusters with specific quality rating

---

## Workflow Examples

### Example 1: Fix Incoherent Cluster

**Problem:** Cluster 143 contains unrelated queries.

```bash
# 1. Inspect cluster
uv run lmsys inspect kmeans-200-20251008 143

# 2. Tag as low quality
uv run lmsys edit tag-cluster kmeans-200-20251008 \
  --cluster-id 143 \
  --coherence 1 \
  --quality low \
  --notes "Multiple unrelated themes"

# 3. Move misplaced queries to correct clusters
uv run lmsys edit move-query kmeans-200-20251008 \
  --query-id 203 \
  --to-cluster 67 \
  --reason "Virology query belongs in scientific research cluster"

# 4. Delete remaining incoherent cluster
uv run lmsys edit delete-cluster kmeans-200-20251008 \
  --cluster-id 143 \
  --orphan \
  --reason "Remaining queries are incoherent"
```

### Example 2: Merge Similar Clusters

**Problem:** Clusters 44, 133, and 295 all contain chemical industry queries.

```bash
# 1. View clusters
uv run lmsys list-clusters kmeans-200-20251008 | grep -i chemical

# 2. Merge into single cluster
uv run lmsys edit merge-clusters kmeans-200-20251008 \
  --source 133,295 \
  --target 44 \
  --new-title "Chemical Industry and Biochemistry" \
  --new-description "Questions about industrial chemistry, biochemical processes, and chemical engineering"

# 3. View audit log
uv run lmsys edit history kmeans-200-20251008 --cluster-id 44
```

### Example 3: Clean Up Language Mixing

**Problem:** Find and flag all clusters with language mixing.

```bash
# 1. Find clusters with many languages
uv run lmsys edit select-bad-clusters kmeans-200-20251008 \
  --min-languages 5

# 2. Inspect a problem cluster
uv run lmsys inspect kmeans-200-20251008 283

# 3. Flag for review
uv run lmsys edit flag-cluster kmeans-200-20251008 \
  --cluster-id 283 \
  --flag "language_mixing"

# 4. Tag with quality rating
uv run lmsys edit tag-cluster kmeans-200-20251008 \
  --cluster-id 283 \
  --quality low \
  --notes "18 different languages - needs splitting by language"
```

---

## Web UI Integration

All curation data is visible in the web viewer:

- **Cluster Detail Page**: View metadata panel (coherence, quality, flags, notes) and edit history panel
- **Orphaned Queries Page**: `/runs/<RUN_ID>/orphaned` - Browse all orphaned queries
- **Audit Log Page**: `/runs/<RUN_ID>/audit` - Full edit history with filtering

Access at `http://localhost:3000` after running `cd web && npm run dev`.

---

## Database Schema

Curation operations create entries in three new tables:

### `cluster_edits`
Audit trail for all operations. Tracks WHO changed WHAT and WHY.

### `cluster_metadata`
Quality annotations: coherence scores, quality ratings, flags, notes.

### `orphaned_queries`
Queries removed from clusters during deletion operations.

See `CLAUDE.md` for full schema documentation.

---

## Best Practices

1. **Always provide reasons**: Use `--reason` to document why changes are made
2. **Tag before deleting**: Use `tag-cluster` to mark quality issues before deletion
3. **Review audit logs**: Check `edit history` after complex operations
4. **Use batch operations**: Find problematic clusters with `select-bad-clusters` before manual curation
5. **Web UI for review**: Use web viewer to visually inspect changes

---

## See Also

- [CLI Overview](overview.md)
- [Analysis Commands](analysis.md)
- [Clustering Commands](clustering.md)
