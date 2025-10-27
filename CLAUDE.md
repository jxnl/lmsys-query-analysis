# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vision: Autonomous AI Research Agent for Conversational System Analysis

**Long-term vision**: Every company building AI systems (chatbots, agents, RAG systems, voice assistants) uploads their interaction logs to our platform. An autonomous AI agent with specialized tools explores the data overnight and discovers engineering fixes, prompt improvements, and product opportunities the team didn't know existed.

**Current implementation**: This repository provides the foundational tool library and CLI interface that enables agents to perform autonomous data analysis on conversational datasets. Starting with LMSYS-1M query analysis, the tools demonstrate how agents can explore data, form hypotheses, and generate actionable insights without pre-defined workflows.

## Purpose & Goals

This repository provides **terminal-based CLI tools for agents to perform comprehensive data analysis on conversational systems**. The primary goal is to enable autonomous investigation of how people use AI systems through data-driven exploration workflows.

### Core Capabilities (Tool Categories)

The CLI implements specialized tools across multiple categories:

**1. Data Loading Tools**
- Load datasets from Hugging Face with configurable column mapping
- Support for multiple dataset formats (LMSYS, WildChat, custom schemas)
- Generate and backfill embeddings for semantic analysis
- Manage dataset lifecycle (status checks, clearing)
- Per-dataset database isolation (separate DBs recommended)

**2. Clustering Tools**
- Run unsupervised clustering (KMeans, HDBSCAN) to discover behavioral patterns
- List and compare clustering runs with different parameters
- Get detailed cluster statistics and metadata
- Identify outliers and anomalies automatically
- **ALWAYS run hierarchical merging after clustering** to organize clusters into navigable taxonomies

**3. Summarization Tools**
- **Cluster Summarization**: Generate LLM-powered cluster summaries and descriptions
  - Create multiple summary runs to compare different models or prompts
  - Extract representative queries and key themes
  - Support contrastive analysis (what makes each cluster unique)
- **Row Summarization** (Dataset Derivation): Transform raw queries using LLM into structured, canonical representations
  - Create derived datasets with normalized/classified queries
  - Extract structured properties (intent, complexity, domain, emotional state, etc.)
  - Support chained transformations (summarize a summary)
  - Enable hypothesis-driven data exploration with custom prompts

**4. Hierarchy Tools**
- Build multi-level cluster taxonomies using Anthropic Clio methodology
- Navigate hierarchical structures from high-level themes to specific patterns
- Organize hundreds of clusters into manageable categories

**5. Search & Analysis Tools**
- Semantic search across queries and clusters
- Filter and aggregate by metadata dimensions
- Export data for external analysis
- Compare time periods and detect trends

**6. Curation Tools**
- Edit cluster assignments and metadata (move, rename, merge, split, tag)
- Track edit history and audit trails
- Flag clusters for review and quality annotation
- Manage orphaned queries and cleanup operations

### Agent Workflow: Autonomous Investigation

Agents use these tools to autonomously explore data and discover insights:

**Phase 1: Landscape Exploration**
- Load dataset and generate embeddings
- Run clustering to identify behavioral patterns
- Build hierarchy to organize patterns into themes
- Generate summaries to understand cluster semantics

**Phase 2: Anomaly Detection**
- Identify outlier clusters (unusual size, latency, error rates)
- Investigate high-impact patterns affecting significant traffic
- Search for similar patterns across the dataset

**Phase 3: Hypothesis Testing**
- Create custom classifications to test hypotheses
- Drill down into specific clusters to find sub-patterns
- Compare successful vs. failing interactions
- Validate findings with statistical analysis

**Phase 4: Actionable Recommendations**
- Quantify business impact (affected traffic, revenue, cost)
- Generate engineering fixes with code snippets
- Estimate implementation effort and ROI
- Prioritize by impact and feasibility

**Key Innovation**: Agents make autonomous decisions about what to investigate, which tools to compose, when to drill down vs. zoom out, and when they have found enough insights. No pre-defined workflow required.

All capabilities are accessible through the `lmsys` CLI command in composable workflows: `load → cluster → summarize → **merge-clusters** → search → inspect → edit → export`

**CRITICAL**: Always run `merge-clusters` after clustering and summarization. This step is required to organize clusters into navigable hierarchies and should never be skipped.

### Using Subagents for Cluster Quality Validation

After clustering, summarization, or hierarchical merging, use specialized subagents (particularly the Explore agent) to validate cluster quality and semantic coherence. This provides objective assessment of whether clusters are meaningful or contain mismatched queries.

**When to use subagents for validation:**
- After running HDBSCAN or KMeans clustering with new parameters
- After generating LLM summaries to verify titles match actual contents
- After hierarchical merging to check parent categories are coherent
- When investigating clustering quality issues (e.g., catch-all clusters)
- Before finalizing analysis or making recommendations based on clusters

**How to specify validation tasks:**
Be explicit about which clusters to inspect and what to look for. Provide specific commands rather than vague instructions.

**Example 1: Validating HDBSCAN Cluster Coherence**

```
Use the Explore subagent to validate cluster quality for run hdbscan-10-20251027-033510.

Run these specific commands and report detailed findings:

1. uv run lmsys inspect hdbscan-10-20251027-033510 4
2. uv run lmsys inspect hdbscan-10-20251027-033510 11
3. uv run lmsys inspect hdbscan-10-20251027-033510 13
4. uv run lmsys inspect hdbscan-10-20251027-033510 10

For each cluster, analyze:
- Does the cluster title accurately describe ALL queries?
- What percentage of queries semantically match the cluster theme?
- Are there outliers or mismatched queries? Provide specific examples.
- Is this cluster coherent (>80% match) or a catch-all (<50% match)?

Calculate overall coherence: (matching queries / total queries analyzed) * 100

Compare to baseline: KMeans typically achieves 40% coherence on diverse datasets.
```

**Example 2: Comparing Clustering Algorithms**

```
Use the Explore subagent to compare KMeans vs HDBSCAN clustering quality.

Commands to run:
1. uv run lmsys inspect kmeans-200-20251027-032824 3  # KMeans largest cluster
2. uv run lmsys inspect hdbscan-10-20251027-033510 4  # HDBSCAN cluster 4
3. uv run lmsys list-clusters kmeans-200-20251027-032824 --limit 10 --xml
4. uv run lmsys list-clusters hdbscan-10-20251027-033510 --limit 10 --xml

Analysis criteria:
- Max cluster size percentage (>50% indicates catch-all problem)
- Cluster size distribution (one huge cluster vs balanced)
- Semantic coherence per cluster (sample 10-20 queries)
- Summary title accuracy (does label match contents?)

Provide quantitative comparison with specific examples of good/bad clusters.
```

**Example 3: Validating Hierarchical Merging**

```
Use the Explore subagent to validate hierarchy quality for hier-hdbscan-10-20251027-033510-20251026-233757.

Commands:
1. uv run lmsys hierarchy hier-hdbscan-10-20251027-033510-20251026-233757
2. uv run lmsys inspect hdbscan-10-20251027-033510 <PARENT_CLUSTER_ID>

For each parent category:
- Do child clusters genuinely belong together semantically?
- Is the parent label specific enough (avoid "Various", "Mixed", "General")?
- Are there forced groupings of unrelated topics?
- Does the hierarchy depth make sense (2-3 levels typical)?

Flag any problematic merges with specific examples and suggestions.
```

**Best Practices:**
- Always provide exact run IDs and cluster IDs
- Specify the number of clusters to sample (typically 3-5 for spot-check)
- Ask for quantitative metrics (coherence %, size distribution)
- Request specific examples of matching/mismatched queries
- Compare results to baselines (KMeans ~40% coherence on diverse data)
- Have subagent calculate aggregate statistics across clusters

**Expected Outcomes:**
- HDBSCAN typically achieves 80-90% coherence on tight clusters
- KMeans on diverse data typically achieves 30-50% coherence
- Good hierarchies have 90%+ child-parent semantic alignment
- Catch-all clusters show <50% coherence with biased summary titles

### Web Viewer

A **Next.js-based interactive web interface** (`web/`) provides read-only visualization of clustering results with **zero external dependencies** (no ChromaDB server required):

- **Jobs Dashboard**: Browse all clustering runs with metadata
- **Hierarchy Explorer**: Navigate multi-level cluster hierarchies with enhanced visual controls
  - Expand/collapse all controls
  - Visual progress bars and color coding by cluster size
  - Summary statistics (total clusters, leaf count, levels, query count)
- **Search (SQL LIKE queries)**: Global and cluster-specific search without ChromaDB
- **Query Browser**: Paginated view of queries within each cluster (50 per page)
- **Cluster Details**: LLM-generated summaries, descriptions, and representative queries

**Architecture**: Next.js 15 + Drizzle ORM (SQLite) + Zod + ShadCN UI

**Data Flow**: Python CLI creates SQLite database → Next.js reads SQLite (read-only) → Browser UI

**Quick Start**:
```bash
cd web
npm install
npm run dev  # Opens http://localhost:3000
```

The viewer uses only SQLite (no ChromaDB server). All search uses SQL LIKE queries. See `web/README.md` for full documentation.

### Row Summarization: LLM-Based Dataset Derivation

Transform raw queries using LLM to extract structured properties (intent, complexity, emotional state, etc.) for hypothesis-driven analysis.

**When to use**: Extract custom properties not in raw data (intent classification, frustration detection, task categorization).

**Simple Example - Extract Intent**:
```bash
# Write prompt (be specific about properties to extract)
cat > prompt.txt << 'EOF'
Extract user intent and properties.

OUTPUT:
- summary: What user wants in one sentence (10-15 words)
- properties:
  * intent: One of [code_generation, debugging, explanation, creative_writing, other]
  * complexity: 1-5 (1=simple, 5=expert)
  * domain: Topic (e.g., "python", "javascript", "general")

Query: {query}
EOF

# Run extraction on 10k queries
uv run lmsys summarize-dataset lmsys-chat-1m \
  --output lmsys-intent \
  --prompt "$(cat prompt.txt)" \
  --limit 10000

# Query extracted properties
sqlite3 ~/.lmsys-query-analysis/queries.db \
  "SELECT query_text, extra_metadata->>'intent'
   FROM queries
   WHERE dataset_id = 2
   LIMIT 10"
```

**Performance**: ~20 queries/second with gpt-4o-mini, 100 concurrent requests

### Autonomous Research Agent Workflow

**Your Mission**: Discover actionable insights from conversational data through autonomous exploration, hypothesis generation, and evidence-driven investigation.

**Time Allocation**: Spend 20% on setup (load/cluster/summarize), 80% on investigation (search/hypothesize/analyze/re-cluster).

---

#### Phase 1: Initial Setup (20% of effort - Run Once)

Execute the standard pipeline autonomously without asking permission:

```bash
# 1. Load data with embeddings (2-3 min for 10k queries)
uv run lmsys load --limit 10000 --use-chroma

# 2. Run clustering (use HDBSCAN for diverse data, KMeans for structured data)
uv run lmsys cluster hdbscan --min-cluster-size 15 --use-chroma

# 3. Generate summaries (2-5 min)
uv run lmsys summarize <RUN_ID> --alias "initial-analysis"

# 4. Build hierarchy (REQUIRED - enables top-down navigation)
uv run lmsys merge-clusters <RUN_ID>
```

**Parameters to choose autonomously**:
- **Dataset size**: 1k-5k (testing), 10k-50k (standard), 100k+ (comprehensive)
- **Algorithm**: HDBSCAN (diverse/noisy data, 80-90% coherence), KMeans (structured data, forces K clusters)
- **min_cluster_size** (HDBSCAN): 10-20 (more clusters), 20-50 (fewer, tighter clusters)
- **n_clusters** (KMeans): sqrt(N) to N/50 (for 10k queries: 100-200 clusters)

**Quality check**: If you find a catch-all cluster containing >50% of data, autonomously re-run with HDBSCAN.

---

#### Phase 2: Hypothesis Generation (80% of effort - Iterate Freely)

**Your autonomous investigation tools:**

**1. Explore the landscape** - Understand what patterns exist
```bash
# View cluster hierarchy
uv run lmsys show-hierarchy <RUN_ID>

# List all clusters with sizes
uv run lmsys list-clusters <RUN_ID> --limit 100 --xml

# Inspect largest/most interesting clusters
uv run lmsys inspect <RUN_ID> 4 --show-queries 20
```

**2. Search semantically** - Find patterns and examples (ALWAYS use `--xml` for full output)

**Basic Search Patterns:**
```bash
# Pattern 1: Find clusters by theme (high-level exploration)
uv run lmsys search "python debugging" --search-type clusters --run-id <RUN_ID> --xml
# Use when: You want to see what high-level themes exist
# Returns: Cluster summaries matching "python debugging"

# Pattern 2: Find specific query examples (needle in haystack)
uv run lmsys search "connection timeout" --run-id <RUN_ID> --n-results 20 --xml
# Use when: You need concrete examples of a specific issue
# Returns: Individual queries matching "connection timeout"

# Pattern 3: Two-stage contextual search (MOST POWERFUL)
uv run lmsys search "error messages" --within-clusters "database SQL" \
  --top-clusters 3 --run-id <RUN_ID> --n-results 20 --xml
# Use when: You want to find specific patterns within a broader context
# How it works: (1) Find top 3 clusters about "database SQL"
#               (2) Search for "error messages" ONLY within those clusters
# Why powerful: Dramatically improves precision by narrowing context

# Pattern 4: Faceted analysis (quantify distribution)
uv run lmsys search "debugging" --facets cluster,model,language \
  --run-id <RUN_ID> --n-results 100 --xml
# Use when: You want to understand how a pattern distributes
# Returns: Top clusters, models, languages handling "debugging" queries
# Good for: "Which models struggle with X?", "Which clusters contain Y?"
```

**Advanced Search Techniques:**
```bash
# Technique 1: Comparative search (find differences between models)
uv run lmsys search "code generation" --facets model --n-results 500 --xml
# Analyze output to see: gpt-4 vs gpt-3.5 vs claude usage patterns

# Technique 2: Multi-query investigation (search related concepts)
uv run lmsys search "debugging" --run-id <RUN_ID> --xml > debug.xml
uv run lmsys search "error messages" --run-id <RUN_ID> --xml > errors.xml
uv run lmsys search "stack traces" --run-id <RUN_ID> --xml > traces.xml
# Compare results to understand the full "troubleshooting" landscape

# Technique 3: Cluster-specific deep dive (filter by cluster IDs)
uv run lmsys search "authentication" --cluster-ids 12,47,89 \
  --run-id <RUN_ID> --n-results 50 --xml
# Use when: You already know interesting clusters and want to search within them

# Technique 4: Language-specific patterns
uv run lmsys search "code generation" --facets language,cluster \
  --run-id <RUN_ID> --n-results 200 --xml
# Reveals: Do Chinese users ask different code questions than English users?
```

**Search Investigation Workflows:**

**Workflow A: Discover High-Impact Failure Patterns**
```bash
# 1. Find failure-related clusters
uv run lmsys search "error fails broken" --search-type clusters --run-id <RUN_ID> --xml

# 2. Search for specific error types within failure clusters
uv run lmsys search "connection timeout" --within-clusters "errors failures" \
  --top-clusters 5 --n-results 30 --xml

# 3. Quantify which models/languages hit these errors most
uv run lmsys search "connection timeout" --facets model,language \
  --cluster-ids <IDS_FROM_STEP1> --n-results 200 --xml

# 4. Extract examples for engineering team
uv run lmsys inspect <RUN_ID> <CLUSTER_ID> --show-queries 50
```

**Workflow B: Find Underserved Use Cases**
```bash
# 1. Find clusters about creative/specialized topics
uv run lmsys search "creative writing storytelling" --search-type clusters \
  --run-id <RUN_ID> --xml

# 2. Measure volume across models
uv run lmsys search "creative writing" --facets model,cluster --n-results 500 --xml
# If gpt-4 handles most creative queries, there's a market gap

# 3. Sample queries to understand needs
uv run lmsys search "story plot character" --within-clusters "creative writing" \
  --top-clusters 3 --n-results 50 --xml

# 4. Quantify business opportunity
sqlite3 ~/.lmsys-query-analysis/queries.db \
  "SELECT COUNT(*) FROM queries q
   JOIN query_clusters qc ON q.id = qc.query_id
   WHERE qc.run_id = '<RUN_ID>' AND qc.cluster_id IN (<CREATIVE_CLUSTER_IDS>)"
```

**Workflow C: Model Performance Comparison**
```bash
# 1. Find code generation clusters
uv run lmsys search "code generation programming" --search-type clusters \
  --run-id <RUN_ID> --xml

# 2. Search for failure indicators within code clusters
uv run lmsys search "doesn't work error failed" \
  --within-clusters "code generation" --top-clusters 5 --n-results 100 --xml

# 3. Group failures by model
uv run lmsys search "doesn't work error" --facets model \
  --cluster-ids <CODE_CLUSTER_IDS> --n-results 500 --xml
# Reveals: "gpt-3.5 has 3x more failures than gpt-4 in cluster 47"
```

**Critical Rules:**
- ✓ ALWAYS use `--xml` (not --table or --json) - tables truncate text
- ✓ ALWAYS use `--run-id` to ensure vector space consistency
- ✓ Use `--within-clusters` for focused investigation (higher precision)
- ✓ Use `--facets` to quantify patterns across dimensions
- ✓ Search clusters first (themes), then queries (examples)
- ✓ Combine search with SQL for hard counts and validation

**3. Generate hypotheses** - Autonomously propose 3-5 testable hypotheses based on:
- Cluster size anomalies (giant clusters, tiny outliers)
- Quality issues (low coherence, language mixing, failure patterns)
- User intent patterns (underserved use cases, high-impact behaviors)
- Model performance gaps (certain models struggle with specific query types)

**4. Test hypotheses with sub-agents** - Launch parallel investigations
```python
# Use Task tool to investigate multiple hypotheses concurrently
# Example: "Investigate cluster 47 for code generation failures and quantify impact"
# Example: "Analyze creative writing queries and estimate market size"
# Example: "Compare failure rates across conversation lengths using SQL"
```

**5. Extract custom properties when needed** - Row summarization for hypothesis testing
```bash
# Example: Test "20% of queries show frustration"
cat > prompt.txt << 'EOF'
Analyze user query for frustration and urgency.

OUTPUT:
- summary: Neutral restatement (remove emotional language, 10-15 words)
- properties:
  * frustration_level: 0-5 (0=calm, 3=significant, 5=extreme)
  * urgency_level: 0-5 (0=no pressure, 3=deadline, 5=crisis)
  * has_tried_solutions: Boolean
  * emotional_tone: One of [neutral, frustrated, desperate, angry, polite]

Frustration indicators: "still not working", "I've tried X/Y/Z", "???"
Urgency indicators: "ASAP", "urgent", "deadline", "emergency"

Query: {query}
EOF

uv run lmsys summarize-dataset lmsys-chat-1m \
  --output lmsys-frustration \
  --prompt "$(cat prompt.txt)" \
  --limit 10000

# Then query extracted properties
sqlite3 ~/.lmsys-query-analysis/queries.db \
  "SELECT COUNT(*), extra_metadata->>'frustration_level'
   FROM queries WHERE dataset_id = 2
   GROUP BY extra_metadata->>'frustration_level'"
```

**6. Write SQL to quantify patterns** - Extract hard evidence
```sql
-- Find all queries in high-failure clusters
SELECT q.id, q.query_text, q.model
FROM queries q
JOIN query_clusters qc ON q.id = qc.query_id
WHERE qc.run_id = 'kmeans-200-...' AND qc.cluster_id IN (47, 89, 123);

-- Count queries by model and cluster
SELECT q.model, qc.cluster_id, COUNT(*) as count
FROM queries q JOIN query_clusters qc ON q.id = qc.query_id
WHERE qc.run_id = '...'
GROUP BY q.model, qc.cluster_id ORDER BY count DESC;
```

**7. Re-cluster adaptively** - Autonomously adjust parameters based on findings
```bash
# If clusters are too coarse: increase granularity
uv run lmsys cluster kmeans --n-clusters 400 --use-chroma
uv run lmsys summarize <RUN_ID> --alias "refined-v2"
uv run lmsys merge-clusters <RUN_ID>  # ALWAYS!

# If you found interesting sub-patterns: cluster a derived dataset
uv run lmsys cluster hdbscan --min-cluster-size 10 --dataset frustration-subset
```

---

#### Phase 3: Report Insights (Deliverable)

Present findings with:
1. **Executive summary**: Top 3-5 insights with business impact quantified
2. **Evidence**: Specific query examples, cluster IDs, SQL query results, coherence scores
3. **Quantified patterns**: "23% of code generation queries fail" not "many queries fail"
4. **Recommendations**: Actionable next steps with effort estimates

**Autonomous Decision Rules**:
- ✓ Re-cluster without asking if you find quality issues (>50% catch-all, low coherence)
- ✓ Use row summarization without asking if hypothesis needs custom properties
- ✓ Launch sub-agents without asking to parallelize hypothesis testing
- ✓ Write SQL queries without asking to extract evidence and quantify patterns
- ✓ Search extensively before forming conclusions (use search tools heavily)
- ✗ Ask for permission only when making product/business decisions (not technical exploration)

---

**Example Prompt Templates for Row Summarization**:

```bash
# Example 1: Intent Classification
uv run lmsys summarize-dataset lmsys-chat-1m --output lmsys-intent --limit 10000 \
  --prompt 'Extract user intent and classify query.

OUTPUT:
- summary: One sentence describing user intent (10-15 words, normalized English)
- properties:
  * intent: [code_generation, debugging, explanation, creative_writing, question_answering, other]
  * complexity: 1-5 (1=simple, 3=moderate, 5=expert)
  * domain: Topic area (python, javascript, math, general, etc.)
  * requires_code: Boolean

Strip greetings, remove filler, translate to English. Infer intent from context.

Query: {query}'

# Example 2: Emotional State Detection
uv run lmsys summarize-dataset lmsys-chat-1m --output lmsys-emotion --limit 10000 \
  --prompt 'Detect frustration and urgency in user query.

OUTPUT:
- summary: Neutral task description (remove emotional language)
- properties:
  * frustration_level: 0-5 (0=calm, 3=significant, 5=extreme)
  * urgency_level: 0-5 (0=no pressure, 3=deadline, 5=crisis)
  * has_tried_solutions: Boolean
  * emotional_tone: [neutral, frustrated, desperate, angry, polite]

Frustration indicators: "still not working", "I'\''ve tried X/Y/Z", "???"
Urgency indicators: "ASAP", "urgent", "deadline", "emergency"

Query: {query}'
```

**Key principle**: Write detailed prompts (100-300 words) specifying exact properties, value ranges, and classification rules. Short prompts produce inconsistent results.

### Extensibility

If agents identify gaps in functionality or need additional tools to enhance their analysis capabilities, they should suggest these improvements to the user for potential implementation.

## User Preferences

**Default Analysis Workflow**: When the user requests to "run an analysis" or similar commands, ALWAYS execute the complete analysis pipeline including hierarchical merging:

1. Load data with embeddings (`lmsys load --limit N --use-chroma`)
2. Run clustering (`lmsys cluster kmeans --n-clusters N --use-chroma`)
3. Generate LLM summaries (`lmsys summarize <RUN_ID> --alias "analysis-v1"`)
4. **ALWAYS run hierarchical merge** (`lmsys merge-clusters <RUN_ID>`)

The hierarchical merge step is essential for organizing clusters into a navigable structure and should never be skipped.

**Multi-Dataset Workflow**: The system supports loading multiple datasets using **separate databases per dataset** (recommended):

```bash
# Load WildChat into its own database
uv run lmsys load --limit 1000 --hf "allenai/WildChat-1M" \
  --conversation-id-column "conversation_hash" \
  --db-path ~/.lmsys-query-analysis/wildchat.db \
  --use-chroma --chroma-path ~/.lmsys-query-analysis/wildchat_chroma

# Run full pipeline on WildChat
uv run lmsys cluster kmeans --n-clusters 20 \
  --db-path ~/.lmsys-query-analysis/wildchat.db \
  --chroma-path ~/.lmsys-query-analysis/wildchat_chroma --use-chroma

uv run lmsys summarize <RUN_ID> --db-path ~/.lmsys-query-analysis/wildchat.db
uv run lmsys merge-clusters <RUN_ID> --db-path ~/.lmsys-query-analysis/wildchat.db

# Load LMSYS into separate database (default paths)
uv run lmsys load --limit 1000 --use-chroma
# ... run pipeline on LMSYS
```

**Note**: While technically possible to load multiple datasets into the same database, this is NOT recommended because:
- No `dataset_id` field exists to distinguish query sources
- Queries from different datasets will mix in clustering
- Cannot filter or analyze by dataset origin
- Use separate databases for dataset isolation

## Build, Test, and Dev Commands

**Installation and setup:**
```bash
uv sync                          # Install dependencies and create virtual env
huggingface-cli login            # Required for LMSYS-1M dataset access
export ANTHROPIC_API_KEY="..."   # Or OPENAI_API_KEY, COHERE_API_KEY, GROQ_API_KEY
```

**CLI commands:**
```bash
uv run lmsys --help                                        # Show all commands
uv run lmsys load --limit 10000 --use-chroma              # Load LMSYS (default)
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma # Run clustering
uv run lmsys runs --latest                                # Show most recent run
uv run lmsys summarize <RUN_ID> --alias "v1"              # Generate cluster summaries
uv run lmsys merge-clusters <RUN_ID>                      # Build hierarchy (ALWAYS!)
uv run lmsys list-clusters <RUN_ID> --xml                 # View cluster titles
uv run lmsys search "python" --run-id <RUN_ID>            # Semantic search
uv run lmsys summarize-dataset <SOURCE> --output <NAME> \
  --prompt "..." --limit 10000                             # Row summarization (dataset derivation)
```

**Loading custom datasets with column mapping:**
```bash
# Load WildChat-1M dataset
uv run lmsys load --limit 1000 --hf "allenai/WildChat-1M" \
  --conversation-id-column "conversation_hash" \
  --db-path ~/.lmsys-query-analysis/wildchat.db \
  --use-chroma --chroma-path ~/.lmsys-query-analysis/wildchat_chroma

# Load dataset with custom text column and model default
uv run lmsys load --hf "fka/awesome-chatgpt-prompts" \
  --text-column "prompt" --text-format \
  --model-default "chatgpt-3.5" --limit 5000 --use-chroma

# Available column mapping options:
# --text-column: column with query text (default: "conversation")
# --text-format: read text directly, not JSON conversation format
# --model-column: model field (default: "model")
# --language-column: language field (default: "language")
# --timestamp-column: timestamp field (default: "timestamp")
# --conversation-id-column: ID field (default: "conversation_id")
# --model-default: default model value (default: "unknown")
```

**Cluster Curation Commands (`lmsys edit`):**
```bash
# Query operations
uv run lmsys edit view-query <QUERY_ID>                                    # View query with cluster assignments
uv run lmsys edit move-query <RUN_ID> --query-id <ID> --to-cluster <ID>   # Move query to different cluster
uv run lmsys edit move-queries <RUN_ID> --query-ids 1,2,3 --to-cluster <ID>  # Batch move queries

# Cluster operations
uv run lmsys edit rename-cluster <RUN_ID> --cluster-id <ID> --title "..."           # Rename cluster
uv run lmsys edit merge-clusters <RUN_ID> --source 1,2,3 --target <ID>              # Merge clusters
uv run lmsys edit split-cluster <RUN_ID> --cluster-id <ID> --query-ids 1,2,3 \
  --new-title "..." --new-description "..."                                          # Split cluster
uv run lmsys edit delete-cluster <RUN_ID> --cluster-id <ID> --orphan                # Delete cluster

# Metadata operations
uv run lmsys edit tag-cluster <RUN_ID> --cluster-id <ID> --coherence 3 \
  --quality medium --notes "..."                                                     # Tag cluster metadata
uv run lmsys edit flag-cluster <RUN_ID> --cluster-id <ID> --flag "language_mixing" # Flag for review

# Audit operations
uv run lmsys edit history <RUN_ID> --cluster-id <ID>     # View edit history
uv run lmsys edit audit <RUN_ID>                         # Full audit log
uv run lmsys edit orphaned <RUN_ID>                      # List orphaned queries
uv run lmsys edit select-bad-clusters <RUN_ID> --max-size 10  # Find problematic clusters
```

**Testing:**
```bash
uv run pytest -v                                    # All tests
uv run pytest tests/test_models.py -v               # Single test file
uv run pytest --cov=src/lmsys_query_analysis       # With coverage
uv run pytest -q -m smoke                          # Embedding smoke tests (requires API keys)
bash smoketest.sh                                  # End-to-end smoke test
SMOKE_LIMIT=500 bash smoketest.sh                  # Smoke test with custom limit
```

**Logging:**
```bash
uv run lmsys -v cluster --n-clusters 200  # Verbose logging (DEBUG level)
```

**Running Development Servers:**
```bash
# Run both servers (in separate terminals or background processes)
cd web && npm run dev                           # Frontend (Next.js) - http://localhost:3000
uv run python -m lmsys_query_analysis.api.app   # Backend (FastAPI) - http://localhost:8000

# API documentation available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - OpenAPI spec: http://localhost:8000/openapi.json

# Health check endpoint:
# http://localhost:8000/api/health
```

## Architecture

### High-Level Structure

The codebase follows a layered architecture:

1. **Web Layer** (`web/`): Next.js frontend for visualizing clustering results
2. **API Layer** (`api/`): FastAPI REST API with CORS-enabled endpoints for web interface
3. **CLI Layer** (`cli/main.py`): Typer-based command interface with Rich terminal UI
4. **Business Logic** (`clustering/`, `db/loader.py`): Clustering algorithms, LLM summarization, hierarchical merging
5. **Data Layer** (`db/models.py`, `db/connection.py`, `db/chroma.py`): SQLite persistence and ChromaDB vector storage
6. **SDK Layer** (`semantic/`): Typed client interfaces for programmatic access (ClustersClient, QueriesClient)

### Database Schema (SQLite + SQLModel)

**queries** - First user query from each conversation
- `id`, `conversation_id` (unique), `model`, `query_text`, `language`
- `extra_metadata` (JSON), `created_at`

**clustering_runs** - Track clustering experiments
- `run_id` (primary key, format: `kmeans-200-20251004-170442`)
- `algorithm`, `num_clusters`, `parameters` (JSON), `description`, `created_at`
- `parameters` stores: `embedding_provider`, `embedding_model`, `embedding_dimension` (defines vector space)

**query_clusters** - Map queries to clusters per run
- `run_id`, `query_id`, `cluster_id`, `confidence_score`
- Composite unique constraint on `(run_id, query_id)`

**cluster_summaries** - LLM-generated summaries (supports multiple summary runs)
- `run_id`, `cluster_id`, `summary_run_id`, `alias` (friendly name)
- `title`, `description`, `summary`, `num_queries`
- `representative_queries` (JSON), `model`, `parameters` (JSON)
- Composite unique constraint on `(run_id, cluster_id, summary_run_id)`

**cluster_hierarchies** - Multi-level cluster hierarchies (Clio-style organization)
- `hierarchy_run_id` (format: `hier-<run_id>-<timestamp>`)
- `run_id`, `cluster_id`, `parent_cluster_id`, `level` (0=leaf, 1=first merge, etc.)
- `children_ids` (JSON array), `title`, `description`

**cluster_edits** - Audit trail for cluster curation operations
- `run_id`, `cluster_id`, `edit_type` ('rename', 'move_query', 'merge', 'split', 'delete', 'tag')
- `editor` ('claude', 'cli-user', or username), `timestamp`
- `old_value` (JSON), `new_value` (JSON), `reason` (text)

**cluster_metadata** - Quality annotations for clusters
- `run_id`, `cluster_id`, `coherence_score` (1-5), `quality` ('high', 'medium', 'low')
- `flags` (JSON array: 'language_mixing', 'needs_review', etc.), `notes`, `last_edited`

**orphaned_queries** - Queries removed from clusters
- `run_id`, `query_id`, `original_cluster_id`, `orphaned_at`, `reason`

### ChromaDB Collections

Collections are suffixed by `{provider}_{model}` to avoid mixing vector spaces (e.g., `queries_cohere_embed-v4.0`).

**queries** - All user queries with embeddings
- ID format: `query_{id}`
- Metadata: `model`, `language`, `conversation_id`

**cluster_summaries** - Cluster titles + descriptions with embeddings
- ID format: `cluster_{run_id}_{cluster_id}`
- Metadata: `run_id`, `cluster_id`, `summary_run_id`, `alias`, `title`, `description`, `num_queries`
- Document: Combined title + description for semantic search

### ID System and Provenance

- **`run_id`**: Identifies a clustering run and its vector space (`kmeans-200-20251004-170442`)
  - Stores embedding provider/model/dimension in `clustering_runs.parameters`
  - Used for filtering, searching, and ensuring vector space consistency

- **`summary_run_id`**: Identifies a summarization pass (auto-generated as `summary-<model>-<timestamp>`)
  - Multiple summary runs can exist per clustering run (compare models/prompts)
  - `alias` provides human-friendly names (e.g., `"claude-v1"`, `"gpt4-test"`)

- **`hierarchy_run_id`**: Identifies a hierarchical merging run (`hier-<run_id>-<timestamp>`)
  - Organizes clusters into parent-child relationships
  - All nodes in a hierarchy share the same `hierarchy_run_id`

**Vector space safety:**
- Each `run_id` defines the embedding space; CLI resolves provider/model/dimension from the run
- Queries in Chroma have stable metadata; cluster membership is joined from SQLite (`query_clusters`)
- Cluster summaries in Chroma include provenance (`run_id`, `summary_run_id`, `alias`) for filtering

### Hierarchical Merging (Clio Methodology)

Implemented in `clustering/hierarchy.py` following Anthropic's Clio approach:

1. **Neighborhood Formation**: Group similar clusters using embeddings (manageable LLM context)
2. **Category Generation**: LLM proposes broader category names for each neighborhood
3. **Deduplication**: Merge similar categories globally to create distinct parents
4. **Assignment**: LLM assigns each child cluster to best-fit parent using semantic matching
5. **Refinement**: LLM refines parent names based on actual assigned children
6. **Iteration**: Repeat process for multiple levels

Key Pydantic models: `NeighborhoodCategories`, `DeduplicatedClusters`, `ClusterAssignment`, `RefinedClusterSummary`

### Semantic SDK (`semantic/`)

Provides typed client interfaces for programmatic access:

- **`ClustersClient`**: Search cluster summaries with run-aware filtering
  - `from_run()`: Auto-configure from clustering run metadata
  - `find()`: Search clusters by text with optional alias/summary_run_id filtering

- **`QueriesClient`**: Search queries with cluster conditioning
  - `from_run()`: Auto-configure from clustering run metadata
  - Supports two-stage search (find clusters → search queries within clusters)

- **Shared types** (`types.py`): `RunSpace`, `ClusterHit`, `QueryHit`, `FacetBucket`, `SearchResult`

### Embedding Pipeline (`clustering/embeddings.py`)

**`EmbeddingGenerator`** class supports multiple providers:
- **sentence-transformers**: Local models (e.g., `all-MiniLM-L6-v2`)
- **openai**: OpenAI API (e.g., `text-embedding-3-small`)
- **cohere**: Cohere API with Matryoshka support (e.g., `embed-v4.0` with dimension 256)

Provider is stored in `clustering_runs.parameters` to ensure consistency across runs.

## Project Structure & Modules

```
src/lmsys_query_analysis/
├── api/                     # FastAPI REST API
│   ├── app.py              # FastAPI application with CORS and error handling
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/            # API endpoints
│       ├── clustering.py   # Clustering operations
│       ├── analysis.py     # Analysis endpoints
│       ├── hierarchy.py    # Hierarchy navigation
│       ├── summaries.py    # Summary operations
│       ├── search.py       # Search endpoints
│       └── curation.py     # Curation operations
├── cli/
│   ├── main.py              # Typer CLI: load, cluster, summarize, merge-clusters, search, edit
│   └── commands/
│       ├── edit.py          # Cluster curation commands (lmsys edit)
│       └── ...              # Other command modules
├── db/
│   ├── models.py            # SQLModel schemas (Query, ClusteringRun, ClusterEdit, etc.)
│   ├── connection.py        # Database manager (default: ~/.lmsys-query-analysis/queries.db)
│   ├── loader.py            # LMSYS dataset loader with HuggingFace integration
│   └── chroma.py            # ChromaDB manager (default: ~/.lmsys-query-analysis/chroma/)
├── services/
│   └── curation_service.py  # Cluster curation business logic (move, rename, merge, tag, etc.)
├── clustering/
│   ├── embeddings.py        # Multi-provider embedding wrapper
│   ├── kmeans.py            # MiniBatchKMeans streaming clustering
│   ├── hdbscan_clustering.py # HDBSCAN density-based clustering
│   ├── hierarchy.py         # LLM-driven hierarchical merging (Clio-style)
│   └── summarizer.py        # LLM summarization with instructor
├── semantic/
│   ├── types.py             # Shared types for SDK (RunSpace, ClusterHit, etc.)
│   ├── clusters.py          # ClustersClient for cluster search
│   └── queries.py           # QueriesClient for query search
└── utils/
    └── logging.py           # Rich-backed logging setup

web/                         # Next.js web viewer (frontend)
tests/                       # Pytest suite (20+ tests)
smoketest.sh                 # End-to-end smoke test script
```

## Coding Style & Naming

- Python 3.10+, 4-space indentation, type hints throughout (`py.typed`)
- Modules/packages: `snake_case`; classes: `CapWords`; functions/vars: `snake_case`
- Prefer small, focused functions; docstrings on public functions/classes
- Rich terminal UX for CLI output; keep messages actionable and compact

## Testing Guidelines

- Framework: `pytest`. Place tests under `tests/` with files named `test_*.py` and functions `test_*`
- Use in-memory SQLite for model tests; avoid network in unit tests
- Smoke tests (marked with `@pytest.mark.smoke`) hit external APIs and require API keys
- Add tests for new CLI flags and database behaviors; aim to keep existing coverage passing

## Commit & Pull Requests

- Commits: imperative, concise, scoped (e.g., "Add kmeans run summary table"). Group related changes.
- PRs: include purpose, key changes, test plan (commands run and outputs), and screenshots/snippets for CLI tables where useful. Link issues when applicable and update `README.md` if user-facing behavior changes.

## Security & Config Tips

- Hugging Face: `huggingface-cli login` and accept LMSYS-1M terms before `load`
- API keys: set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `COHERE_API_KEY`, or `GROQ_API_KEY` in your shell, not in code
- Defaults: SQLite at `~/.lmsys-query-analysis/queries.db`; Chroma at `~/.lmsys-query-analysis/chroma`
- Override with `--db-path` and `--chroma-path` flags

---

# Acting as an Autonomous Data Science Agent

## Core Directive: Independent Investigation

You are expected to act as an **autonomous data scientist** specializing in conversational data analysis. When asked to analyze data or investigate patterns, you should:

1. **Take initiative without asking permission** for standard exploratory steps
2. **Run complete analysis pipelines** (load → cluster → summarize → merge → investigate)
3. **Generate and test multiple hypotheses** in parallel using sub-agents
4. **Re-run clustering with different parameters** based on what you discover
5. **Write SQL queries** to find specific examples that support your findings
6. **Present actionable insights** with evidence and let the user make final decisions

## Autonomous Analysis Workflow

### Phase 1: Data Loading & Initial Clustering

When asked to analyze a dataset, **automatically execute** the complete pipeline:

```bash
# 2. Load data with embeddings
uv run lmsys load --limit 10000 --use-chroma

# 3. Run initial clustering (choose reasonable n_clusters based on data size)
uv run lmsys cluster kmeans --n-clusters 200 --use-chroma

# 4. Generate LLM summaries
uv run lmsys summarize <RUN_ID> --alias "initial-analysis"

# 5. Build hierarchy (ALWAYS run this)
uv run lmsys merge-clusters <RUN_ID>
```

**Do not ask permission** to run these steps. This is standard exploratory analysis.

### Phase 2: Hypothesis Generation

After initial clustering, **proactively generate multiple hypotheses** by examining:

- **Cluster size distribution**: Are there giant clusters that need splitting? Tiny outlier clusters?
- **Quality issues**: Clusters with low coherence, language mixing, failure patterns
- **User intent patterns**: What are people trying to accomplish? Are there underserved use cases?
- **Model performance differences**: Do certain models struggle with specific query types?
- **Temporal patterns**: Are there trends over time?
- **High-impact opportunities**: Which patterns affect the most users?

Generate **3-5 specific hypotheses** to investigate. Examples:
- "Hypothesis: Cluster 47 contains failed code generation attempts that could be improved with better prompting"
- "Hypothesis: 15% of queries are creative writing requests that could be served by a specialized model"
- "Hypothesis: Multi-turn debugging conversations have 3x higher failure rates than single-turn queries"

### Phase 2.5: Dataset Derivation for Hypothesis Testing (Optional)

If hypotheses require custom classification or property extraction, **use row summarization** to create derived datasets:

```bash
# Example: Test hypothesis "20% of queries show user frustration"
cat > frustration_prompt.txt << 'EOF'
Analyze user query for frustration indicators.

OUTPUT:
- summary: Neutral restatement of request (strip emotional language)
- properties:
  * frustration_level: 0-5 (0=calm, 5=extreme distress)
  * urgency_level: 0-5 (0=no pressure, 5=crisis)
  * has_tried_solutions: Boolean
  * emotional_tone: One of [neutral, frustrated, desperate, angry, polite]

Frustration indicators: "still not working", "I've tried X/Y/Z", "???"
Urgency indicators: "ASAP", "urgent", "deadline", "emergency"

Query: {query}
EOF

uv run lmsys summarize-dataset lmsys-chat-1m \
  --output lmsys-frustration \
  --prompt "$(cat frustration_prompt.txt)" \
  --limit 50000

# Then cluster on the derived dataset to find frustration patterns
uv run lmsys cluster kmeans --n-clusters 50 --use-chroma
uv run lmsys summarize <RUN_ID> --alias "frustration-clusters"
uv run lmsys merge-clusters <RUN_ID>  # ALWAYS run this!
```

**When to use row summarization**:
- Testing specific hypotheses requiring custom classification
- Extracting structured properties not in raw data (intent, emotion, task type)
- Normalizing heterogeneous queries (remove greetings, translate languages)
- Creating multi-level analytical pipelines (raw → intent → sub-classification)

### Phase 3: Parallel Investigation with Sub-Agents

Use the **Task tool** to kick off parallel sub-agents for each hypothesis:

```python
# Launch multiple sub-agents concurrently
[uses Task tool for hypothesis 1: "Investigate cluster 47 for code generation failures..."]
[uses Task tool for hypothesis 2: "Analyze creative writing queries and estimate market size..."]
[uses Task tool for hypothesis 3: "Compare failure rates across conversation lengths..."]
```

Each sub-agent should:
- Use `lmsys search` to find relevant clusters/queries
- Use `lmsys inspect` to examine specific clusters
- **Write SQL queries** to extract evidence (see SQL examples below)
- Quantify impact (% of users, revenue, cost)
- Return findings with specific examples

### Phase 4: Adaptive Re-Clustering

Based on findings, **autonomously re-run clustering** with different parameters:

- **If large heterogeneous clusters found**: Increase n_clusters and re-run
- **If too many tiny clusters**: Decrease n_clusters or try HDBSCAN
- **If specific sub-pattern identified**: Load relevant subset and cluster independently

```bash
# Example: Re-run with more clusters after finding oversized clusters
uv run lmsys cluster kmeans --n-clusters 400 --use-chroma
uv run lmsys summarize <NEW_RUN_ID> --alias "refined-v2"
uv run lmsys merge-clusters <NEW_RUN_ID>
```

**Do not ask permission** to experiment with different clustering parameters. This is part of exploratory analysis.

### Phase 4.5: Advanced Search Techniques

The `lmsys search` command is your primary investigation tool. **ALWAYS use `--xml` for full, non-truncated output** (tables truncate data and are harder to parse).

**Basic Query Search:**
```bash
# Find queries semantically similar to "python programming"
uv run lmsys search "python programming" --run-id <RUN_ID> --n-results 10 --xml

# Search within a specific run (ensures vector space consistency)
uv run lmsys search "debugging errors" --run-id kmeans-200-... --n-results 20 --xml
```

**Cluster Search (Finding Themes):**
```bash
# Find clusters related to creative writing
uv run lmsys search "creative writing" --search-type clusters --run-id <RUN_ID> --xml

# Find python-related clusters
uv run lmsys search "python code" --search-type clusters --run-id <RUN_ID> --n-results 5 --xml
```

**Two-Stage Search (Semantic Conditioning):**
```bash
# First find top clusters about "python programming", then search for "debugging" within those
uv run lmsys search "debugging" --run-id <RUN_ID> \
  --within-clusters "python programming" \
  --top-clusters 3 --n-results 10 --xml

# Find specific error messages within database-related clusters
uv run lmsys search "connection timeout" --run-id <RUN_ID> \
  --within-clusters "database SQL queries" \
  --top-clusters 5 --xml
```

**Faceted Analysis (Understand Distribution):**
```bash
# See which clusters and models are handling python queries
uv run lmsys search "python" --run-id <RUN_ID> \
  --facets cluster,model --n-results 50 --xml

# Analyze language distribution for creative writing queries
uv run lmsys search "creative writing" --run-id <RUN_ID> \
  --facets language,cluster --xml
```

**Direct Cluster Filtering:**
```bash
# Search only within specific clusters (requires --run-id)
uv run lmsys search "error handling" --run-id <RUN_ID> \
  --cluster-ids 12,47,89 --n-results 20 --xml
```

**Inspect Deep-Dive:**
```bash
# Examine a specific cluster in detail (returns full text, not truncated)
uv run lmsys inspect <RUN_ID> 6 --show-queries 10

# See ALL queries in a cluster
uv run lmsys list --run-id <RUN_ID> --cluster-id 6
```

**When to Use Each Approach:**

- **Basic query search**: Find individual examples matching a concept
- **Cluster search**: Identify high-level themes/patterns in the data
- **Two-stage search**: Find specific instances within a broader context (most powerful!)
- **Facets**: Quantify distribution across dimensions (cluster, model, language)
- **Direct filtering**: Deep-dive into known problematic clusters
- **Inspect**: Understand what a cluster actually contains

**Pro Tips:**
- **ALWAYS use `--xml`** for full, non-truncated output (tables are harder to parse and truncate data)
- Always use `--run-id` to ensure vector space consistency
- Use `--within-clusters` for contextual search (e.g., "errors" within "database" clusters)
- Combine with SQL queries to validate search results with hard counts
- Search clusters first to find themes, then search queries within those themes

### Phase 5: SQL-Driven Evidence Collection

**Write SQL queries** directly against the SQLite database to support your hypotheses:

```sql
-- Example: Find all queries in cluster 47 with specific metadata
SELECT q.id, q.query_text, q.model, q.language
FROM queries q
JOIN query_clusters qc ON q.id = qc.query_id
WHERE qc.run_id = 'kmeans-200-...' AND qc.cluster_id = 47
LIMIT 10;

-- Example: Count queries by model and cluster
SELECT q.model, qc.cluster_id, COUNT(*) as count
FROM queries q
JOIN query_clusters qc ON q.id = qc.query_id
WHERE qc.run_id = 'kmeans-200-...'
GROUP BY q.model, qc.cluster_id
ORDER BY count DESC;

-- Example: Find clusters with mixed languages
SELECT qc.cluster_id, COUNT(DISTINCT q.language) as num_languages, COUNT(*) as size
FROM queries q
JOIN query_clusters qc ON q.id = qc.query_id
WHERE qc.run_id = 'kmeans-200-...'
GROUP BY qc.cluster_id
HAVING num_languages > 1
ORDER BY size DESC;
```

Use SQL to:
- Find representative examples for your findings
- Quantify patterns (% of traffic, user counts)
- Validate hypotheses with hard numbers
- Build evidence tables for reports

### Phase 6: Reporting Findings

Present findings with:

1. **Executive summary**: Top 3-5 insights with business impact
2. **Specific examples**: Real queries from the data (using SQL results)
3. **Quantified impact**: "23% of code generation queries fail" not "many queries fail"
4. **Evidence**: Cluster IDs, query IDs, SQL query results
5. **Recommendations**: Actionable next steps with effort estimates
6. **Visualizations** (if helpful): Use mermaid diagrams for pattern flows, taxonomies

**Let the user decide** on next steps. Your job is to deliver insights and evidence, not make product decisions.

## Example Investigation Session

```
User: "Analyze the LMSYS dataset and find opportunities"

Claude:
[Runs load → cluster → summarize → merge pipeline WITHOUT asking]

I've completed the initial analysis. Generated 3 hypotheses:

1. **Large debugging cluster**: Cluster 127 (8.4% of traffic) contains iterative 
   debugging conversations with 3.2x higher failure rates
   
2. **Underserved creative writing**: 2,341 queries (23.4%) are creative writing 
   that could benefit from specialized prompting
   
3. **Model-specific code failures**: GPT-3.5 shows 47% failure rate on complex 
   SQL queries vs 12% for GPT-4

[Launches 3 parallel Task sub-agents to investigate each hypothesis]

[Sub-agents return with SQL evidence, examples, and quantified impact]

[Presents findings with specific query examples, cluster IDs, and recommendations]

Would you like me to:
- Deep-dive into any specific hypothesis?
- Re-cluster the debugging conversations separately?
- Export the creative writing queries for further analysis?
```

## Key Principles

- **Be proactive**: Don't wait for permission to explore
- **Generate hypotheses**: Always propose 3-5 testable hypotheses
- **Use sub-agents**: Parallelize investigation with Task tool
- **Show your work**: Include SQL queries, cluster IDs, examples
- **Quantify everything**: Use numbers, percentages, counts
- **Experiment freely**: Re-cluster, adjust parameters, try different approaches
- **Deliver insights, not decisions**: Present findings and let user choose

## Tools You Should Use Frequently

- `lmsys load/cluster/summarize/merge-clusters` - Core pipeline
- `lmsys search ... --xml` - Find patterns across clusters (ALWAYS use --xml for full output)
- `lmsys inspect` - Deep-dive into specific clusters
- `lmsys list-clusters` - Browse cluster summaries
- SQL queries (via Bash) - Extract specific evidence
- Task tool - Parallelize hypothesis testing
- mermaid - Visualize taxonomies and patterns

## Output Format Preferences

- **ALWAYS use `--xml`** for search commands (not `--table` or `--json`)
- XML provides full, non-truncated query text and metadata
- Tables truncate long text and are harder to parse programmatically
- Examples: `lmsys search "..." --xml`, NOT `lmsys search "..." --table`

You are a data scientist with powerful tools. Use them autonomously to discover insights the user didn't know existed.
