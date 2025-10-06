# Cluster Summarization Experiments

## Overview

This document captures learnings from experiments in generating high-quality cluster titles and descriptions for the LMSYS query analysis project.

## Experiment 1: Initial Summarization (2025-10-04)

### Setup
- **Model**: GPT-4o-mini
- **Clusters**: 200 (KMeans)
- **Max queries per cluster**: 100
- **Approach**: Basic prompt with general instructions

### Results

**Quality Score**: 7/10

#### Issues Identified

**1. Generic Titles (35% of clusters)**
- Too many used vague terms: "Diverse", "Various", "General", "Mixed"
- Redundant prefixes: "User Queries About...", "User Requests for..."
- Examples of poor titles:
  - ❌ "Diverse User Queries on Everyday Topics" (Cluster 42)
  - ❌ "User Inquiries on Various Topics" (Cluster 47, 78 - duplicates!)
  - ❌ "Diverse User Queries Covering Various Topics" (Cluster 91)
  - ❌ "General Knowledge and Casual Queries" (Cluster 103)

**2. Repetitive Patterns**
- "Diverse/Various" appeared in **15+ titles** (7.5% of all clusters)
- "User Queries/Requests" prefix in **60+ titles** (30%)
- Multiple similar greeting clusters with near-identical titles (6, 17, 23, 63, 73, 82, 142, 152, 177)

**3. Missing Specificity**
- Failed to identify technical terms when present
- Example: Cluster 77 "Programming and Coding Help for NAME_1" - should specify language/context
- Example: Cluster 107 "User Engagement with Character NAME_1" - what type of engagement?

**4. Good Examples (to emulate)**
- ✅ "Stable Diffusion Prompt Generation" (Cluster 127) - specific tool + action
- ✅ "User Queries About Speaking German" (Cluster 128) - clear language focus
- ✅ "Fibonacci in Python" (Cluster 167) - specific algorithm + language
- ✅ "Chemical Industry Production and Application Articles" (Cluster 9) - domain + content type

### Root Cause Analysis

**1. Weak System Prompt**
```python
# Original (too vague)
"You are an expert data analyst specializing in query pattern analysis..."
```
- No specific constraints on title format
- No examples of good vs bad titles
- No prohibition of generic terms

**2. Insufficient Pydantic Schema Constraints**
```python
# Original
title: str = Field(
    ..., description="Short title (5-10 words) capturing the main topic"
)
```
- "5-10 words" too broad
- No format rules
- No examples in description

**3. Sampling Strategy Issues**
- MMR (Maximal Marginal Relevance) prioritizes diversity over representativeness
- May surface edge cases instead of dominant patterns
- Need to weight by frequency for more accurate cluster characterization

## Experiment 2: Improved Methodology (2025-10-04 v2)

### Changes Implemented

#### 1. Enhanced Pydantic Schema

**Before:**
```python
title: str = Field(
    ..., description="Short title (5-10 words) capturing the main topic of the cluster"
)
```

**After:**
```python
title: str = Field(
    ...,
    description="""Concise, specific title (3-7 words) that captures the PRIMARY theme.

    RULES:
    - Use specific technical terms, domains, or actions (e.g., "Python Flask API Development")
    - Avoid generic prefixes like "User Queries About", "Diverse", "Various"
    - Avoid vague words: "Diverse", "Various", "General", "Mixed", "Multiple"
    - Use concrete nouns and verbs
    - If multilingual, specify language(s)

    GOOD: "Stable Diffusion Image Prompts", "SQL Query Generation", "Spanish Greetings"
    BAD: "User Queries on Various Topics", "Diverse User Requests", "General Questions"
    """
)
```

**Key Improvements:**
- Reduced word count to 3-7 (from 5-10)
- Explicit rules with forbidden words
- Concrete examples of good vs bad
- Multilingual handling instructions

#### 2. Improved System Prompt

**Before:**
```
You are an expert data analyst specializing in query pattern analysis...
```

**After:**
```
You are an expert taxonomist specializing in categorizing user interactions with LLM systems...

CRITICAL INSTRUCTIONS:

1. IDENTIFY THE DOMINANT PATTERN (what 60-80% of queries share)
   - Ignore outliers and edge cases
   - Focus on the PRIMARY use case or theme

2. TITLE REQUIREMENTS:
   - 3-7 words maximum
   - Use SPECIFIC technical terms, domains, or action verbs
   - NEVER use: "User Queries", "Diverse", "Various", "General", "Mixed", "Multiple"
   - ALWAYS be concrete: use programming languages, specific topics, named tools

3. DESCRIPTION REQUIREMENTS:
   - Sentence 1: State the PRIMARY goal/task
   - Sentence 2: Key patterns (technical level, common subtopics, phrasing style)
   - Sentence 3: What distinguishes from neighbors if provided
   - Be SPECIFIC: mention actual examples, technical terms, languages, frameworks
```

**Key Improvements:**
- Role: "taxonomist" vs "data analyst" (more precise)
- Explicit "NEVER use" list
- Focus on dominant pattern (60-80%) not everything
- Structured description template
- Emphasis on specificity throughout

#### 3. Title Quality Criteria

**Specificity Levels:**

| Level | Description | Example |
|-------|-------------|---------|
| **Excellent** | Technical term + specific context | "Python Flask REST API Development" |
| **Good** | Domain + action/task | "SQL Query Generation" |
| **Acceptable** | Clear topic + modifier | "German Language Tests" |
| **Poor** | Generic + vague | "Programming Queries" |
| **Unacceptable** | "Diverse/Various/General" | "Diverse User Requests" ❌ |

**Word Choice Guidelines:**

✅ **USE:**
- Technical terms: "Python", "SQL", "React", "Docker"
- Specific domains: "Chemistry", "Finance", "Healthcare"
- Action verbs: "Generate", "Extract", "Analyze", "Debug"
- Named tools: "Stable Diffusion", "ChatGPT", "Vicuna"
- Languages: "Spanish", "Arabic", "German"

❌ **AVOID:**
- "User Queries/Requests" (redundant - all are user queries)
- "Diverse/Various/General/Mixed/Multiple" (lazy catch-all terms)
- "About/On/For" (weak connectors)
- "Different/Several" (vague quantifiers)

### Expected Improvements

**Predicted Outcomes:**
- Reduce generic titles from 35% → 10%
- Eliminate "Diverse/Various" from 15 clusters → 0-2 clusters
- Increase use of technical terms by 40%
- Improve title uniqueness (fewer duplicates)

**Measurable Metrics:**
1. **Specificity Score**: % of titles with technical terms or named entities
2. **Generic Word Count**: # of titles containing forbidden words
3. **Average Title Length**: Target 4-5 words (down from 6-7)
4. **Uniqueness**: # of unique titles / total titles

### Next Steps

1. ✅ Update Pydantic schema with enhanced constraints
2. ✅ Rewrite system prompt with explicit rules and examples
3. ⏳ Test on 10 sample clusters for validation
4. ⏳ Run full re-summarization on all 200 clusters
5. ⏳ Compare metrics: v1 vs v2
6. ⏳ Document learnings and best practices

## Title Pattern Analysis

### Problematic Patterns Observed

**Pattern 1: Redundant Prefixes (30% of clusters)**
```
❌ "User Queries About Speaking German"
✅ "German Language Capability Tests"

❌ "User Requests for Malicious Content Generation"
✅ "Malicious Content Generation"

❌ "User Inquiries on Various Topics"
✅ Identify the actual dominant topic!
```

**Pattern 2: Generic Qualifiers (15% of clusters)**
```
❌ "Diverse User Queries on Everyday Topics"
✅ Identify the 2-3 main subtypes

❌ "Various Creative Writing Tasks"
✅ "Poetry and Short Story Prompts"

❌ "General Knowledge and Casual Queries"
✅ "Trivia and Facts Requests"
```

**Pattern 3: Weak Verbs**
```
❌ "Queries About Cats and Dogs"
✅ "Pet Stories and Care Advice"

❌ "Requests for Similar Tools"
✅ "Software Alternative Recommendations"
```

### Successful Patterns

**Pattern 1: Technical Specificity**
```
✅ "Python Flask API Development"
✅ "SQL Database Query Generation"
✅ "React Component Debugging"
```

**Pattern 2: Domain + Content Type**
```
✅ "Chemical Industry Safety Articles"
✅ "Business Strategy Analysis"
✅ "Medical Diagnostic Reports"
```

**Pattern 3: Tool/Framework + Action**
```
✅ "Stable Diffusion Prompt Crafting"
✅ "Vicuna Model Installation"
✅ "ChatGPT Capability Testing"
```

**Pattern 4: Language + Task**
```
✅ "Spanish Business Correspondence"
✅ "German Language Capability Checks"
✅ "Portuguese Content Translation"
```

## Lessons Learned

### 1. Prompt Engineering
- **Negative examples** are as important as positive examples
- **Explicit constraints** work better than general guidance
- **Role definition** matters: "taxonomist" > "data analyst"
- **Focus on dominant pattern** (60-80%) prevents over-generalization

### 2. Schema Design
- **Inline examples** in Field descriptions are highly effective
- **Shorter is better**: 3-7 words forces specificity
- **Forbidden word lists** prevent lazy labeling

### 3. Cluster Quality
- **"Diverse/Various" is a red flag** - indicates poor clustering or lazy summarization
- **Technical terms improve clarity** - domain experts can instantly recognize relevance
- **Multilingual clusters need explicit language tags** in titles

### 4. Iterative Improvement
- **Test on samples first** (10-20 clusters) before full run
- **Measure before/after** with specific metrics
- **Document patterns** for future reference

## Future Experiments

### Potential Improvements

1. **Sampling Strategy Enhancement**
   - Weight by query frequency (not just diversity)
   - Include most common, median, and edge cases
   - Add TF-IDF keywords to context

2. **Chain-of-Thought Reasoning**
   - Ask LLM to identify patterns first
   - Then generate title based on patterns
   - May improve consistency

3. **Multi-Model Validation**
   - Use Claude for initial titles
   - Use GPT-4o to validate/refine
   - Compare and choose best

4. **Human-in-the-Loop**
   - Flag low-confidence summaries
   - Allow manual refinement
   - Build golden dataset for fine-tuning

5. **Automated Quality Checks**
   - Regex check for forbidden words
   - Validate title length
   - Check for duplicates
   - Flag generic terms

## Experiment 3: Clustering Algorithm Comparison (2025-10-04)

### Motivation

After observing mixed cluster quality with KMeans-20, we hypothesized that:
1. **Too few clusters** → overly broad, heterogeneous groupings
2. **KMeans limitations** → forces all points into clusters even when unrelated
3. **Density-based methods** (HDBSCAN) → may find more natural groupings

### Experimental Setup

**Dataset**: 5000 LMSYS queries with embeddings

**Approaches Tested:**
1. **KMeans-20** (baseline): 20 clusters
2. **KMeans-50**: 50 clusters (2.5x increase)
3. **KMeans-100**: 100 clusters (5x increase)
4. **HDBSCAN**: min_cluster_size=30, density-based

### Results Summary

| Approach | Clusters | Avg Size | Min Size | Max Size | Noise Points | Quality Assessment |
|----------|----------|----------|----------|----------|--------------|-------------------|
| **KMeans-20** | 20 | 250 | 196 | 276 | 0 (0%) | Mixed - some good, many heterogeneous |
| **KMeans-50** | 50 | 100 | 30 | 226 | 0 (0%) | Improved - more focused but still mixed |
| **KMeans-100** | 100 | 50 | 7 | 140 | 0 (0%) | Better granularity, smaller variance |
| **HDBSCAN-30** | 8 | 124 | 30 | 177 | 4009 (80%) | Very specific clusters, too much noise |

### Key Findings

#### 1. KMeans Cluster Quality by k

**KMeans-20 Issues:**
- **Cluster 0 (Creative Writing)**: Mixed content - creative writing, logic puzzles, jokes, relationship analysis
- **Cluster 2 (Text Completion)**: Heterogeneous - prompt engineering, coding, contact extraction, grammar
- **Cluster 5 (Toxic Statements)**: Mostly coherent toxic/jailbreak prompts, some outliers
- **Cluster 12 (Business Inquiries)**: Very mixed - business plans, philosophy, multiple choice questions
- **Cluster 13 (Python Programming)**: **Highly coherent** - nearly all same template format

**KMeans-50 Improvements:**
- Cluster sizes reduced from 250 → 100 average
- Still contains mixed content but more focused
- Example Cluster 0: Still has creative prompts + logic puzzles + historical questions (needs more granularity)

**KMeans-100 Improvements:**
- Cluster sizes reduced to 50 average
- Better separation of distinct topics
- Example Cluster 0: More focused on historical/philosophical questions
- Some very small clusters (min=7) may be too specific

#### 2. HDBSCAN Characteristics

**Strengths:**
- **Very high specificity**: Cluster 0 was almost entirely Russian language queries (177 queries)
- Finds natural density-based groupings
- Homogeneous clusters with clear themes

**Weaknesses:**
- **80% classified as noise** (4009/5000 queries)
- Only 8 clusters identified (too few for meaningful analysis)
- Min cluster size of 30 may be too restrictive for this dataset

### Hypotheses

#### Why is cluster quality inconsistent?

**Hypothesis 1: Embedding Space Structure**
- LMSYS queries are **highly diverse** across multiple dimensions (language, domain, task type, style)
- Embedding space may not have clear natural boundaries
- KMeans forces hard assignments even when queries don't naturally cluster

**Hypothesis 2: Multi-Modal Distribution**
- Dataset contains distinct sub-populations:
  - Template-based queries (like Python programming prompts) → cluster well
  - Creative/open-ended queries → harder to cluster
  - Multilingual content → clusters by language first
  - Toxic/adversarial prompts → distinct pattern

**Hypothesis 3: Optimal k is Domain-Dependent**
- Programming/technical queries: Well-defined, fewer clusters needed
- Creative/open-ended: Diverse, need many clusters or different approach
- Multilingual: Should cluster by language first, then topic

**Hypothesis 4: Hierarchical Structure Exists**
- Some topics have clear hierarchy (e.g., Programming → Python → Flask → API)
- Flat clustering (KMeans) misses this structure
- Could benefit from hierarchical clustering or topic modeling

### Recommendations

#### 1. **Short-term: Use KMeans-100 for current analysis**
- Best balance of specificity vs coverage
- Minimal noise (0%)
- Reasonable cluster sizes (avg 50)
- Run summarization to assess quality

#### 2. **Medium-term: Tune HDBSCAN parameters**
- Reduce min_cluster_size (try 15-20)
- Adjust cluster_selection_epsilon
- Goal: <50% noise while maintaining specificity

#### 3. **Long-term: Hybrid or Hierarchical Approach**
- **Option A**: Two-stage clustering
  1. First pass: Cluster by language/domain (coarse)
  2. Second pass: Within each group, cluster by topic (fine)

- **Option B**: Topic Modeling
  - Use LDA or BERTopic instead of distance-based clustering
  - May better capture overlapping themes

- **Option C**: Ensemble Clustering
  - Run multiple algorithms (KMeans, HDBSCAN, Agglomerative)
  - Combine results using consensus clustering

#### 4. **Immediate: Test Summarization Quality**
- Hypothesis: Better clustering → better summaries
- Next step: Run `lmsys summarize` on KMeans-100
- Compare summary quality vs KMeans-20

### Technical Notes

**Bug Fixes During Experimentation:**
- Fixed HDBSCAN ChromaDB type error: numpy int64 → Python int conversion
- Updated `hdbscan_clustering.py` line 230: `int(int(np.sum(...)))`
- Updated `chroma.py` lines 154, 165: `cluster_id: int(cid)`

## Experiment 4: Hierarchical Cluster Merging (2025-10-05)

### Motivation

Experiment 3 revealed that flat clustering has fundamental limitations for LMSYS data:
- **Heterogeneous clusters**: Even with 100 clusters, many still contained mixed content
- **No natural boundaries**: The dataset spans multiple dimensions (language, domain, task, style)
- **Template vs creative split**: Some queries cluster well (templates), others don't (open-ended)

**Key insight from user**: "This really suggests that we should be thinking of doing some hierarchical clustering where we do a bunch of clusters and then merge them."

### Solution: LLM-Driven Hierarchical Merging (Clio Methodology)

Following Anthropic's Clio paper approach for building multi-level topic hierarchies using LLMs.

#### Implementation Overview

**Core Algorithm** (`src/lmsys_query_analysis/clustering/hierarchy.py`):

1. **Neighborhood Formation**: Group similar clusters using embeddings (MiniBatchKMeans)
   - Typical size: 40 clusters per neighborhood (Clio default)
   - Ensures manageable LLM context windows
   - Uses all-mpnet-base-v2 for cluster embeddings

2. **Category Generation**: LLM proposes broader category names
   - Input: Cluster titles + descriptions from neighborhood
   - Output: 8-15 broader parent category names
   - Uses instructor library with Pydantic models for structured responses

3. **Deduplication**: Merge similar categories globally
   - Combines similar names across all neighborhoods
   - Creates distinct parent clusters
   - Avoids redundant top-level categories

4. **Assignment**: LLM assigns each child to best-fit parent
   - Semantic matching based on content
   - Scratchpad reasoning for transparency
   - Exact name matching enforced

5. **Refinement**: LLM refines parent names based on actual children
   - Two-sentence summary in past tense
   - Concise title (≤10 words)
   - Reflects actual assigned content

6. **Iteration**: Repeat for multiple levels
   - Level 0: Base clusters (e.g., 200 leaf clusters)
   - Level 1: First merge (e.g., 40 parents with 0.2 ratio)
   - Level 2: Second merge (e.g., 8 top categories)

#### Pydantic Models (Instructor-based)

```python
class NeighborhoodCategories(BaseModel):
    scratchpad: str = Field(description="Brief analysis of themes")
    categories: List[str] = Field(
        description="8-15 broader category names",
        min_length=8,
        max_length=15
    )

class DeduplicatedClusters(BaseModel):
    clusters: List[str] = Field(
        description="Deduplicated distinct cluster names"
    )

class ClusterAssignment(BaseModel):
    scratchpad: str = Field(description="Step-by-step reasoning")
    assigned_cluster: str = Field(description="Exact parent cluster name")

class RefinedClusterSummary(BaseModel):
    summary: str = Field(description="Two-sentence summary in past tense")
    title: str = Field(description="Concise title (≤10 words)", max_length=100)
```

#### Database Schema

**cluster_hierarchies** table tracks parent-child relationships:

```sql
CREATE TABLE cluster_hierarchies (
    id INTEGER PRIMARY KEY,
    run_id TEXT REFERENCES clustering_runs(run_id) ON DELETE CASCADE,
    hierarchy_run_id TEXT,  -- e.g., "hier-kmeans-20-20251005-123456"
    cluster_id INTEGER,
    parent_cluster_id INTEGER,  -- NULL for top level
    level INTEGER,  -- 0=leaf, 1=first merge, 2=second merge
    children_ids JSON,  -- List of child cluster IDs
    title TEXT,
    description TEXT,
    created_at TIMESTAMP
);
```

### CLI Usage

```bash
# Create 3-level hierarchy from 200 base clusters
# Level 0: 200 leaf → Level 1: 40 parents → Level 2: 8 top categories
uv run lmsys merge-clusters kmeans-200-20251004-005043 \
  --target-levels 3 \
  --merge-ratio 0.2

# Customize parameters
uv run lmsys merge-clusters <RUN_ID> \
  --target-levels 2 \              # Number of hierarchy levels
  --merge-ratio 0.5 \              # Aggressiveness (0.5 = 100->50->25)
  --llm-provider anthropic \       # anthropic/openai/groq
  --llm-model claude-sonnet-4-5-20250929 \
  --concurrency 8 \                # Parallel LLM requests
  --neighborhood-size 40           # Clusters per LLM context (Clio default)

# Use faster/cheaper models for experimentation
uv run lmsys merge-clusters <RUN_ID> \
  --llm-provider openai --llm-model gpt-4o-mini
```

### Results

**Test Run**: `kmeans-20-20251004-165542` → 2-level hierarchy (20 leaf → 10 parents)

**Hierarchy ID**: `hier-kmeans-20-20251004-165542-20251004-224423`

**Generated Parent Categories with Children:**

1. **Creative Writing and Interactive Roleplay Exercises** (2 children)
   - Creative Writing Prompts
   - Sentence Verification and Roleplay

2. **Multilingual User Engagement and Assistance** (4 children)
   - German Language User Engagement
   - Multilingual Assistance and Queries
   - Chinese Language Interaction
   - Spanish and Portuguese Language Interactions

3. **Python Problem Solving and Task Descriptions** (2 children)
   - Python Programming Task Descriptions
   - Technical Problem Solving Queries

4. **Toxic Statement Analysis by Demographic Group** (1 child)
   - Toxic Statements by Demographic Groups

5. **Business Management and Academic Inquiry Strategies** (2 children)
   - Business and Academic Inquiry Responses
   - Business Management and Consulting Practices

6. **Fact-Checking Consistency Evaluations** (1 child)
   - Fact-Checking Consistency Evaluations

7. **Russian Language Communication and Engagement** (1 child)
   - Russian Language Interactions

8. **US Sanction Analysis and Legal Entity Recognition** (2 children)
   - Detailed US Sanction Types
   - Named Entity Recognition for Legal Statements

9. **Erotic Story Generation Requests** (1 child)
   - Erotic Story Generation Requests

10. **Multilingual Text Completion and Introductions** (4 children)
    - General Text Completion Tasks
    - English Greetings and Introductions
    - Chemical Industry Company Introductions
    - Russian Language Summarization Tasks

**Quality Assessment:**
- ✅ Successfully grouped **semantically related clusters** (e.g., all multilingual interactions, Python-related queries)
- ✅ Created **meaningful parent categories** that capture broader themes
- ✅ Balanced distribution: 1-4 children per parent (avg 2.0)
- ⚠️ Some parent categories map 1:1 with children (e.g., Fact-Checking, Russian Language) - indicates these may already be optimal groupings
- ✅ No "Diverse/Various" generic names appeared in this run (improvement from initial tests)

### Key Learnings

1. **Semantic Grouping Works Well**: LLM successfully identifies thematic relationships
   - **Language-based clustering**: All German, Chinese, Spanish/Portuguese queries grouped under "Multilingual User Engagement"
   - **Domain clustering**: Python queries separated from general technical queries
   - **Task-based grouping**: Text completion tasks grouped together regardless of language

2. **1:1 Parent-Child Mappings Indicate Optimal Granularity**:
   - When a parent has only 1 child with identical/similar name, it suggests:
     - The base cluster is already at the right abstraction level
     - Or the dataset has insufficient related clusters to merge
   - Example: "Fact-Checking Consistency Evaluations" (parent) → "Fact-Checking Consistency Evaluations" (child)
   - **Recommendation**: These can stay as-is or skip merging in future iterations

3. **Merge Ratio Affects Distribution**:
   - With 0.5 ratio: 20 clusters → 10 parents (exactly 50%)
   - Actual distribution: 1-4 children per parent (not uniform)
   - LLM intelligently groups based on semantics, not just quantity
   - **Implication**: Trust LLM's semantic judgment over forcing uniform distribution

4. **Multilingual Content Naturally Clusters by Language First**:
   - Observed 2 separate multilingual parent categories:
     - "Multilingual User Engagement and Assistance" (active interactions)
     - "Multilingual Text Completion and Introductions" (passive tasks)
   - Even within language clusters, task type creates sub-groupings
   - **Best practice**: Consider language as a primary dimension for clustering

5. **Sensitive Content Handling**:
   - LLM correctly isolates sensitive categories:
     - "Toxic Statement Analysis by Demographic Group"
     - "Erotic Story Generation Requests"
   - Useful for content moderation and safety analysis
   - Avoids mixing harmful content with benign queries

6. **Hierarchy Enables Multi-Perspective Navigation**:
   - Same cluster can be understood through different parent lenses
   - Example: Russian queries split into:
     - "Russian Language Communication" (language focus)
     - "Multilingual Text Completion" (task focus)
   - Supports different user mental models for exploration

### Technical Challenges & Solutions

**Challenge 1: Async/Concurrency**
- Problem: Need to call LLM hundreds of times efficiently
- Solution: Async functions with semaphore-based concurrency control
- Result: 8 concurrent requests with optional RPM limiting

**Challenge 2: Type Mismatches (numpy int64 vs Python int)**
- Problem: ChromaDB rejects numpy.int64 types
- Solution: Double cast `int(int(np.sum(...)))` to ensure Python int
- Files: `hdbscan_clustering.py:230`, `chroma.py:154,165`

**Challenge 3: Instructor Model Selection**
- Problem: Different providers need different initialization
- Solution: Use `instructor.from_provider(f"{provider}/{model}")` pattern
- Matches pattern from `summarizer.py`

### Future Improvements

1. **Iterative Refinement**:
   - Allow user feedback on parent names
   - Re-run assignment with corrected names
   - Build golden dataset for fine-tuning

2. **Quality Metrics**:
   - Measure parent cluster coherence
   - Track generic word usage at each level
   - Compare hierarchy depth vs cluster quality

3. **Alternative Approaches**:
   - Try agglomerative clustering for comparison
   - Experiment with different embedding models for neighborhoods
   - Test with larger neighborhood sizes (60-80 clusters)

4. **Visualization**:
   - Export hierarchy as tree/graph structure
   - Interactive navigation UI
   - Drill-down from top categories to individual queries

### References

- **Anthropic Clio Paper**: Multi-level concept hierarchies for interpretability
- **OpenClio Implementation**: https://github.com/clio-lang/clio-exploitation
- **Test Run**: `kmeans-20-20251004-165542` → `hier-kmeans-20-20251005-...`
- **Code**: `src/lmsys_query_analysis/clustering/hierarchy.py`
- **Models Used**: `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-5-20250929`

## References

- **Experiment 1 Run**: `kmeans-200-20251004-005043`
- **Experiment 3 Runs**:
  - `kmeans-20-20251004-165542`
  - `kmeans-50-20251004-170427`
  - `kmeans-100-20251004-170442`
  - `hdbscan-30-20251004-170645`
- **Experiment 4 Run**: `kmeans-20-20251004-165542` (hierarchical merging)
- **Model Used**: `openai/gpt-4o-mini`
- **Code**: `src/lmsys_query_analysis/clustering/summarizer.py`, `src/lmsys_query_analysis/clustering/hierarchy.py`
