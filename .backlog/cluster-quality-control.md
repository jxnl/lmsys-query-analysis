# Cluster Quality Control: Coherence Detection and Orphan Handling

## Problem Statement

The clustering pipeline produces incoherent clusters and overly generic hierarchies because both the **summarization step** and **hierarchical merging step** lack quality control mechanisms. They are forced to create coherent narratives even when the underlying data is genuinely incoherent or poorly clustered.

### Concrete Examples

#### Example 1: Cluster 143 - "Virology Topic Proposals" (Incoherent Leaf Cluster)

**Run ID:** `kmeans-300-20251006-223440`
**Cluster ID:** 143
**Title (generated):** "Virology Topic Proposals"
**Problem:** Only 1/10 queries is about virology; the other 9 are completely unrelated.

**Actual queries in cluster:**
1. "propose 10 specific topics of virology" ✓ (THE ONE RELEVANT QUERY)
2. "traduce esto: podemos improvisar un sitio donde cenar si es necesario" (Spanish translation)
3. "list all topics we have spoken about, excluding this meta topic." (Meta conversation)
4. "give me a list of word that contain the letters f, a, r, t, in that order" (Word puzzle)
5. "Sinop'un ilceleri nelerdir ?" (Turkish geography)
6. "List all topics you cannot discuss." (Meta inquiry)
7. "Translate into english [Arabic text]" (Arabic translation)
8. "Liste subtitulos para esse tema..." (Portuguese subtitle generation)
9. "write a comprehensive market analysis...bamboo smart village" (Market analysis)
10. "list all topics we have spoken about, excluding this meta topic." (Duplicate meta)

**Root cause:** KMeans with 300 clusters on 5,000 queries created many tiny "trash bin" clusters. The LLM summarizer saw the virology query in the representative sample and created a specific title, ignoring the fact that 90% of queries are unrelated.

**What should have happened:** The cluster should be flagged as incoherent with a title like:
- "Miscellaneous: Topic Lists, Translations, Queries" (coherence_score: 1-2)

---

#### Example 2: Cluster 363 - "Creative Writing, Role-Playing, and Content Strategies" (Overly Broad Parent)

**Run ID:** `kmeans-300-20251006-223440`
**Hierarchy Run ID:** `hier-kmeans-300-20251006-223440-20251006-161308`
**Cluster ID:** 363 (Level 2 parent)
**Title (generated):** "Creative Writing, Role-Playing, and Content Strategies"
**Problem:** Merges 9 children, but only 6 are actually related. 3 are forced fits.

**Child clusters:**
1. ✓ 349: "Erotic Narrative and Role-Playing Techniques" (fits: creative writing + role-playing)
2. ✓ 350: "Creative Writing Prompts and Narrative Generation" (fits: creative writing)
3. ✓ 352: "Role-Playing Character Creation and Analysis" (fits: role-playing)
4. ✓ 358: "Humorous Content Creation: Jokes and Prompts" (fits: creative writing)
5. ~ 353: "SEO Content Strategy and Creative Development" (tangential: content strategy)
6. ~ 357: "English Writing Clarity and Simplification Techniques" (tangential: content strategy)
7. ❌ **307: "Sports Trivia and Player Inquiries"** (DOES NOT FIT)
8. ❌ **317: "Travel Itinerary Planning and Writing"** (DOES NOT FIT)
9. ❌ **355: "Text Completion Model Interaction Analysis"** (DOES NOT FIT)

**Root cause:** The hierarchical merging algorithm (line 304 in `hierarchy.py`) says "you MUST choose one", forcing every child to be assigned to a parent even when semantic overlap is poor. The LLM tries to find the "least bad" fit, resulting in overly generic parent titles.

**What should have happened:**
- Clusters 307, 317, 355 should be marked as **ORPHANS** (don't fit any existing parent)
- Parent cluster 363 should only contain the 6 relevant children
- Orphans either remain at their current level or get assigned to a generic "Miscellaneous" parent

---

## Root Cause Analysis

### 1. Clustering Algorithm Issues (upstream)

**Distribution analysis** for run `kmeans-300-20251006-223440`:
- Total queries: 5,000
- Number of clusters: 300 (16.7 queries/cluster average)
- **23 clusters** have only 1-4 queries (tiny "trash bins")
- **71 clusters** have 5-9 queries
- **5 clusters** have 50+ queries

**Insight:** 300 clusters is too granular for 5,000 queries. Many small clusters are KMeans "overflow" bins for outliers that don't fit well anywhere. These will naturally be incoherent.

**Recommendation (out of scope for this issue):**
- Suggest `--n-clusters 100-150` instead of 300 for better coherence
- Document cluster size distribution expectations in docs

### 2. Summarization Step: No Coherence Detection

**File:** `src/lmsys_query_analysis/clustering/summarizer.py`

**Current behavior (lines 74-103):**
```python
class ClusterSummaryResponse(BaseModel):
    """Structured response from LLM for cluster summarization."""

    title: str = Field(...)  # MUST create a title, no escape hatch
    description: str = Field(...)  # MUST create a description
```

**Problem:**
- No `coherence_score` field to assess cluster quality
- No option to mark clusters as "mixed" or "miscellaneous"
- LLM is forced to create a specific title even when queries are genuinely unrelated
- System prompt (lines 238-273) says "IDENTIFY THE DOMINANT PATTERN" but doesn't say what to do when there is NO dominant pattern

**Current prompt logic:**
- Line 242: "IDENTIFY THE DOMINANT PATTERN (what 60-80% of queries share)"
- Line 243: "Ignore outliers and edge cases"

**Gap:** What if 60-80% DON'T share a pattern? No guidance provided.

### 3. Hierarchical Merging: Forced Assignments

**File:** `src/lmsys_query_analysis/clustering/hierarchy.py`

**Current behavior (line 304):**
```python
user_prompt = f"""...
Steps:
1. Analyze the key characteristics of the specific cluster
2. Consider which higher-level clusters could potentially fit
3. Determine the BEST match (you MUST choose one)  # ← FORCED
4. Be sensible - don't force poor fits  # ← CONTRADICTORY!
```

**Problem:** Instructions are contradictory. Line 3 says "you MUST choose one", but line 4 says "don't force poor fits". The LLM prioritizes the imperative command and forces assignments.

**Assignment logic (lines 618-636):**
```python
for cluster, assignment in assignments:
    if assignment.assigned_cluster not in parent_children:
        # Error: LLM hallucinated a parent name
        logger.error(error_msg)
        # Fallback: assign to first parent as fallback
        parent_children[parent_names[0]].append(cluster["cluster_id"])
    else:
        # Blindly accept the assignment, no quality check
        parent_children[assignment.assigned_cluster].append(cluster["cluster_id"])
```

**Gap:** No mechanism to:
1. Allow "orphan" clusters that don't fit any parent
2. Measure assignment confidence/fit quality
3. Create a "Miscellaneous" parent for low-quality assignments

---

## Proposed Solution

### Part 1: Summarization Quality Control

**Objective:** Enable the summarizer to detect and flag incoherent clusters with appropriate title formats.

#### Changes to `src/lmsys_query_analysis/clustering/summarizer.py`

**Step 1.1: Update `ClusterSummaryResponse` schema (lines 74-103)**

Add coherence detection fields:

```python
class ClusterSummaryResponse(BaseModel):
    """Structured response from LLM for cluster summarization."""

    coherence_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="""Rate cluster coherence on a 1-5 scale based on query similarity:

        5 - Highly coherent: 80%+ of queries share a clear, specific theme
            Example: All queries about "Python Flask API development"

        4 - Mostly coherent: 60-80% share a theme, with minor outliers
            Example: Mostly Python web dev, with 2-3 unrelated queries

        3 - Mixed coherence: 50-60% share a theme, OR 2-3 distinct sub-themes of equal size
            Example: 50% translation requests + 50% word puzzles

        2 - Low coherence: Multiple unrelated themes, no dominant pattern (30-50% coherence)
            Example: Mix of translations, code, trivia, with no theme >50%

        1 - Incoherent: Completely random "trash bin" cluster (<30% coherence)
            Example: Virology + Spanish translation + Turkish geography + word puzzles

        IMPORTANT: Be honest. Low scores are valuable for identifying clustering quality issues.
        Do not try to force a coherent narrative when queries are genuinely unrelated.
        """
    )

    coherence_explanation: str = Field(
        ...,
        max_length=200,
        description="""Brief explanation (1-2 sentences) of why you assigned this coherence score.

        Examples:
        - Score 5: "All queries request Python Flask API code examples with similar structure"
        - Score 3: "Cluster contains two distinct themes: Spanish translations (50%) and cooking recipes (50%)"
        - Score 1: "Queries are completely unrelated - includes virology, translations, word puzzles, and geography with no common thread"

        Focus on percentage breakdown of themes.
        """
    )

    title: str = Field(
        ...,
        max_length=100,
        description="""Concise title (3-10 words) that accurately reflects cluster coherence.

        TITLE FORMAT RULES (based on coherence_score):

        IF coherence_score >= 4 (Coherent):
        ✓ Use specific, concrete terminology
        ✓ Examples: "Python Flask API Development", "German Language Tests", "SQL Query Generation"
        ✗ NEVER use: "User Queries", "Diverse", "Various", "General", "Mixed", "Multiple"

        IF coherence_score == 3 (Mixed):
        ✓ Use format: "Mixed: [Theme A] and [Theme B]"
        ✓ Examples: "Mixed: Translation Requests and Word Puzzles", "Mixed: Python Code and Math Problems"

        IF coherence_score <= 2 (Low coherence / Incoherent):
        ✓ Use format: "Miscellaneous: [top 2-3 patterns]"
        ✓ Examples: "Miscellaneous: Translations, Trivia, Lists", "Miscellaneous: Random Queries"

        The format MUST match the coherence_score. Do not use specific titles for incoherent clusters.
        """
    )

    description: str = Field(
        ...,
        description="""2-3 sentences explaining the cluster's purpose and patterns.

        Structure based on coherence:

        IF coherence_score >= 4:
        - Sentence 1: What users are trying to accomplish (main goal/task)
        - Sentence 2: Common characteristics (technical level, phrasing style, specific subtopics)
        - Sentence 3: (Optional) What distinguishes from neighbors

        IF coherence_score == 3:
        - Sentence 1: Describe the 2-3 main themes present
        - Sentence 2: Approximate percentages and how they differ
        - Sentence 3: Note that the cluster contains mixed content

        IF coherence_score <= 2:
        - Sentence 1: Acknowledge the cluster is incoherent/miscellaneous
        - Sentence 2: List the main patterns present (if any)
        - Sentence 3: Suggest this indicates potential clustering quality issues

        Example (score 1): "This cluster contains unrelated queries with no dominant theme.
        Queries span virology topics, Spanish translations, Turkish geography, word puzzles,
        and meta-conversation requests. This suggests a potential 'trash bin' cluster from
        the KMeans algorithm."

        Focus on the DOMINANT pattern(s), not every variation. Be honest about quality.
        """
    )
```

**Step 1.2: Update system prompt (lines 238-273)**

Add quality control instructions:

```python
messages = [
    {
        "role": "system",
        "content": """You are an expert taxonomist specializing in categorizing user interactions with LLM systems. Your goal is to create PRECISE, SPECIFIC cluster labels that enable quick understanding.

**CRITICAL ROLE: QUALITY CONTROL**

Your PRIMARY responsibility is to HONESTLY ASSESS cluster coherence. If a cluster is incoherent,
you MUST flag it with a low coherence_score and use appropriate title formatting.

Do NOT try to force a coherent narrative when queries are genuinely unrelated. Low coherence
scores are VALUABLE signals for identifying clustering quality issues - they are not failures.

CRITICAL INSTRUCTIONS:

0. ASSESS COHERENCE FIRST (NEW STEP)
   - Calculate what percentage of queries share common themes
   - If 80%+ share a specific theme → score 5 (highly coherent)
   - If 60-80% share a theme → score 4 (mostly coherent)
   - If 50-60% OR 2-3 equal themes → score 3 (mixed)
   - If 30-50% coherence → score 2 (low coherence)
   - If <30% coherence → score 1 (incoherent)
   - Use coherence_explanation to document your reasoning

1. IDENTIFY THE DOMINANT PATTERN (what 60-80% of queries share)
   - If coherence_score >= 4: Proceed with standard analysis
   - If coherence_score == 3: Identify the 2-3 main sub-themes
   - If coherence_score <= 2: Note the cluster is incoherent and list patterns
   - Focus on the PRIMARY use case or theme (if one exists)

2. TITLE REQUIREMENTS (based on coherence_score):

   FOR COHERENT CLUSTERS (score 4-5):
   - 3-7 words maximum
   - Use SPECIFIC technical terms, domains, or action verbs
   - NEVER use: "User Queries", "Diverse", "Various", "General", "Mixed", "Multiple", "Different"
   - ALWAYS be concrete: use programming languages, specific topics, named tools/frameworks
   - Examples of GOOD titles:
     * "Python Web Scraping Code"
     * "German Language Capability Tests"
     * "SQL Database Query Generation"
     * "Stable Diffusion Art Prompts"
   - Examples of BAD titles (NEVER DO THIS):
     * "User Queries About Programming" ❌
     * "Diverse Technical Requests" ❌
     * "Various Creative Writing Tasks" ❌

   FOR MIXED CLUSTERS (score 3):
   - Use format: "Mixed: [Theme A] and [Theme B]"
   - Examples:
     * "Mixed: Translation Requests and Word Puzzles"
     * "Mixed: Python Code and SQL Queries"

   FOR INCOHERENT CLUSTERS (score 1-2):
   - Use format: "Miscellaneous: [top 2-3 patterns]"
   - Examples:
     * "Miscellaneous: Translations, Trivia, Lists"
     * "Miscellaneous: Unrelated Queries"
     * "Miscellaneous: Random Topic Requests"

3. DESCRIPTION REQUIREMENTS:
   - Sentence 1: State the PRIMARY goal/task OR acknowledge incoherence
   - Sentence 2: Key patterns (technical level, common subtopics, phrasing style) OR list themes
   - Sentence 3: (Optional) What distinguishes from neighbors OR note clustering quality issues
   - Be SPECIFIC: mention actual examples, technical terms, languages, frameworks
   - Focus on DOMINANT pattern(s), not every variation
   - For incoherent clusters, be explicit: "This cluster contains unrelated queries..."

4. MULTILINGUAL CLUSTERS:
   - Always specify the language(s) in the title
   - Example: "Portuguese Business Writing", "Arabic General Knowledge"

Follow the Pydantic schema rules exactly. Remember: low coherence scores are VALUABLE, not failures.""",
    },
    {"role": "user", "content": prompt},
]
```

**Step 1.3: Update logging (around line 245)**

Add coherence score to logs:

```python
results[cid] = summary
self.logger.info(
    "Completed summary for cluster %d (coherence: %d/5): %s",
    cid,
    summary.get("coherence_score", 0),
    summary["title"]
)
```

**Step 1.4: Update return value (line 284-288)**

Include coherence fields:

```python
return {
    "title": response.title,
    "description": response.description,
    "sample_queries": sampled,
    "coherence_score": response.coherence_score,
    "coherence_explanation": response.coherence_explanation,
}
```

---

### Part 2: Hierarchical Merging Quality Control

**Objective:** Allow orphan clusters that don't fit any parent, and measure assignment confidence.

#### Changes to `src/lmsys_query_analysis/clustering/hierarchy.py`

**Step 2.1: Update `ClusterAssignment` schema (lines 79-95)**

Add confidence scoring:

```python
class ClusterAssignment(BaseModel):
    """Response for assigning a cluster to its best-fit parent category."""

    scratchpad: str = Field(
        description="""Step-by-step reasoning (2-4 sentences):
        1. Key characteristics of the cluster being assigned
        2. Which parent categories could potentially fit
        3. Estimate semantic overlap percentage for best candidate
        4. Why the chosen parent is the best match (or why ORPHAN if no good fit)
        """
    )

    fit_confidence: int = Field(
        ...,
        ge=1,
        le=5,
        description="""Rate how well this child semantically fits the chosen parent (1-5):

        5 - Perfect fit: 80%+ semantic overlap
            Example: "Python Flask Development" → "Python Web Development Frameworks"

        4 - Strong fit: 60-80% semantic overlap, minor gaps
            Example: "Django REST APIs" → "Python Web Development Frameworks"

        3 - Moderate fit: 40-60% semantic overlap
            Example: "Web Scraping Scripts" → "Python Web Development Frameworks"

        2 - Weak fit: 20-40% semantic overlap (SHOULD BE ORPHAN unless no better option)
            Example: "Spanish Travel Writing" → "Creative Writing and Content"

        1 - Poor fit: <20% semantic overlap (MUST BE ORPHAN)
            Example: "Sports Trivia" → "Creative Writing and Content"

        CRITICAL: If fit_confidence <= 2, you should strongly prefer assigning "ORPHAN"
        instead of forcing a poor fit.
        """
    )

    assigned_cluster: str = Field(
        description="""Exact name of the chosen parent cluster OR "ORPHAN".

        CRITICAL: Must match exactly one of the provided parent cluster names, OR use "ORPHAN".

        Use "ORPHAN" when:
        - fit_confidence would be <= 2 for all available parents
        - Semantic overlap is <40% with all candidates
        - Forcing an assignment would dilute the parent's specificity

        Copy the full parent name without modification, or use the exact string "ORPHAN".
        """
    )
```

**Step 2.2: Update assignment prompt (lines 289-307)**

Allow orphan option:

```python
async def assign_to_parent_cluster(
    client: instructor.AsyncInstructor,
    child_cluster: Dict[str, str],
    parent_candidates: List[str]
) -> ClusterAssignment:
    """Assign a child cluster to the best-fit parent category OR mark as orphan.

    Args:
        client: Async instructor client
        child_cluster: Dict with 'title' and 'description' of cluster to assign
        parent_candidates: List of parent cluster names

    Returns:
        ClusterAssignment with reasoning, confidence, and chosen parent (or "ORPHAN")
    """
    parents_str = "\n".join([f"<cluster>{name}</cluster>" for name in parent_candidates])

    system_prompt = """You are categorizing clusters for observability, monitoring, and content moderation.

CRITICAL: Your job is to maintain taxonomy QUALITY. Only assign a child to a parent if there is
meaningful semantic overlap (40%+). If no parent is a good fit, use "ORPHAN" - this is PREFERRED
over forcing a poor fit that dilutes parent specificity."""

    user_prompt = f"""Categorize this specific cluster into one of the provided higher-level clusters,
OR mark it as ORPHAN if no parent is a good semantic fit.

<specific_cluster>
Title: {child_cluster['title']}
Description: {child_cluster['description']}
</specific_cluster>

<higher_level_clusters>
{parents_str}
<cluster>ORPHAN</cluster>
</higher_level_clusters>

Steps:
1. Analyze the key characteristics and themes of the specific cluster
2. For EACH parent candidate, estimate semantic overlap percentage:
   - What percentage of the child's content aligns with the parent's theme?
   - Example: "Python Flask" has ~90% overlap with "Python Web Development"
   - Example: "Sports Trivia" has ~5% overlap with "Creative Writing"
3. Calculate fit_confidence (1-5) based on the BEST candidate's overlap:
   - 80%+ overlap → confidence 5
   - 60-80% → confidence 4
   - 40-60% → confidence 3
   - 20-40% → confidence 2 (prefer ORPHAN)
   - <20% → confidence 1 (MUST use ORPHAN)
4. Choose the best match:
   - If fit_confidence >= 3: Assign to that parent
   - If fit_confidence <= 2: Use "ORPHAN"

REMEMBER: "ORPHAN" is a VALID and PREFERRED choice for poor fits (<40% overlap).
Do not force assignments to maintain artificial completeness.

Use <scratchpad> for reasoning (2-4 sentences), rate fit_confidence, then provide the
exact parent cluster name or "ORPHAN"."""

    logger.info(f"Assigning to parent cluster: {child_cluster['title']}")
    response = await client.chat.completions.create(
        response_model=ClusterAssignment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response
```

**Step 2.3: Update assignment handling (lines 593-637)**

Track orphans and warn on low confidence:

```python
# Step 5: Assign children to parents (parallelized)
step_start = time.time()
logger.info(f"Assigning {n_current} children to {len(parent_names)} parents (concurrency={concurrency})...")
parent_children = {name: [] for name in parent_names}
orphaned_clusters = []  # NEW: Track orphans
assignment_errors = []
low_confidence_assignments = []  # NEW: Track questionable assignments
semaphore = anyio.Semaphore(concurrency)
assignments = []

# Worker function for parallel assignments
async def assign_worker(cluster):
    async with semaphore:
        if limiter:
            async with limiter:
                assignment = await assign_to_parent_cluster(client, cluster, parent_names)
        else:
            assignment = await assign_to_parent_cluster(client, cluster, parent_names)
        assignments.append((cluster, assignment))
        progress.update(task, advance=1)

# Run all assignments in parallel
async with anyio.create_task_group() as tg:
    for cluster in current_clusters:
        tg.start_soon(assign_worker, cluster)

# Validate and organize assignments
for cluster, assignment in assignments:
    cid = cluster["cluster_id"]
    title = cluster["title"]
    assigned_parent = assignment.assigned_cluster
    confidence = assignment.fit_confidence

    # Handle orphans (NEW)
    if assigned_parent == "ORPHAN":
        logger.info(
            f"Cluster {cid} ('{title}') marked as ORPHAN (no suitable parent, confidence={confidence})"
        )
        orphaned_clusters.append({
            "cluster_id": cid,
            "title": title,
            "description": cluster["description"],
            "fit_confidence": confidence,
            "reasoning": assignment.scratchpad
        })
        continue

    # Validate parent exists
    if assigned_parent not in parent_children:
        error_msg = (
            f"Invalid assignment for cluster {cid} ('{title}'): "
            f"LLM returned '{assigned_parent}' which is not in parent list. "
            f"Valid parents: {parent_names}"
        )
        logger.error(error_msg)
        assignment_errors.append({
            "cluster_id": cid,
            "cluster_title": title,
            "invalid_parent": assigned_parent,
            "valid_parents": parent_names
        })
        # Assign to first parent as fallback
        parent_children[parent_names[0]].append(cid)
        logger.warning(f"Falling back to parent '{parent_names[0]}' for cluster {cid}")
        continue

    # Warn on low-confidence assignments (NEW)
    if confidence <= 2:
        logger.warning(
            f"Low confidence assignment ({confidence}/5): '{title}' → '{assigned_parent}'"
        )
        logger.debug(f"  Reasoning: {assignment.scratchpad}")
        low_confidence_assignments.append({
            "cluster_id": cid,
            "cluster_title": title,
            "parent": assigned_parent,
            "confidence": confidence,
            "reasoning": assignment.scratchpad
        })

    # Accept assignment
    parent_children[assigned_parent].append(cid)
    logger.debug(
        f"Assigned cluster {cid} ('{title}') → '{assigned_parent}' (confidence={confidence}/5)"
    )

# Report validation and quality metrics (NEW)
logger.info(f"Assignment complete: {len(assignments)} total")
logger.info(f"  ✓ Successfully assigned: {sum(len(children) for children in parent_children.values())}")
if orphaned_clusters:
    logger.info(f"  ⚠ Orphaned clusters: {len(orphaned_clusters)}")
    for orphan in orphaned_clusters[:5]:  # Show first 5
        logger.debug(f"    - {orphan['title']} (confidence={orphan['fit_confidence']})")
    if len(orphaned_clusters) > 5:
        logger.debug(f"    ... and {len(orphaned_clusters) - 5} more")
if low_confidence_assignments:
    logger.warning(f"  ⚠ Low confidence assignments: {len(low_confidence_assignments)}")
    for lc in low_confidence_assignments[:3]:  # Show first 3
        logger.debug(f"    - {lc['cluster_title']} → {lc['parent']} (confidence={lc['confidence']})")
if assignment_errors:
    logger.warning(f"  ✗ Assignment errors (hallucinated parents): {len(assignment_errors)}")
    logger.warning("    First 5 errors: " + str(assignment_errors[:5]))

logger.debug(f"  Completed assignments in {time.time() - step_start:.1f}s")
```

**Step 2.4: Handle orphans in next iteration (after line 647)**

Option A: Keep orphans at current level (don't promote them)
Option B: Create a "Miscellaneous" parent for all orphans

```python
# Step 6: Refine parent names based on children (parallelized)
# ... existing code ...

# NEW: Handle orphans
if orphaned_clusters:
    logger.info(f"Handling {len(orphaned_clusters)} orphaned clusters...")

    # Option A: Keep at current level (don't add to next_level_clusters)
    # They will not be promoted to the next hierarchy level
    logger.info("Orphans will remain at current level and not be promoted to next level")

    # Option B (OPTIONAL): Create a "Miscellaneous" parent for orphans
    # Uncomment below to enable:
    """
    misc_parent_id = next_cluster_id
    next_cluster_id += 1

    orphan_ids = [o["cluster_id"] for o in orphaned_clusters]
    orphan_titles = [o["title"] for o in orphaned_clusters]

    # Add miscellaneous parent to hierarchy
    hierarchy.append({
        "hierarchy_run_id": hierarchy_run_id,
        "run_id": run_id,
        "cluster_id": misc_parent_id,
        "parent_cluster_id": None,
        "level": current_level + 1,
        "children_ids": orphan_ids,
        "title": "Miscellaneous: Uncategorized Clusters",
        "description": f"Contains {len(orphan_ids)} clusters that did not fit well into "
                      f"any specific parent category. Includes: {', '.join(orphan_titles[:3])}..."
    })

    # Update orphan children to point to misc parent
    for orphan_id in orphan_ids:
        for h in hierarchy:
            if h["cluster_id"] == orphan_id and h["level"] == current_level:
                h["parent_cluster_id"] = misc_parent_id
                break

    # Add to next level
    next_level_clusters.append({
        "cluster_id": misc_parent_id,
        "title": "Miscellaneous: Uncategorized Clusters",
        "description": f"Contains {len(orphan_ids)} orphaned clusters"
    })

    logger.info(f"Created miscellaneous parent {misc_parent_id} for {len(orphan_ids)} orphans")
    """
```

---

### Part 3: Database Schema Updates

**Objective:** Store coherence and confidence metrics for analysis.

#### Option A: Add columns to existing tables (RECOMMENDED)

**File:** `src/lmsys_query_analysis/db/models.py`

```python
class ClusterSummary(SQLModel, table=True):
    # ... existing fields ...

    # NEW FIELDS (add after line ~100):
    coherence_score: int | None = Field(
        default=None,
        description="Cluster coherence rating (1-5): 5=highly coherent, 1=incoherent"
    )
    coherence_explanation: str | None = Field(
        default=None,
        description="Brief explanation of coherence score"
    )
```

**Migration:** Since SQLModel/SQLite doesn't have built-in migrations, add a migration function:

```python
# In src/lmsys_query_analysis/db/connection.py

def migrate_add_coherence_columns(db: Database):
    """Add coherence_score and coherence_explanation columns to cluster_summaries table."""
    with db.get_session() as session:
        # Check if columns already exist
        result = session.exec(text("PRAGMA table_info(cluster_summaries)")).fetchall()
        columns = [row[1] for row in result]

        if "coherence_score" not in columns:
            session.exec(text("ALTER TABLE cluster_summaries ADD COLUMN coherence_score INTEGER"))
            logger.info("Added coherence_score column to cluster_summaries")

        if "coherence_explanation" not in columns:
            session.exec(text("ALTER TABLE cluster_summaries ADD COLUMN coherence_explanation TEXT"))
            logger.info("Added coherence_explanation column to cluster_summaries")

        session.commit()
```

**Call migration in CLI** (`src/lmsys_query_analysis/cli/main.py`):

```python
# In the summarize command (around line 300):
def summarize_clusters(...):
    db = Database(db_path)

    # Run migration (idempotent)
    from src.lmsys_query_analysis.db.connection import migrate_add_coherence_columns
    migrate_add_coherence_columns(db)

    # ... rest of function
```

#### Option B: Store in JSON parameters field (NO SCHEMA CHANGE)

Use existing `parameters` JSON field in `cluster_summaries` table:

```python
# In summarize CLI command:
params = {
    "model": model,
    "max_queries": max_queries,
    "contrast_neighbors": contrast_neighbors,
    # NEW:
    "coherence_score": summary["coherence_score"],
    "coherence_explanation": summary["coherence_explanation"],
}

summary_record = ClusterSummary(
    # ... existing fields ...
    parameters=params
)
```

**RECOMMENDATION:** Use Option A for queryability. Coherence score will be useful for filtering/sorting.

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_summarizer.py` (new file)

```python
import pytest
from src.lmsys_query_analysis.clustering.summarizer import ClusterSummaryResponse

def test_coherence_validation():
    """Test that coherence_score must be 1-5."""
    # Valid
    resp = ClusterSummaryResponse(
        coherence_score=5,
        coherence_explanation="All queries about Python Flask",
        title="Python Flask Development",
        description="Users request Flask code examples."
    )
    assert resp.coherence_score == 5

    # Invalid (out of range)
    with pytest.raises(ValueError):
        ClusterSummaryResponse(
            coherence_score=6,  # Invalid
            coherence_explanation="Test",
            title="Test",
            description="Test"
        )

def test_title_format_enforcement():
    """Test that title format matches coherence score."""
    # This is more of a prompt engineering test - would need LLM call
    # Can do manual validation tests instead
    pass

@pytest.mark.smoke
def test_incoherent_cluster_summarization():
    """Test that summarizer correctly identifies incoherent clusters."""
    from src.lmsys_query_analysis.clustering.summarizer import ClusterSummarizer

    # Simulate Cluster 143 (virology + translations + etc)
    queries = [
        "propose 10 specific topics of virology",
        "traduce esto: podemos improvisar un sitio donde cenar",
        "list all topics we have spoken about",
        "give me a list of word that contain the letters f, a, r, t",
        "Sinop'un ilceleri nelerdir ?",
        "List all topics you cannot discuss.",
        "Translate into english: [Arabic text]",
        "Liste subtitulos para esse tema...",
        "write a market analysis for bamboo smart village",
    ]

    summarizer = ClusterSummarizer(model="openai/gpt-4o-mini")
    summary = summarizer.generate_cluster_summary(queries, cluster_id=143)

    # Should detect low coherence
    assert summary["coherence_score"] <= 2, \
        f"Expected coherence_score <= 2 for incoherent cluster, got {summary['coherence_score']}"

    # Should use "Miscellaneous:" prefix
    assert summary["title"].startswith("Miscellaneous:"), \
        f"Expected 'Miscellaneous:' prefix for score {summary['coherence_score']}, got: {summary['title']}"

    print(f"✓ Incoherent cluster detected: {summary['title']} (score: {summary['coherence_score']})")
```

**File:** `tests/test_hierarchy.py` (update existing)

```python
@pytest.mark.smoke
def test_orphan_assignment():
    """Test that poor-fit clusters are marked as ORPHAN."""
    from src.lmsys_query_analysis.clustering.hierarchy import assign_to_parent_cluster
    import instructor

    client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

    # Sports cluster trying to fit into creative writing parents
    child_cluster = {
        "title": "Sports Trivia and Player Inquiries",
        "description": "Users request sports facts, player statistics, and game information."
    }

    parent_candidates = [
        "Creative Writing Prompts and Narrative Generation",
        "Role-Playing Character Creation",
        "SEO Content Strategy"
    ]

    import anyio
    assignment = anyio.run(assign_to_parent_cluster, client, child_cluster, parent_candidates)

    # Should be marked as ORPHAN with low confidence
    assert assignment.assigned_cluster == "ORPHAN", \
        f"Expected ORPHAN for sports/creative mismatch, got: {assignment.assigned_cluster}"
    assert assignment.fit_confidence <= 2, \
        f"Expected low confidence, got {assignment.fit_confidence}/5"

    print(f"✓ Poor fit correctly identified as ORPHAN (confidence: {assignment.fit_confidence})")
```

### Integration Tests

**Smoke test with real data** (update `smoketest.sh`):

```bash
#!/bin/bash
set -e

# Run full pipeline with small dataset
LIMIT=${SMOKE_LIMIT:-500}

echo "Running smoke test with $LIMIT queries..."

# 1. Load data
uv run lmsys load --limit $LIMIT --use-chroma

# 2. Run clustering (fewer clusters for better coherence)
RUN_ID=$(uv run lmsys cluster kmeans --n-clusters 50 --use-chroma | grep "run_id" | cut -d: -f2)

# 3. Summarize
uv run lmsys summarize "$RUN_ID" --alias "smoke-test"

# 4. Check for coherence scores in output
echo "Checking coherence scores..."
uv run python -c "
from src.lmsys_query_analysis.db.connection import Database
from src.lmsys_query_analysis.db.models import ClusterSummary
from sqlmodel import Session, select

db = Database()
with db.get_session() as session:
    stmt = select(ClusterSummary).where(ClusterSummary.run_id == '$RUN_ID')
    summaries = session.exec(stmt).all()

    low_coherence = [s for s in summaries if s.coherence_score and s.coherence_score <= 2]

    print(f'Total clusters: {len(summaries)}')
    print(f'Low coherence clusters (score <= 2): {len(low_coherence)}')

    if low_coherence:
        print('\nLow coherence examples:')
        for s in low_coherence[:3]:
            print(f'  - Cluster {s.cluster_id}: {s.title} (score: {s.coherence_score})')
            print(f'    Reason: {s.coherence_explanation}')
"

# 5. Run hierarchical merge
uv run lmsys merge-clusters "$RUN_ID"

echo "✓ Smoke test passed"
```

### Manual Validation

**After implementing changes, re-run the problematic clusters:**

```bash
# Get cluster 143 summary with new prompt
uv run lmsys summarize kmeans-300-20251006-223440 --alias "v2-coherence-check" --cluster-id 143

# Check results
sqlite3 ~/.lmsys-query-analysis/queries.db \
  "SELECT coherence_score, title, coherence_explanation
   FROM cluster_summaries
   WHERE run_id = 'kmeans-300-20251006-223440'
   AND cluster_id = 143
   AND alias = 'v2-coherence-check'"
```

**Expected output:**
```
1|Miscellaneous: Translations, Topic Lists, Trivia|Cluster contains completely unrelated queries including virology, Spanish/Turkish/Arabic translations, word puzzles, and meta-conversation requests with no common theme (<30% coherence).
```

---

## Implementation Checklist

### Phase 1: Summarization Quality Control
- [ ] Update `ClusterSummaryResponse` schema with coherence fields
- [ ] Update system prompt with quality control instructions
- [ ] Update logging to include coherence scores
- [ ] Update return value to include coherence data
- [ ] Add database migration for coherence columns (or use JSON params)
- [ ] Update CLI to save coherence data to database
- [ ] Write unit tests for schema validation
- [ ] Write smoke test for incoherent cluster detection

### Phase 2: Hierarchy Orphan Handling
- [ ] Update `ClusterAssignment` schema with fit_confidence
- [ ] Update assignment prompt to allow ORPHAN option
- [ ] Update assignment handling to track orphans
- [ ] Add orphan reporting/logging
- [ ] Decide orphan strategy (keep at level vs misc parent)
- [ ] Write unit tests for ORPHAN assignments
- [ ] Write smoke test for poor-fit detection

### Phase 3: Testing & Validation
- [ ] Run smoke test suite with new changes
- [ ] Re-analyze problematic clusters (143, 363) with new prompts
- [ ] Verify coherence scores align with manual inspection
- [ ] Verify orphans are correctly identified
- [ ] Check hierarchy structure with orphans
- [ ] Document coherence score distribution expectations

### Phase 4: Documentation
- [ ] Update README with coherence score explanation
- [ ] Document ORPHAN mechanism in hierarchy docs
- [ ] Add troubleshooting guide for low-coherence clusters
- [ ] Update CLAUDE.md with quality control workflow

---

## Success Criteria

### Quantitative Metrics

1. **Coherence score distribution** (for a 5000-query, 100-cluster run):
   - Expected: 60-70% score 4-5 (coherent)
   - Expected: 20-30% score 3 (mixed)
   - Expected: 10-20% score 1-2 (incoherent)

   If >50% are incoherent, the clustering algorithm needs tuning.

2. **Orphan rate** (for hierarchical merging):
   - Expected: 5-15% orphan rate at each level
   - If >30% orphans: parents may be too specific, increase merge_ratio
   - If <5% orphans: likely forcing bad fits, decrease merge_ratio

3. **Low-confidence assignment rate**:
   - Expected: <20% assignments with confidence <= 2
   - If >20%: parents may not cover child diversity well

### Qualitative Validation

1. **Cluster 143 re-test:**
   - ✓ coherence_score should be 1-2
   - ✓ title should use "Miscellaneous:" prefix
   - ✓ description should acknowledge incoherence

2. **Cluster 363 re-test:**
   - ✓ Sports Trivia (307) → ORPHAN
   - ✓ Travel Itinerary (317) → ORPHAN
   - ✓ Text Completion (355) → ORPHAN
   - ✓ Remaining 6 children should have confidence >= 3

3. **User experience:**
   - ✓ Web viewer shows coherence badges/colors
   - ✓ Low-coherence clusters are visually distinct
   - ✓ Users can filter by coherence score
   - ✓ Hierarchy shows orphan counts per level

---

## Out of Scope (Future Work)

1. **Clustering algorithm improvements**
   - Adaptive cluster count based on data
   - HDBSCAN parameter tuning for better coherence
   - Cluster quality pre-filtering before summarization

2. **UI enhancements**
   - Coherence score visualization in web viewer
   - Filter/sort by coherence
   - Orphan cluster view

3. **Post-processing**
   - Automatic re-clustering of incoherent clusters
   - Cluster splitting based on coherence analysis
   - Hierarchical structure validation/correction

---

## References

### Related Files
- `src/lmsys_query_analysis/clustering/summarizer.py` (summarization)
- `src/lmsys_query_analysis/clustering/hierarchy.py` (hierarchical merging)
- `src/lmsys_query_analysis/db/models.py` (database schema)
- `src/lmsys_query_analysis/cli/main.py` (CLI commands)

### Related Issues
- Cluster size distribution analysis needed (300 clusters → 5000 queries)
- ChromaDB metadata consistency for coherence scores
- Web viewer UX for quality signals

### Research
- Anthropic Clio methodology: https://www.anthropic.com/research/clio
- Clustering quality metrics: silhouette score, Davies-Bouldin index
- Hierarchical clustering validation methods
