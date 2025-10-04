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

## References

- **Experiment 1 Run**: `kmeans-200-20251004-005043`
- **Model Used**: `openai/gpt-4o-mini`
- **Code**: `src/lmsys_query_analysis/clustering/summarizer.py`
