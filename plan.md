# LLM-Based Row Summarization Implementation Plan

## Motivation & Vision

### The Problem: Heterogeneous Raw Queries

Current clustering on raw LMSYS queries produces **low-quality, catch-all clusters** because:

1. **Lexical diversity**: Identical intents expressed in different words ("help me code Python" vs "write Python script for me")
2. **Noise and verbosity**: Raw queries contain typos, greetings, unnecessary context
3. **Multi-lingual mixing**: Same intent in different languages clusters separately
4. **Varying specificity**: "help with coding" vs "debug my React useEffect hook that re-renders infinitely"

**Example:** A cluster labeled "Python Programming" might contain:
- ✅ "Write a Python function to reverse a string"
- ✅ "Como escribir un bucle for en Python" (Spanish)
- ✅ "pls help me code python thx"
- ❌ "I'm learning to code and need general advice on where to start" (not Python-specific)
- ❌ "Can you explain what programming is?" (not Python)

Clustering quality metrics:
- **Current KMeans coherence**: ~30-40% on diverse datasets
- **Expected HDBSCAN coherence**: 80-90% on semantically normalized data

### The Solution: LLM-Powered Query Normalization

Create **derived datasets** where each query is transformed via LLM into a **canonical, semantically-focused representation**:

```
Raw Query → LLM Summarization → Normalized Query
```

**Example transformations:**

| Raw Query | Summarized Query (Intent Extraction) |
|-----------|-------------------------------------|
| "hey can u help me write a python script that reverses a string? thx" | "Write Python function to reverse string" |
| "Como escribir un bucle for en Python" | "Write Python for loop" |
| "I'm trying to learn coding, should I start with Python or JavaScript? What's easier?" | "Compare Python vs JavaScript for beginners" |

**Benefits:**
1. **Better clustering**: Semantically similar queries cluster together regardless of phrasing
2. **Flexible analysis views**: Create different datasets for different hypotheses (intent, domain, complexity)
3. **Agent autonomy**: Claude can generate custom prompts to test data analysis hypotheses
4. **Reusable infrastructure**: Same mechanism for classification, extraction, enrichment

### Autonomous Agent Use Case

The key insight: **Summarization = Programmable Data Transformation**

Claude can autonomously:
1. **Generate hypothesis** ("I think 20% of queries are jailbreak attempts")
2. **Create custom prompt** ("Classify this query: {safe, jailbreak_roleplay, jailbreak_injection, jailbreak_obfuscation}")
3. **Run summarization** to create classification dataset
4. **Cluster on classifications** to validate hypothesis
5. **Iterate** with refined prompts based on findings

This enables **self-directed data exploration** without hardcoded analysis workflows.

---

## Architecture Overview

### Summarization as Dataset Derivation

**Core principle:** Summarized queries are NOT a separate entity—they're just **new Datasets containing transformed Query records**.

```
┌─────────────────┐
│ Dataset:        │
│ "lmsys-1m"      │──┐
└─────────────────┘  │
                     │ Summarization
                     │ (prompt_hash: abc123)
                     ▼
              ┌─────────────────┐
              │ Dataset:        │
              │ "lmsys-1m-      │
              │  intent-v1"     │
              └─────────────────┘
                     │
                     │ Further Summarization
                     │ (prompt_hash: def456)
                     ▼
              ┌─────────────────┐
              │ Dataset:        │
              │ "lmsys-1m-      │
              │  intent-v1-     │
              │  classified"    │
              └─────────────────┘
```

**Key insight:** You can summarize a summary (chain transformations).

### Data Flow: XML Input → Structured JSON Output

The summarization pipeline uses **instructor for type-safe structured outputs**:

```
┌─────────────────────────────────────────────────────────────┐
│ Input: Query Record (Python object)                         │
│   query_text: "hey can you help me write python script?"    │
│   model: "gpt-3.5-turbo"                                     │
│   language: "en"                                             │
│   timestamp: 2024-01-15T10:30:00Z                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ serialize_query_to_xml()
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ XML Serialization (sent to LLM)                             │
│                                                              │
│ <query>                                                      │
│   <model>gpt-3.5-turbo</model>                              │
│   <query_text>hey can you help me write python script?</... │
│   <language>en</language>                                   │
│   <timestamp>2024-01-15T10:30:00Z</timestamp>              │
│ </query>                                                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Formatted into user prompt
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM Prompt                                                   │
│                                                              │
│ "Extract user intent in one sentence:                       │
│                                                              │
│  <query>                                                     │
│    <model>gpt-3.5-turbo</model>                             │
│    <query_text>hey can you help me write python script...</  │
│  </query>"                                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ LLM processing
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM Response (JSON via instructor)                          │
│                                                              │
│ {                                                            │
│   "summary": "Write Python script",                         │
│   "properties": [                                           │
│     {"name": "intent", "value": "code_generation"},         │
│     {"name": "complexity", "value": 2},                     │
│     {"name": "programming_language", "value": "python"},    │
│     {"name": "requires_code_output", "value": true}         │
│   ]                                                          │
│ }                                                            │
│                                                              │
│ (Automatically parsed into QuerySummaryResponse Pydantic)   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Extract response.summary + properties
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Output: New Query Record                                    │
│   query_text: "Write Python script"  ← LLM summary          │
│   model: "gpt-3.5-turbo"             ← Copied from original │
│   language: "en"                     ← Copied from original │
│   conversation_id: "abc-123"         ← Same as original     │
│   dataset_id: 2                      ← New dataset          │
│   extra_metadata: {                                         │
│     "source_query_id": 12345,                               │
│     "summarization_model": "gpt-4o-mini",                   │
│     "intent": "code_generation",            ← From properties│
│     "complexity": 2,                        ← From properties│
│     "programming_language": "python",       ← From properties│
│     "requires_code_output": true            ← From properties│
│   }                                                          │
└─────────────────────────────────────────────────────────────┘
```

**Key benefits of this approach:**

1. **Type safety**: Instructor ensures LLM returns valid JSON matching `QuerySummaryResponse`
2. **Retry-friendly**: Structured outputs more reliable than free-form text
3. **Metadata preservation**: Full Query context (model, language, timestamp) available to LLM
4. **Multi-dimensional extraction**: Properties enable extracting multiple features in ONE LLM call
5. **SQL-queryable properties**: Stored in `extra_metadata` JSON field, queryable via SQLite JSON operators
6. **Flexible schema**: No need to pre-define columns—add any property dynamically

**Example SQL queries enabled by properties:**

```sql
-- Find all code generation queries
SELECT * FROM queries 
WHERE extra_metadata->>'intent' = 'code_generation';

-- Find complex Python queries
SELECT * FROM queries 
WHERE extra_metadata->>'programming_language' = 'python' 
  AND CAST(extra_metadata->>'complexity' AS INTEGER) >= 3;

-- Analyze intent distribution
SELECT 
    extra_metadata->>'intent' as intent,
    COUNT(*) as count
FROM queries 
WHERE dataset_id = 2
GROUP BY intent
ORDER BY count DESC;
```

---

## Schema Changes

### 1. New `Prompt` Table

Stores reusable prompt templates with content-based hashing for deduplication.

```python
class Prompt(SQLModel, table=True):
    """Stores LLM prompt templates for query summarization.
    
    Uses content-based hashing (SHA256) to deduplicate identical prompts
    and track prompt lineage for LLM-generated iterations.
    """
    __tablename__ = "prompts"
    
    prompt_hash: str = Field(primary_key=True)  # SHA256(prompt_text)
    prompt_text: str  # Template with {query} placeholder
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional: Track LLM-generated prompt evolution
    generated_from: str | None = Field(foreign_key="prompts.prompt_hash")
    generation_context: str | None = None  # Why/how it was generated
    
    # Optional: Usage tracking
    usage_count: int = Field(default=0)
```

**Hash function:**
```python
import hashlib

def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA256 hash of prompt text (no normalization)."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
```

**Design decisions:**
- ✅ Hash as-is (no whitespace normalization) - precision over deduplication
- ✅ SHA256 for collision resistance
- ✅ `generated_from` enables tracking LLM-generated prompt iterations

### 2. Update `Dataset` Table

Add foreign keys to track dataset lineage and summarization metadata.

```python
class Dataset(SQLModel, table=True):
    # ... existing fields (id, name, source, description, created_at, column_mapping, query_count)
    
    # NEW FIELDS for summarization lineage
    source_dataset_id: int | None = Field(
        default=None,
        foreign_key="datasets.id",
        description="Immediate parent dataset (one level up)"
    )
    root_dataset_id: int | None = Field(
        default=None,
        foreign_key="datasets.id",
        description="Original root dataset (top of lineage tree)"
    )
    prompt_hash: str | None = Field(
        default=None,
        foreign_key="prompts.prompt_hash",
        description="Prompt used to generate this dataset (if summarized)"
    )
    summarization_model: str | None = Field(
        default=None,
        description="LLM model used (e.g., 'gpt-4o-mini', 'claude-sonnet-4')"
    )
    
    # Relationships
    queries: list["Query"] = Relationship(back_populates="dataset")
    source_dataset: "Dataset | None" = Relationship(
        back_populates="derived_datasets",
        sa_relationship_kwargs={"remote_side": "Dataset.id", "foreign_keys": "[Dataset.source_dataset_id]"}
    )
    derived_datasets: list["Dataset"] = Relationship(back_populates="source_dataset")
    root_dataset: "Dataset | None" = Relationship(
        sa_relationship_kwargs={"remote_side": "Dataset.id", "foreign_keys": "[Dataset.root_dataset_id]"}
    )
```

**Auto-population logic:**

```python
def create_derived_dataset(source_dataset_id: int, ...) -> Dataset:
    """Create a derived dataset with auto-populated root_dataset_id."""
    source_ds = session.get(Dataset, source_dataset_id)
    
    # Inherit root from parent, or parent becomes root if it has no root
    root_id = source_ds.root_dataset_id or source_ds.id
    
    return Dataset(
        source_dataset_id=source_dataset_id,
        root_dataset_id=root_id,
        ...
    )
```

**Example dataset records:**

```python
# Root dataset
Dataset(
    id=1,
    name="lmsys-1m",
    source="lmsys/lmsys-chat-1m",
    source_dataset_id=None,  # No parent
    root_dataset_id=None,    # This IS the root
    prompt_hash=None,
    summarization_model=None
)

# First-level derived dataset
Dataset(
    id=2,
    name="lmsys-1m-intent-v1",
    source="derived from lmsys-1m",
    source_dataset_id=1,     # Parent
    root_dataset_id=1,       # Root (inherited from parent or parent itself)
    prompt_hash="abc123...",
    summarization_model="gpt-4o-mini"
)

# Second-level derived dataset (chained summarization)
Dataset(
    id=3,
    name="lmsys-1m-intent-v1-classified",
    source="derived from lmsys-1m-intent-v1",
    source_dataset_id=2,     # Parent
    root_dataset_id=1,       # Root (same as parent's root)
    prompt_hash="def456...",
    summarization_model="gpt-4o-mini"
)
```

**Benefits of `root_dataset_id`:**

1. **Simple lineage queries:**
   ```sql
   -- Find all datasets derived from lmsys-1m (root_id=1)
   SELECT * FROM datasets WHERE root_dataset_id = 1;
   
   -- Find root of any derived dataset
   SELECT * FROM datasets d1
   JOIN datasets d2 ON d2.id = d1.root_dataset_id
   WHERE d1.name = 'lmsys-intent-classified';
   ```

2. **Cleanup operations:**
   ```bash
   # Delete root and all descendants
   uv run lmsys clear --dataset "lmsys-1m" --cascade
   ```

3. **Provenance visualization:**
   ```bash
   # Show full lineage tree
   uv run lmsys datasets lineage "lmsys-intent-classified"
   # Output:
   # lmsys-1m (root, id=1)
   #   └─> lmsys-intent-v1 (id=2, prompt=abc123)
   #       └─> lmsys-intent-classified (id=3, prompt=def456) ← current
   ```

### 3. Update `Query` Table Constraint

**Problem:** Current unique constraint on `conversation_id` prevents reusing same IDs across datasets.

**Solution:** Change to composite unique constraint on `(dataset_id, conversation_id)`.

```python
class Query(SQLModel, table=True):
    __tablename__ = "queries"
    __table_args__ = (
        Index("ix_queries_dataset_id", "dataset_id"),
        UniqueConstraint("dataset_id", "conversation_id", name="uq_query_dataset_conversation"),  # MODIFIED
    )
    
    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(sa_column=Column("dataset_id", ForeignKey("datasets.id", ondelete="CASCADE")))
    conversation_id: str = Field(index=True)  # REMOVED unique=True
    # ... rest of fields unchanged
```

**Migration impact:**
- ✅ Preserves `conversation_id` from source queries (enables joining across datasets)
- ✅ Allows multiple datasets to have queries with same `conversation_id`
- ⚠️ Requires database migration (Alembic or manual ALTER TABLE)

**Migration SQL:**
```sql
-- Drop old unique constraint
DROP INDEX IF EXISTS ix_queries_conversation_id;

-- Create new composite unique constraint
CREATE UNIQUE INDEX uq_query_dataset_conversation ON queries(dataset_id, conversation_id);
```

### 4. Query Metadata for Summarized Datasets

When creating summarized queries, preserve original metadata:

```python
Query(
    dataset_id=2,  # New dataset
    conversation_id="abc-123",  # SAME as original
    query_text="[LLM summary output]",  # NEW: Summarized text
    model="gpt-3.5",  # PRESERVE from original
    language="en",  # PRESERVE from original
    timestamp=original_timestamp,  # PRESERVE from original
    extra_metadata={
        "source_dataset_id": 1,
        "source_query_id": 12345,
        "summarization_model": "gpt-4o-mini",  # Track summarization model
        "original_length": 156,  # Optional: stats
        "summary_length": 42
    }
)
```

**Design rationale:**
- Metadata preservation enables filtering (e.g., "only English summaries")
- `conversation_id` preservation enables joining raw ↔ summarized datasets
- `extra_metadata` tracks full provenance

---

## File Structure & Implementation

### Files to Reference (Study These First)

1. **`src/lmsys_query_analysis/clustering/summarizer.py`**
   - Study: Async LLM batch processing with `instructor`
   - Pattern: `ClusterSummarizer._async_generate_batch_summaries()`
   - Copy: Concurrency control with `anyio.Semaphore`
   - Copy: Retry logic with `@retry` decorator
   - Copy: Progress bars with `rich.progress`

2. **`src/lmsys_query_analysis/db/loader.py`**
   - Study: Batch insert patterns (`executemany` with pre-check)
   - Pattern: Progress tracking during long operations
   - Copy: `chunk_iter()` helper for batching
   - Copy: SQLite PRAGMA tuning for bulk inserts

3. **`src/lmsys_query_analysis/clustering/embeddings.py`**
   - Study: Async embedding generation with multiple providers
   - Pattern: `EmbeddingGenerator.generate_embeddings_async()`
   - Copy: Provider-agnostic interface

4. **`src/lmsys_query_analysis/db/models.py`**
   - Modify: Add `Prompt`, update `Dataset`, change `Query` constraint
   - Reference: Existing relationship patterns

5. **`src/lmsys_query_analysis/cli/commands/clustering.py`**
   - Study: CLI command structure with Typer
   - Pattern: Database path handling, progress output
   - Copy: Error handling and validation

### Files to Create

#### 1. `src/lmsys_query_analysis/clustering/row_summarizer.py`

**Purpose:** LLM-based query summarization service (async batch processing).

**Key components:**

```python
"""LLM-based row-level query summarization.

Transforms raw queries into normalized/summarized text using LLMs.
Supports custom prompts with {query} and {examples} placeholders.
Serializes entire Query records as XML for rich context.
"""

import logging
import anyio
import instructor
from jinja2 import Template
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from ..db.models import Query

class Property(BaseModel):
    """Key-value property extracted from query.
    
    Enables flexible, prompt-driven feature extraction. The user specifies
    property names and types directly in the prompt:
    
    Examples in prompt:
    - "Extract user_frustration (int 0-10)"
    - "Extract category (str: code_gen, debugging, explanation)"
    - "Extract is_question (bool)"
    - "Extract complexity_score (float 0.0-1.0)"
    
    The LLM extracts these and returns them as Property objects.
    """
    name: str = Field(description="Name of the property (e.g., 'user_frustration', 'category')")
    value: str | int | float | bool | None = Field(description="Value of the property, type determined by prompt")

class QuerySummaryResponse(BaseModel):
    """Structured LLM response for query summarization.
    
    Uses instructor for type-safe structured outputs. The LLM returns JSON
    matching this schema, which we extract and store in the database.
    
    The flexible Property schema allows prompt-driven extraction:
    - User defines properties in the prompt itself
    - No need to modify Python code for new features
    - LLM infers types from prompt instructions
    """
    summary: str = Field(
        description="Normalized/summarized query text. This will become the query_text field in the derived dataset."
    )
    properties: list[Property] = Field(
        default_factory=list,
        description="Properties extracted from query as specified in the prompt. Each property has a name and typed value."
    )

# Jinja2 template for query XML serialization
QUERY_XML_TEMPLATE = Template("""<query>
  <conversation_id>{{ query.conversation_id }}</conversation_id>
  <model>{{ query.model }}</model>
  <query_text>{{ query.query_text }}</query_text>
  {%- if query.language %}
  <language>{{ query.language }}</language>
  {%- endif %}
  {%- if query.timestamp %}
  <timestamp>{{ query.timestamp.isoformat() }}</timestamp>
  {%- endif %}
  {%- if query.extra_metadata %}
  <extra_metadata>{{ query.extra_metadata | tojson }}</extra_metadata>
  {%- endif %}
</query>""")

# Jinja2 template for examples XML serialization
EXAMPLES_XML_TEMPLATE = Template("""<examples>
{%- for input_query, output in examples %}
  <example>
    <input>
      <query>
        <query_text>{{ input_query.query_text }}</query_text>
        <model>{{ input_query.model }}</model>
        {%- if input_query.language %}
        <language>{{ input_query.language }}</language>
        {%- endif %}
      </query>
    </input>
    <output>{{ output }}</output>
  </example>
{%- endfor %}
</examples>""")

def serialize_query_to_xml(query: Query) -> str:
    """Serialize Query record to XML format using Jinja2 template.
    
    Example output:
        <query>
          <conversation_id>abc-123</conversation_id>
          <model>gpt-3.5-turbo</model>
          <query_text>hey can you help me write a python script?</query_text>
          <language>en</language>
          <timestamp>2024-01-15T10:30:00Z</timestamp>
          <extra_metadata>{"source": "web"}</extra_metadata>
        </query>
    """
    return QUERY_XML_TEMPLATE.render(query=query)

def format_examples_xml(examples: list[tuple[Query, str]]) -> str:
    """Format few-shot examples as XML using Jinja2 template.
    
    Args:
        examples: List of (input_query, expected_output) tuples
    
    Returns:
        XML string with formatted examples
        
    Example output:
        <examples>
          <example>
            <input>
              <query>
                <query_text>help me code python</query_text>
                <model>gpt-3.5</model>
                <language>en</language>
              </query>
            </input>
            <output>Write Python code</output>
          </example>
        </examples>
    """
    return EXAMPLES_XML_TEMPLATE.render(examples=examples)

class RowSummarizer:
    """Summarize individual queries using LLM with custom prompts.
    
    Args:
        model: Model in format "provider/model_name" (e.g., "openai/gpt-4o-mini")
        prompt_template: Template string with {query} and optional {examples} placeholders
        examples: Optional list of (input_query, expected_output) for few-shot learning
        concurrency: Number of concurrent LLM requests (default: 100)
        api_key: Optional API key (or use env vars)
    
    Example:
        >>> summarizer = RowSummarizer(
        ...     model="openai/gpt-4o-mini",
        ...     prompt_template="Examples:\n{examples}\n\nNow extract intent: {query}",
        ...     examples=[(query1, "Write Python code"), (query2, "Debug JavaScript")]
        ... )
        >>> summaries = summarizer.summarize_batch(queries)
    """
    
    def __init__(
        self,
        model: str,
        prompt_template: str,
        examples: list[tuple[Query, str]] | None = None,
        concurrency: int = 100,
        api_key: str | None = None,
    ):
        self.model = model
        self.prompt_template = prompt_template
        self.examples = examples or []
        self.concurrency = concurrency
        self.logger = logging.getLogger("lmsys")
        
        # Validate prompt template has {query} placeholder
        if "{query}" not in prompt_template:
            raise ValueError("prompt_template must contain {query} placeholder")
        
        # Pre-format examples if provided
        self.formatted_examples = ""
        if self.examples:
            self.formatted_examples = format_examples_xml(self.examples)
            if "{examples}" not in prompt_template:
                self.logger.warning(
                    "Examples provided but {examples} placeholder not found in prompt template. "
                    "Examples will be ignored."
                )
        
        # Initialize async instructor client
        if api_key:
            self.async_client = instructor.from_provider(model, api_key=api_key, async_client=True)
        else:
            self.async_client = instructor.from_provider(model, async_client=True)
    
    def summarize_batch(
        self,
        queries: list[Query],
    ) -> dict[int, dict]:
        """Summarize batch of queries concurrently.
        
        Args:
            queries: List of Query objects
        
        Returns:
            Dict mapping query.id -> {"summary": str, "properties": dict}
        """
        return anyio.run(self._async_summarize_batch, queries)
    
    async def _async_summarize_batch(
        self,
        queries: list[Query],
    ) -> dict[int, dict]:
        """Async concurrent summarization."""
        results: dict[int, dict] = {}
        semaphore = anyio.Semaphore(self.concurrency)
        
        async def worker(query: Query):
            async with semaphore:
                result = await self._summarize_single(query)
                results[query.id] = result
        
        async with anyio.create_task_group() as tg:
            for query in queries:
                tg.start_soon(worker, query)
        
        return results
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    async def _summarize_single(self, query: Query) -> dict:
        """Summarize single query with retries.
        
        Args:
            query: Full Query record (will be serialized to XML)
        
        Returns:
            Dict with "summary" and "properties" keys
            Example: {
                "summary": "Write Python code",
                "properties": {
                    "intent": "code_generation",
                    "complexity": 2,
                    "programming_language": "python"
                }
            }
        """
        # Serialize query to XML
        query_xml = serialize_query_to_xml(query)
        
        # Format prompt with XML query and examples
        prompt_content = self.prompt_template.format(
            query=query_xml,
            examples=self.formatted_examples
        )
        
        messages = [
            {
                "role": "user",
                "content": prompt_content
            }
        ]
        
        # LLM returns structured JSON via instructor
        # Example: {"summary": "Write Python code", "properties": [...]}
        response = await self.async_client.chat.completions.create(
            response_model=QuerySummaryResponse,
            messages=messages,
        )
        
        # Convert properties list to dict for easy storage in extra_metadata
        properties_dict = {prop.name: prop.value for prop in response.properties}
        
        return {
            "summary": response.summary,
            "properties": properties_dict
        }
```

**Design notes:**
- High default concurrency (100) since gpt-4o-mini is fast
- Template validation at initialization
- Retry with exponential backoff
- Returns dict for easy merging with source data

#### 2. `src/lmsys_query_analysis/cli/commands/dataset_summary.py`

**Purpose:** CLI command for dataset summarization.

**Command signature:**

```python
"""CLI command for LLM-based dataset summarization."""

import typer
from rich.console import Console
from sqlmodel import select
from ...db.connection import Database
from ...db.models import Dataset, Query, Prompt
from ...clustering.row_summarizer import RowSummarizer
from ...clustering.embeddings import EmbeddingGenerator
from ...db.chroma import ChromaManager
import hashlib
from datetime import datetime
import json

app = typer.Typer()
console = Console()

@app.command()
def summarize(
    source_dataset: str = typer.Argument(..., help="Source dataset name"),
    output: str = typer.Option(..., "--output", help="Output dataset name"),
    prompt: str = typer.Option(..., "--prompt", help="Prompt template with {query} and optional {examples}"),
    model: str = typer.Option("openai/gpt-4o-mini", "--model", help="LLM model (provider/model)"),
    limit: int | None = typer.Option(None, "--limit", help="Max queries to summarize (first N)"),
    where: str | None = typer.Option(None, "--where", help="SQL WHERE clause for filtering"),
    examples_file: str | None = typer.Option(None, "--examples", help="JSONL file with few-shot examples"),
    example: list[str] | None = typer.Option(None, "--example", help="Inline example 'query_id:output'"),
    concurrency: int = typer.Option(100, "--concurrency", help="Concurrent LLM requests"),
    use_chroma: bool = typer.Option(False, "--use-chroma", help="Generate embeddings for summaries"),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model"),
    embedding_provider: str = typer.Option("openai", "--embedding-provider"),
    db_path: str = typer.Option("~/.lmsys-query-analysis/queries.db", "--db-path"),
    chroma_path: str = typer.Option("~/.lmsys-query-analysis/chroma", "--chroma-path"),
):
    """Summarize queries using LLM to create derived dataset.
    
    The {query} placeholder in the prompt is replaced with the full Query record serialized as XML,
    including fields like query_text, model, language, timestamp, and extra_metadata.
    
    Examples:
        # Basic summarization
        lmsys summarize "lmsys-1m" \\
          --output "lmsys-1m-intent" \\
          --prompt "Extract user intent from <query_text>: {query}" \\
          --limit 10000 \\
          --use-chroma
        
        # With few-shot examples from file
        lmsys summarize "lmsys-1m" \\
          --output "lmsys-1m-classified" \\
          --prompt "Examples:\\n{examples}\\n\\nClassify this query: {query}" \\
          --examples examples.jsonl
        
        # With inline examples
        lmsys summarize "lmsys-1m" \\
          --output "lmsys-1m-intent" \\
          --prompt "Examples: {examples}\\nIntent: {query}" \\
          --example "12345:Write Python code" \\
          --example "67890:Debug JavaScript error"
        
        # Metadata-aware prompt
        lmsys summarize "lmsys-1m" \\
          --output "lmsys-1m-normalized" \\
          --prompt "If <language> is not English, translate <query_text> to English. Otherwise extract intent: {query}"
    """
    
    # Implementation below...
```

**Few-shot examples file format (`examples.jsonl`):**

```jsonl
{"query_id": 12345, "output": "Write Python code"}
{"query_id": 67890, "output": "Debug JavaScript error"}
{"query_id": 11111, "output": "Translate Spanish to English"}
```

**Example: Prompt-Driven Property Extraction**

The power of the `Property` system is that you define what to extract directly in the prompt:

```bash
# Example 1: Multi-dimensional feature extraction
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-enriched" \
  --prompt "
Analyze this query and extract:

1. summary: One-sentence intent (str)
2. user_frustration: Frustration level 0-10 where 0=calm, 10=very frustrated (int)
3. complexity: Query complexity 1-5 where 1=simple, 5=expert-level (int)
4. category: Main category (str: code_gen, debugging, explanation, creative, translation, other)
5. requires_code_output: Does user expect code in response? (bool)
6. programming_language: If code-related, which language? (str or null)
7. is_question: Is this phrased as a question? (bool)

Query to analyze:
{query}
" \
  --limit 10000

# LLM returns:
# {
#   "summary": "Write Python function to reverse string",
#   "properties": [
#     {"name": "user_frustration", "value": 2},
#     {"name": "complexity", "value": 2},
#     {"name": "category", "value": "code_gen"},
#     {"name": "requires_code_output", "value": true},
#     {"name": "programming_language", "value": "python"},
#     {"name": "is_question", "value": false}
#   ]
# }
```

**Example 2: User Sentiment Analysis**

```bash
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-sentiment" \
  --prompt "
Extract:
- summary: Reformulate query neutrally (str)
- sentiment: positive, neutral, negative, frustrated, confused (str)
- politeness_score: 0-10 where 0=rude, 10=very polite (int)
- urgency: low, medium, high, emergency (str)
- has_please_or_thanks: Contains polite words? (bool)

Query: {query}
"
```

**Example 3: Domain-Specific Classification**

```bash
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-domains" \
  --prompt "
Extract:
- summary: Core task description (str)
- domain: tech, creative, education, business, personal, health, legal, other (str)
- subdomain: More specific category within domain (str)
- expertise_required: beginner, intermediate, advanced, expert (str)
- time_sensitivity: Does this seem time-sensitive? (bool)
- estimated_tokens_needed: Rough response length: short(<100), medium(100-500), long(>500) (str)

Query: {query}
"
```

**SQL Analysis with Extracted Properties:**

```sql
-- Find frustrated users with complex queries (high support priority)
SELECT query_text, 
       extra_metadata->>'user_frustration' as frustration,
       extra_metadata->>'complexity' as complexity
FROM queries
WHERE dataset_id = 2
  AND CAST(extra_metadata->>'user_frustration' AS INTEGER) >= 7
  AND CAST(extra_metadata->>'complexity' AS INTEGER) >= 4
ORDER BY CAST(extra_metadata->>'user_frustration' AS INTEGER) DESC
LIMIT 20;

-- Analyze category distribution
SELECT 
    extra_metadata->>'category' as category,
    AVG(CAST(extra_metadata->>'complexity' AS REAL)) as avg_complexity,
    AVG(CAST(extra_metadata->>'user_frustration' AS REAL)) as avg_frustration,
    COUNT(*) as count
FROM queries
WHERE dataset_id = 2
GROUP BY category
ORDER BY count DESC;

-- Find code generation queries that don't expect code output (potential misunderstanding)
SELECT query_text
FROM queries
WHERE dataset_id = 2
  AND extra_metadata->>'category' = 'code_gen'
  AND extra_metadata->>'requires_code_output' = 'false'
LIMIT 10;
```

**Implementation steps:**

1. **Load source dataset and queries**
   ```python
   db = Database(db_path)
   session = db.get_session()
   
   # Find source dataset
   source_ds = session.exec(
       select(Dataset).where(Dataset.name == source_dataset)
   ).first()
   if not source_ds:
       console.print(f"[red]Error: Dataset '{source_dataset}' not found")
       raise typer.Exit(1)
   
   # Build query to fetch full Query objects (not just id, query_text)
   stmt = select(Query).where(Query.dataset_id == source_ds.id)
   if where:
       stmt = stmt.where(text(where))
   if limit:
       stmt = stmt.limit(limit)
   
   queries = session.exec(stmt).all()
   console.print(f"[cyan]Loaded {len(queries)} queries from '{source_dataset}'")
   ```

1b. **Load few-shot examples (if provided)**
   ```python
   examples: list[tuple[Query, str]] = []
   
   # Load from file
   if examples_file:
       with open(examples_file, 'r') as f:
           for line in f:
               ex = json.loads(line)
               query_id = ex["query_id"]
               output = ex["output"]
               
               # Fetch full Query record
               query = session.get(Query, query_id)
               if not query:
                   console.print(f"[yellow]Warning: Query {query_id} not found, skipping example")
                   continue
               
               examples.append((query, output))
       
       console.print(f"[cyan]Loaded {len(examples)} examples from {examples_file}")
   
   # Load inline examples
   if example:
       for ex_str in example:
           try:
               query_id_str, output = ex_str.split(':', 1)
               query_id = int(query_id_str)
               
               query = session.get(Query, query_id)
               if not query:
                   console.print(f"[yellow]Warning: Query {query_id} not found, skipping example")
                   continue
               
               examples.append((query, output))
           except ValueError:
               console.print(f"[yellow]Warning: Invalid example format '{ex_str}', expected 'query_id:output'")
       
       console.print(f"[cyan]Loaded {len(examples)} inline examples")
   ```

2. **Check/create output dataset**
   ```python
   # Check if output exists
   existing = session.exec(select(Dataset).where(Dataset.name == output)).first()
   if existing:
       # Auto-append timestamp
       timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
       output = f"{output}-{timestamp}"
       console.print(f"[yellow]Output dataset exists, using: {output}")
   ```

3. **Create/get Prompt record**
   ```python
   prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
   
   prompt_record = session.exec(
       select(Prompt).where(Prompt.prompt_hash == prompt_hash)
   ).first()
   
   if not prompt_record:
       prompt_record = Prompt(
           prompt_hash=prompt_hash,
           prompt_text=prompt
       )
       session.add(prompt_record)
       session.commit()
       console.print(f"[green]Created new prompt: {prompt_hash[:8]}...")
   else:
       console.print(f"[cyan]Using existing prompt: {prompt_hash[:8]}...")
   ```

4. **Run summarization with examples**
   ```python
   summarizer = RowSummarizer(
       model=model,
       prompt_template=prompt,
       examples=examples,  # Pass few-shot examples
       concurrency=concurrency
   )
   
   with Progress(...) as progress:
       task = progress.add_task("[cyan]Summarizing queries...", total=len(queries))
       # Pass full Query objects
       results = summarizer.summarize_batch(queries)
       progress.update(task, completed=len(queries))
   
   # results is now dict[int, dict] mapping query.id -> {"summary": str, "properties": dict}
   # Example: {12345: {"summary": "Write Python code", "properties": {"intent": "code_gen", "complexity": 2}}}
   ```

5. **Create output dataset and insert summarized queries with properties**
   ```python
   # Compute root_dataset_id
   root_id = source_ds.root_dataset_id or source_ds.id
   
   output_ds = Dataset(
       name=output,
       source=f"derived from {source_dataset}",
       source_dataset_id=source_ds.id,
       root_dataset_id=root_id,  # Inherit root
       prompt_hash=prompt_hash,
       summarization_model=model,
       description=f"Summarized from {source_dataset} using: {prompt[:50]}..."
   )
   session.add(output_ds)
   session.commit()
   session.refresh(output_ds)
   
   # Build Query objects with merged metadata
   new_queries = []
   for source_query in queries:
       result = results[source_query.id]
       
       # Merge properties into extra_metadata
       merged_metadata = {
           "source_dataset_id": source_ds.id,
           "source_query_id": source_query.id,
           "summarization_model": model,
           **result["properties"]  # Add extracted properties (intent, complexity, etc.)
       }
       
       # Preserve original extra_metadata if exists
       if source_query.extra_metadata:
           merged_metadata.update(source_query.extra_metadata)
       
       new_query = Query(
           dataset_id=output_ds.id,
           conversation_id=source_query.conversation_id,  # PRESERVE
           model=source_query.model,  # PRESERVE
           query_text=result["summary"],  # NEW: LLM summary
           language=source_query.language,  # PRESERVE
           timestamp=source_query.timestamp,  # PRESERVE
           extra_metadata=merged_metadata
       )
       new_queries.append(new_query)
   
   # Batch insert
   session.add_all(new_queries)
   session.commit()
   console.print(f"[green]Created {len(new_queries)} summarized queries")
   ```

6. **Optional: Generate embeddings**
   ```python
   if use_chroma:
       embedder = EmbeddingGenerator(
           model_name=embedding_model,
           provider=embedding_provider
       )
       # Generate embeddings and write to ChromaDB
       # (Similar to loader.py pattern)
   ```

### Files to Modify

#### 1. `src/lmsys_query_analysis/db/models.py`

Add schema changes described above:
- New `Prompt` table
- Update `Dataset` with foreign keys
- Update `Query` constraint

#### 2. `src/lmsys_query_analysis/cli/main.py`

Register new command:

```python
from .commands import dataset_summary

# Add to app
app.command("summarize")(dataset_summary.summarize)
```

---

## Testing Strategy

### Unit Tests

#### 1. `tests/test_row_summarizer.py`

```python
import pytest
from lmsys_query_analysis.clustering.row_summarizer import RowSummarizer, QuerySummaryResponse

def test_prompt_template_validation():
    """Test that prompt template must contain {query}."""
    with pytest.raises(ValueError, match="must contain {query}"):
        RowSummarizer(
            model="openai/gpt-4o-mini",
            prompt_template="Summarize this"  # Missing {query}
        )

@pytest.mark.smoke
async def test_single_query_summarization():
    """Smoke test: Summarize single query."""
    summarizer = RowSummarizer(
        model="openai/gpt-4o-mini",
        prompt_template="Extract intent in 5 words: {query}",
        concurrency=1
    )
    
    result = await summarizer._summarize_single(
        "hey can you help me write a python script to reverse a string?"
    )
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result.split()) <= 10  # Roughly 5 words

@pytest.mark.smoke
def test_batch_summarization():
    """Smoke test: Batch summarization."""
    summarizer = RowSummarizer(
        model="openai/gpt-4o-mini",
        prompt_template="Summarize: {query}",
        concurrency=10
    )
    
    queries = [
        (1, "Write Python code"),
        (2, "Help with JavaScript"),
        (3, "Explain React hooks")
    ]
    
    results = summarizer.summarize_batch(queries)
    
    assert len(results) == 3
    assert all(isinstance(v, str) for v in results.values())
```

#### 2. `tests/test_dataset_summary_models.py`

```python
from lmsys_query_analysis.db.models import Dataset, Query, Prompt
from sqlmodel import Session, select
import hashlib

def test_prompt_hash_uniqueness(session: Session):
    """Test prompt hash deduplication."""
    prompt_text = "Extract intent: {query}"
    hash1 = hashlib.sha256(prompt_text.encode()).hexdigest()
    
    p1 = Prompt(prompt_hash=hash1, prompt_text=prompt_text)
    session.add(p1)
    session.commit()
    
    # Try to insert duplicate
    p2 = Prompt(prompt_hash=hash1, prompt_text=prompt_text)
    session.add(p2)
    
    with pytest.raises(Exception):  # IntegrityError
        session.commit()

def test_dataset_lineage(session: Session):
    """Test dataset -> derived_dataset relationship."""
    # Create base dataset
    base = Dataset(name="base", source="test")
    session.add(base)
    session.commit()
    
    # Create derived dataset
    prompt = Prompt(
        prompt_hash="abc123",
        prompt_text="Test: {query}"
    )
    session.add(prompt)
    session.commit()
    
    derived = Dataset(
        name="derived",
        source="from base",
        source_dataset_id=base.id,
        prompt_hash="abc123",
        summarization_model="gpt-4o-mini"
    )
    session.add(derived)
    session.commit()
    
    # Verify relationship
    assert derived.source_dataset.id == base.id
    assert base.derived_datasets[0].id == derived.id

def test_query_conversation_id_uniqueness_per_dataset(session: Session):
    """Test conversation_id can be reused across datasets."""
    ds1 = Dataset(name="ds1", source="test1")
    ds2 = Dataset(name="ds2", source="test2")
    session.add_all([ds1, ds2])
    session.commit()
    
    # Same conversation_id in different datasets should work
    q1 = Query(
        dataset_id=ds1.id,
        conversation_id="abc-123",
        model="test",
        query_text="test1"
    )
    q2 = Query(
        dataset_id=ds2.id,
        conversation_id="abc-123",  # SAME ID
        model="test",
        query_text="test2"
    )
    session.add_all([q1, q2])
    session.commit()  # Should NOT raise error
    
    # But duplicate within same dataset should fail
    q3 = Query(
        dataset_id=ds1.id,
        conversation_id="abc-123",  # DUPLICATE in ds1
        model="test",
        query_text="test3"
    )
    session.add(q3)
    
    with pytest.raises(Exception):  # IntegrityError
        session.commit()
```

### Integration Tests

#### `tests/integration/test_summarize_command.py`

```python
import pytest
from typer.testing import CliRunner
from lmsys_query_analysis.cli.main import app
from lmsys_query_analysis.db.connection import Database
from sqlmodel import select

@pytest.fixture
def test_db(tmp_path):
    """Create test database with sample data."""
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    db.create_tables()
    
    # Insert test dataset with queries
    session = db.get_session()
    ds = Dataset(name="test-dataset", source="test")
    session.add(ds)
    session.commit()
    session.refresh(ds)
    
    queries = [
        Query(
            dataset_id=ds.id,
            conversation_id=f"conv-{i}",
            model="test",
            query_text=f"Test query {i}",
            language="en"
        )
        for i in range(10)
    ]
    session.add_all(queries)
    session.commit()
    session.close()
    
    return db_path

@pytest.mark.smoke
def test_summarize_command_end_to_end(test_db):
    """Test full summarize command."""
    runner = CliRunner()
    
    result = runner.invoke(app, [
        "summarize",
        "test-dataset",
        "--output", "test-summarized",
        "--prompt", "Simplify: {query}",
        "--model", "openai/gpt-4o-mini",
        "--limit", "5",
        "--db-path", str(test_db)
    ])
    
    assert result.exit_code == 0
    
    # Verify output dataset created
    db = Database(str(test_db))
    session = db.get_session()
    
    output_ds = session.exec(
        select(Dataset).where(Dataset.name == "test-summarized")
    ).first()
    
    assert output_ds is not None
    assert output_ds.source_dataset_id is not None
    assert output_ds.prompt_hash is not None
    assert output_ds.summarization_model == "openai/gpt-4o-mini"
    
    # Verify queries created
    queries = session.exec(
        select(Query).where(Query.dataset_id == output_ds.id)
    ).all()
    
    assert len(queries) == 5
    assert all(q.query_text != f"Test query {i}" for i, q in enumerate(queries))
```

### Manual Smoke Test Script

#### `smoketest-summarize.sh`

```bash
#!/bin/bash
set -e

echo "=== Summarization Smoke Test ==="

# 1. Load small dataset
echo "Loading test dataset..."
uv run lmsys clear --yes
uv run lmsys load --limit 100

# 2. Run summarization with intent extraction
echo "Running intent extraction..."
uv run lmsys summarize "lmsys-chat-1m" \
  --output "lmsys-intent" \
  --prompt "Extract the user's intent in one concise sentence: {query}" \
  --limit 50 \
  --model openai/gpt-4o-mini \
  --use-chroma

# 3. Cluster both raw and summarized
echo "Clustering raw dataset..."
uv run lmsys cluster kmeans --n-clusters 10 --use-chroma

echo "Clustering summarized dataset..."
uv run lmsys cluster kmeans --dataset "lmsys-intent" --n-clusters 10 --use-chroma

# 4. Compare cluster quality
echo "Generating summaries for raw clusters..."
RAW_RUN=$(uv run lmsys runs --latest | grep kmeans | head -1 | awk '{print $1}')
uv run lmsys summarize $RAW_RUN --alias "raw-clusters"

echo "Generating summaries for intent clusters..."
INTENT_RUN=$(uv run lmsys runs --latest | grep kmeans | head -1 | awk '{print $1}')
uv run lmsys summarize $INTENT_RUN --alias "intent-clusters"

echo "=== Smoke test complete ==="
echo "Compare cluster quality manually:"
echo "  uv run lmsys list-clusters $RAW_RUN"
echo "  uv run lmsys list-clusters $INTENT_RUN"
```

---

## CLI Usage for Autonomous Agents

### Autonomous Hypothesis Testing Workflow

Claude can now autonomously test data hypotheses by creating custom summarization prompts.

#### Example 1: Jailbreak Detection Hypothesis

**Claude's reasoning:**
> "I hypothesize that 10-15% of queries are jailbreak attempts. Let me create a classification dataset to test this."

**Commands Claude would run:**

```bash
# Step 1: Create binary classification dataset
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-jailbreak-classified" \
  --prompt "Classify this query as: 'safe', 'jailbreak_roleplay', 'jailbreak_injection', or 'jailbreak_other': {query}" \
  --limit 10000 \
  --model openai/gpt-4o-mini \
  --use-chroma

# Step 2: Cluster on classifications (should produce ~4 clusters)
uv run lmsys cluster kmeans \
  --dataset "lmsys-jailbreak-classified" \
  --n-clusters 4 \
  --use-chroma

# Step 3: Get cluster sizes to validate hypothesis
CLASSIFICATION_RUN=$(uv run lmsys runs --latest | grep kmeans | head -1 | awk '{print $1}')
uv run lmsys list-clusters $CLASSIFICATION_RUN

# Step 4: Generate cluster summaries
uv run lmsys summarize $CLASSIFICATION_RUN --alias "jailbreak-analysis"

# Step 5: Search for specific jailbreak patterns
uv run lmsys search "roleplay" \
  --run-id $CLASSIFICATION_RUN \
  --search-type clusters \
  --xml

# Step 6: Export for further analysis
uv run lmsys export $CLASSIFICATION_RUN \
  --format csv \
  --output jailbreak-analysis.csv
```

**SQL validation query Claude would write:**

```sql
-- Count queries by classification
SELECT 
    qc.cluster_id,
    cs.title as classification,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM queries WHERE dataset_id = 2), 2) as percentage
FROM queries q
JOIN query_clusters qc ON q.id = qc.query_id
JOIN cluster_summaries cs ON qc.cluster_id = cs.cluster_id AND qc.run_id = cs.run_id
WHERE q.dataset_id = 2  -- jailbreak-classified dataset
  AND qc.run_id = 'kmeans-4-...'
GROUP BY qc.cluster_id, cs.title
ORDER BY count DESC;
```

**Expected output:**

```
cluster_id | classification        | count | percentage
-----------|-----------------------|-------|------------
0          | safe                  | 8543  | 85.43%
1          | jailbreak_roleplay    | 892   | 8.92%
2          | jailbreak_injection   | 421   | 4.21%
3          | jailbreak_other       | 144   | 1.44%
```

**Claude's conclusion:**
> "Hypothesis validated: ~14.5% of queries are jailbreak attempts. Breaking down by type:
> - Roleplay-based (8.9%): Users create fictional scenarios
> - Injection-based (4.2%): Users try prompt injection
> - Other techniques (1.4%): Obfuscation, encoding, etc.
> 
> Recommendation: Implement roleplay detection as highest-impact safety improvement."

#### Example 2: Programming Language Distribution

**Hypothesis:** "What percentage of coding queries are for each language?"

```bash
# Create language-specific classification
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-coding-languages" \
  --prompt "If this is a coding query, identify the primary programming language mentioned. If not coding-related, return 'not_coding'. Answer with just the language name (python, javascript, java, c++, etc.): {query}" \
  --where "query_text LIKE '%code%' OR query_text LIKE '%program%' OR query_text LIKE '%script%'" \
  --limit 20000 \
  --model openai/gpt-4o-mini \
  --use-chroma

# Cluster to group by language
uv run lmsys cluster kmeans \
  --dataset "lmsys-coding-languages" \
  --n-clusters 20 \
  --use-chroma

# Analyze distribution
LANG_RUN=$(uv run lmsys runs --latest | grep kmeans | head -1 | awk '{print $1}')
uv run lmsys summarize $LANG_RUN --alias "language-distribution"
uv run lmsys list-clusters $LANG_RUN
```

**SQL analysis:**

```sql
-- Language distribution with examples
SELECT 
    cs.title as language,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM queries WHERE dataset_id = 3), 2) as percentage,
    GROUP_CONCAT(q.query_text, '; ') as examples
FROM queries q
JOIN query_clusters qc ON q.id = qc.query_id
JOIN cluster_summaries cs ON qc.cluster_id = cs.cluster_id
WHERE qc.run_id = 'kmeans-20-...'
GROUP BY cs.title
ORDER BY count DESC
LIMIT 10;
```

#### Example 3: Query Complexity Classification

**Hypothesis:** "Beginner vs. expert queries cluster differently."

```bash
# Classify by complexity
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-complexity" \
  --prompt "Rate this query's technical complexity: 'beginner' (simple, general questions), 'intermediate' (specific tools/concepts), or 'expert' (advanced, multi-part problems). Answer with just one word: {query}" \
  --limit 10000 \
  --model openai/gpt-4o-mini

# Cluster by complexity (should produce ~3 main clusters)
uv run lmsys cluster kmeans \
  --dataset "lmsys-complexity" \
  --n-clusters 5 \
  --use-chroma

# Compare to model performance
# (If extra_metadata contains success/failure)
```

**SQL: Complexity vs. Model Performance**

```sql
SELECT 
    cs.title as complexity_level,
    q.model,
    COUNT(*) as total_queries,
    SUM(CASE WHEN q.extra_metadata->>'success' = 'true' THEN 1 ELSE 0 END) as successful,
    ROUND(SUM(CASE WHEN q.extra_metadata->>'success' = 'true' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate
FROM queries q
JOIN query_clusters qc ON q.id = qc.query_id
JOIN cluster_summaries cs ON qc.cluster_id = cs.cluster_id
WHERE qc.run_id = 'kmeans-5-...'
GROUP BY cs.title, q.model
ORDER BY complexity_level, model;
```

#### Example 4: Iterative Prompt Refinement

**Scenario:** Initial clustering is too coarse, Claude refines the prompt.

```bash
# Iteration 1: Generic intent extraction
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-intent-v1" \
  --prompt "What does the user want? {query}" \
  --limit 5000

uv run lmsys cluster kmeans --dataset "lmsys-intent-v1" --n-clusters 50 --use-chroma

# Claude inspects clusters and finds they're still too heterogeneous
# Generate refined prompt using LLM

# Iteration 2: More specific intent extraction
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-intent-v2" \
  --prompt "Classify user intent into ONE category: code_generation, debugging, explanation, creative_writing, translation, general_conversation, homework_help, or other. Answer with just the category: {query}" \
  --limit 5000

uv run lmsys cluster kmeans --dataset "lmsys-intent-v2" --n-clusters 10 --use-chroma

# Compare clustering quality
V1_RUN=$(uv run lmsys runs | grep "lmsys-intent-v1" | awk '{print $1}')
V2_RUN=$(uv run lmsys runs | grep "lmsys-intent-v2" | awk '{print $1}')

# Use Explore subagent to validate
```

**Validation with Explore agent:**

```
Use Explore agent to compare clustering quality:

Commands:
1. uv run lmsys list-clusters $V1_RUN --limit 10
2. uv run lmsys list-clusters $V2_RUN --limit 10
3. uv run lmsys inspect $V1_RUN 3  # Inspect largest cluster
4. uv run lmsys inspect $V2_RUN 3

Analysis:
- Calculate max cluster size % for v1 vs v2
- Sample 10 queries from largest cluster in each
- Compute semantic coherence (% matching cluster label)
- Report which version has better cluster separation
```

### Advanced: Multi-Stage Summarization Pipeline

Claude can chain summarizations for multi-step transformations:

```bash
# Stage 1: Extract intent
uv run lmsys summarize "lmsys-1m" \
  --output "stage1-intent" \
  --prompt "Extract user intent: {query}" \
  --limit 10000

# Stage 2: Classify domain (on intents)
uv run lmsys summarize "stage1-intent" \
  --output "stage2-domain" \
  --prompt "What domain is this? (tech, creative, education, business, personal): {query}"

# Stage 3: Rate complexity (on domains)
uv run lmsys summarize "stage2-domain" \
  --output "stage3-complexity" \
  --prompt "Rate complexity (1-5): {query}" \
  --where "query_text = 'tech'"  # Only tech queries

# Final clustering on multi-dimensional representation
uv run lmsys cluster kmeans --dataset "stage3-complexity" --n-clusters 20
```

---

## Data Analysis Patterns for Claude

### Pattern 1: Hypothesis-Driven Exploration

**Template workflow:**

1. **Observe anomaly** in initial clustering
2. **Form hypothesis** about what's happening
3. **Create classification prompt** to test hypothesis
4. **Summarize dataset** with classification prompt
5. **Cluster classified dataset**
6. **Validate with SQL queries** and cluster inspection
7. **Report findings** with evidence

**Example hypotheses Claude might test:**

- "Users asking for homework help phrase queries differently than professionals"
- "Multi-turn conversations have lower success rates"
- "Non-English queries cluster separately even with similar intent"
- "Jailbreak attempts follow specific linguistic patterns"
- "Code debugging queries mention error messages 80% of the time"

### Pattern 2: Feature Extraction for Enrichment

Use summarization to extract structured features:

```bash
# Extract sentiment
uv run lmsys summarize "lmsys-1m" \
  --output "enriched-sentiment" \
  --prompt "User sentiment (positive, neutral, negative, frustrated): {query}"

# Extract technical entities
uv run lmsys summarize "lmsys-1m" \
  --output "enriched-entities" \
  --prompt "List programming languages, frameworks, or tools mentioned: {query}" \
  --where "query_text LIKE '%code%'"

# Extract question type
uv run lmsys summarize "lmsys-1m" \
  --output "enriched-question-type" \
  --prompt "Question type: how_to, what_is, why, debugging, comparison, recommendation, other: {query}"
```

### Pattern 3: Comparative Analysis

Compare behavior across different subsets:

```bash
# Create domain-specific summaries
uv run lmsys summarize "lmsys-1m" \
  --output "python-queries-intent" \
  --prompt "Specific Python task: {query}" \
  --where "query_text LIKE '%python%'"

uv run lmsys summarize "lmsys-1m" \
  --output "javascript-queries-intent" \
  --prompt "Specific JavaScript task: {query}" \
  --where "query_text LIKE '%javascript%'"

# Cluster each and compare
uv run lmsys cluster kmeans --dataset "python-queries-intent" --n-clusters 30
uv run lmsys cluster kmeans --dataset "javascript-queries-intent" --n-clusters 30

# Compare cluster distributions
```

### Pattern 4: Quality Validation Loop

Use summarization to validate clustering quality:

```bash
# After initial clustering, summarize cluster summaries
# (meta-summarization to check if clusters make sense)

# 1. Get cluster summaries from run
CLUSTER_RUN="kmeans-200-..."

# 2. Export cluster summaries to temp dataset
# (Would need new command: `lmsys export-cluster-summaries`)

# 3. Re-cluster the summaries to see if higher-level patterns emerge
uv run lmsys summarize "cluster-summaries-export" \
  --output "meta-clusters" \
  --prompt "What high-level category does this cluster belong to? {query}"

# 4. Cluster meta-summaries (should align with hierarchy)
uv run lmsys cluster kmeans --dataset "meta-clusters" --n-clusters 20
```

---

## Expected Impact & Success Metrics

### Clustering Quality Improvements

**Before (Raw Queries):**
- KMeans coherence: ~30-40%
- Cluster size distribution: Power law (one huge cluster)
- Silhouette score: ~0.15-0.25
- Manual inspection: 50%+ queries mismatched in large clusters

**After (Summarized Queries - Intent Extraction):**
- Expected coherence: ~70-85%
- Cluster size distribution: More balanced
- Silhouette score: ~0.35-0.50
- Manual inspection: 80%+ semantic alignment

### Agent Autonomy Gains

**Current state:**
- Agent can cluster and inspect
- Limited ability to test custom hypotheses
- Relies on predefined analysis patterns

**With summarization:**
- ✅ Agent can create custom classification schemes
- ✅ Agent can iteratively refine prompts based on results
- ✅ Agent can extract features for enrichment
- ✅ Agent can test behavioral hypotheses (jailbreaks, expertise levels, etc.)
- ✅ Agent can validate clustering quality objectively

### Performance Characteristics

**Cost estimation (for 100K queries with gpt-4o-mini):**
- Input tokens: ~100K queries × 50 tokens avg = 5M tokens
- Output tokens: ~100K summaries × 10 tokens avg = 1M tokens
- Total cost: ~$0.75 (5M × $0.15/1M) = **very cheap**

**Speed estimation (concurrency=100):**
- Throughput: ~1000 queries/minute
- 100K queries: ~100 minutes (~1.5 hours)

**Storage impact:**
- Summarized queries typically shorter than raw
- Minimal database growth (same schema)

---

## Future Extensions

### 1. LLM-Driven Prompt Generation

Command for Claude to auto-generate prompts:

```bash
uv run lmsys prompts generate \
  --based-on "Extract intent: {query}" \
  --goal "Better distinguish beginner vs expert queries" \
  --sample-run "kmeans-200-..." \
  --sample-clusters 12,47,89 \
  --output "intent-v2.txt"
```

Would use LLM to:
1. Read current prompt
2. Inspect sample clusters
3. Identify weaknesses
4. Generate refined prompt
5. Optionally A/B test on subset

### 2. Automatic Prompt Optimization

```bash
uv run lmsys prompts optimize \
  --dataset "lmsys-1m" \
  --target-metric coherence \
  --budget 1000 \
  --iterations 5
```

Would:
- Generate candidate prompts
- Test each on subset
- Cluster and measure coherence
- Select best prompt
- Iterate

### 3. Multi-Modal Summarization

Support structured outputs:

```python
class EnrichedQuerySummary(BaseModel):
    intent: str
    domain: str
    complexity: int  # 1-5
    sentiment: str
    entities: list[str]
```

Store as JSON in `query_text` or separate columns.

### 4. Prompt Library & Sharing

```bash
# Export prompt for reuse
uv run lmsys prompts export "intent-v2" --output prompts/intent-v2.yaml

# Import community prompts
uv run lmsys prompts import prompts/jailbreak-detection.yaml

# List available prompts
uv run lmsys prompts list --sort-by usage_count
```

---

## Open Questions & Design Decisions

1. **Should we support streaming for very large datasets?**
   - Current design loads all queries into memory
   - For 1M+ queries, might need streaming batches

2. **Should we track prompt lineage automatically?**
   - If Claude generates prompts, store `generated_from` chain
   - Enables "prompt evolution" analysis

3. **Should we add `--dry-run` mode for cost estimation?**
   - Preview what would be summarized
   - Estimate API costs before running

4. **Should we support post-summarization validation?**
   - Sample N summaries and verify quality
   - Catch bad prompts early

5. **Should we add built-in prompt templates?**
   - Pre-configured prompts for common tasks
   - `--preset intent-extraction`
   - `--preset classification`
   - `--preset complexity-rating`

---

## Documentation Updates: CLAUDE.md

### New Section to Add: "Phase 4.5: Dataset Transformation via Row Summarization"

Add this section to CLAUDE.md after the existing clustering sections to guide Claude on using row summarization for hypothesis testing:

```markdown
### Phase 4.5: Dataset Transformation via Row Summarization

Use `lmsys summarize` to create **derived datasets** that test specific hypotheses through LLM-driven query transformation.

**Key insight**: Summarization enables you to reshape the data to validate behavioral hypotheses without pre-defined categories.

#### When to Use Row Summarization

- **Hypothesis testing**: "Are 15% of queries jailbreak attempts?" → Create classification dataset
- **Feature extraction**: Extract intent, sentiment, complexity, domain from raw queries
- **Normalization**: Collapse lexical diversity (multiple phrasings → single intent)
- **Multi-lingual analysis**: Translate/normalize across languages
- **Metadata-aware classification**: Use model, language, timestamp context for better categorization

#### Basic Workflow

```bash
# 1. Form hypothesis
# Hypothesis: "20% of queries are jailbreak attempts"

# 2. Create classification prompt with few-shot examples
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-jailbreak-classified" \
  --prompt "Classify query as: safe, jailbreak_roleplay, jailbreak_injection, jailbreak_other\n\nQuery: {query}" \
  --example "12345:safe" \
  --example "67890:jailbreak_roleplay" \
  --example "11111:jailbreak_injection" \
  --limit 10000 \
  --use-chroma

# 3. Cluster on classifications (expect ~4 clusters)
uv run lmsys cluster kmeans --dataset "lmsys-jailbreak-classified" --n-clusters 5 --use-chroma

# 4. Validate hypothesis with cluster sizes
CLASSIFY_RUN=$(uv run lmsys runs --latest | grep kmeans | head -1 | awk '{print $1}')
uv run lmsys list-clusters $CLASSIFY_RUN

# 5. SQL validation
sqlite3 ~/.lmsys-query-analysis/queries.db <<SQL
SELECT 
    cs.title as classification,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM queries WHERE dataset_id = 2), 2) as percentage
FROM queries q
JOIN query_clusters qc ON q.id = qc.query_id
JOIN cluster_summaries cs ON qc.cluster_id = cs.cluster_id
WHERE q.dataset_id = 2 AND qc.run_id = '$CLASSIFY_RUN'
GROUP BY cs.title
ORDER BY count DESC;
SQL
```

#### XML-Aware Prompts (Leverage Full Metadata)

The `{query}` placeholder contains the **full Query record as XML**, not just text. Use this for sophisticated prompts:

```bash
# Language-aware normalization
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-normalized" \
  --prompt "If <language> is not 'en', translate <query_text> to English first. Then extract intent in one sentence.\n\nQuery: {query}"

# Model-aware complexity rating
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-complexity" \
  --prompt "Rate query complexity (1-5). If <model> is gpt-3.5, user likely expected simpler response (factor this in).\n\nQuery: {query}"

# Temporal pattern detection
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-temporal" \
  --prompt "Extract time-sensitive aspects from <query_text>. Note if <timestamp> is weekend vs weekday.\n\nQuery: {query}" \
  --where "timestamp IS NOT NULL"
```

#### Few-Shot Learning Patterns

**Pattern 1: Bootstrap from manual examples**

```bash
# Manually identify 5-10 representative examples
# Export them to examples.jsonl:
cat > examples.jsonl <<EOF
{"query_id": 12345, "output": "Write Python code"}
{"query_id": 67890, "output": "Debug JavaScript error"}
{"query_id": 11111, "output": "Translate Spanish text"}
{"query_id": 22222, "output": "Generate creative story"}
{"query_id": 33333, "output": "Explain technical concept"}
EOF

# Use examples to guide classification
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-intent-classified" \
  --prompt "Examples:\n{examples}\n\nClassify this query into the same categories:\n{query}" \
  --examples examples.jsonl \
  --limit 50000
```

**Pattern 2: Iterative refinement**

```bash
# Iteration 1: Broad classification
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-v1" \
  --prompt "Extract user intent: {query}" \
  --limit 5000

# Cluster and inspect
uv run lmsys cluster kmeans --dataset "lmsys-v1" --n-clusters 20
uv run lmsys list-clusters <RUN_ID>

# Iteration 2: Refined based on findings
# (Add examples of confusing cases discovered in v1)
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-v2" \
  --prompt "Classify intent. Be specific: distinguish 'write code' from 'debug code' from 'explain code'.\n\nExamples:\n{examples}\n\nQuery: {query}" \
  --example "id1:write_python_code" \
  --example "id2:debug_python_error" \
  --example "id3:explain_python_concept" \
  --limit 5000
```

#### Multi-Stage Transformation Pipelines

Chain summarizations for complex analysis:

```bash
# Stage 1: Extract domain
uv run lmsys summarize "lmsys-1m" \
  --output "stage1-domain" \
  --prompt "What domain? (tech, creative, education, business, personal, other): {query}" \
  --limit 10000

# Stage 2: Classify by expertise level (only tech queries)
uv run lmsys summarize "stage1-domain" \
  --output "stage2-expertise" \
  --prompt "Classify technical expertise: beginner, intermediate, expert: {query}" \
  --where "query_text = 'tech'"

# Stage 3: Extract specific framework (only expert tech)
uv run lmsys summarize "stage2-expertise" \
  --output "stage3-framework" \
  --prompt "Which framework/library: react, vue, angular, django, flask, fastapi, other: {query}" \
  --where "query_text = 'expert'"

# Final clustering on multi-dimensional feature space
uv run lmsys cluster kmeans --dataset "stage3-framework" --n-clusters 15
```

#### Comparison Workflow: Raw vs Summarized

Always compare clustering quality:

```bash
# Baseline: Cluster raw data
uv run lmsys cluster kmeans --dataset "lmsys-1m" --n-clusters 50 --use-chroma
RAW_RUN=$(uv run lmsys runs --latest | grep kmeans | head -1 | awk '{print $1}')

# Summarized: Cluster intent-extracted data
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-intent" \
  --prompt "Extract user intent in one sentence: {query}" \
  --limit 10000 \
  --use-chroma

uv run lmsys cluster kmeans --dataset "lmsys-intent" --n-clusters 50 --use-chroma
INTENT_RUN=$(uv run lmsys runs --latest | grep kmeans | head -1 | awk '{print $1}')

# Generate summaries for both
uv run lmsys summarize $RAW_RUN --alias "raw-clusters"
uv run lmsys summarize $INTENT_RUN --alias "intent-clusters"

# Compare (use Explore subagent for detailed analysis)
# Expected: Intent-based clustering shows 70-85% coherence vs 30-40% for raw
```

#### Cost Management

Estimate costs before running:

```bash
# Check query count
sqlite3 ~/.lmsys-query-analysis/queries.db "SELECT COUNT(*) FROM queries WHERE dataset_id = 1;"

# Estimate:
# - 100K queries × 50 tokens avg × $0.15/1M tokens = $0.75 (gpt-4o-mini)
# - 100K queries × 10 tokens output × $0.60/1M tokens = $0.60
# - Total: ~$1.35 for 100K queries

# Start with small sample
uv run lmsys summarize "lmsys-1m" --output "lmsys-test" --limit 100 --prompt "Intent: {query}"
# Validate quality, then scale up
```

#### Autonomous Prompt Generation (Future)

Claude can generate custom prompts based on observed patterns:

```python
# (Pseudo-code for future implementation)
# 1. Inspect low-quality clusters
uv run lmsys inspect <RUN_ID> <BAD_CLUSTER_ID>

# 2. Identify pattern: "Cluster mixes 'write code' and 'debug code'"

# 3. Generate refined prompt:
new_prompt = """
Classify query precisely:
- write_code: User wants new code written
- debug_code: User has broken code and needs fix
- explain_code: User wants conceptual explanation
- other_code: Other code-related request

Query: {query}
"""

# 4. Re-run with refined prompt
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-code-refined" \
  --prompt "$new_prompt" \
  --where "query_text LIKE '%code%' OR query_text LIKE '%program%'"
```

**Pro Tips:**
- Start with small `--limit` to validate prompt quality (~100-1000 queries)
- Use `--where` clause to focus on relevant subsets
- Provide 3-5 few-shot examples minimum for classification tasks
- XML-aware prompts enable metadata-conditional logic
- Always compare clustering quality: raw vs summarized
- Chain transformations for multi-dimensional analysis
- **Use properties for multi-dimensional extraction** - extract 5-10 features in ONE pass instead of multiple summarization runs

#### Prompt-Driven Property Extraction

The `properties` field enables extracting arbitrary typed features by specifying them in your prompt:

```bash
# Extract multiple dimensions for rich analysis
uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-enriched" \
  --prompt "
Analyze and extract:
- summary: One-sentence intent (str)
- user_frustration: 0-10 scale (int)
- complexity: 1-5 scale (int) 
- category: code_gen, debugging, creative, etc (str)
- requires_code: Expects code in response? (bool)
- language_tech: Programming language if relevant (str or null)

Query: {query}
" \
  --limit 10000
```

**Property types automatically inferred:**
- `(int)` → stored as integer
- `(float)` → stored as float  
- `(bool)` → stored as boolean
- `(str)` → stored as string
- Add `or null` to make optional

**Query with properties:**

```sql
-- High-priority support: frustrated + complex queries
SELECT query_text
FROM queries
WHERE CAST(extra_metadata->>'user_frustration' AS INT) >= 7
  AND CAST(extra_metadata->>'complexity' AS INT) >= 4;

-- Segment analysis: category × complexity
SELECT 
    extra_metadata->>'category' as cat,
    AVG(CAST(extra_metadata->>'complexity' AS REAL)) as avg_complexity,
    COUNT(*) as count
FROM queries
WHERE dataset_id = 2
GROUP BY cat;
```

**Benefits:**
- **One pass instead of many**: Extract 10 features in one summarization run
- **No code changes**: Add new properties just by changing the prompt
- **Flexible types**: int, float, bool, str automatically handled
- **SQL-queryable**: Use SQLite JSON operators for filtering and aggregation
```

### Examples Section Updates

Add to the existing CLAUDE.md examples section:

**Example 4: Autonomous Jailbreak Detection (Row Summarization)**

```
User: "Analyze the dataset for safety issues"

Claude:
[Runs initial clustering, identifies potential jailbreak patterns in cluster inspection]

I've identified potential jailbreak attempts mixed into several clusters. Let me create a 
dedicated classification to quantify this:

[Creates few-shot examples by inspecting suspicious queries]

uv run lmsys summarize "lmsys-1m" \
  --output "lmsys-safety-classified" \
  --prompt "Classify safety: safe, jailbreak_roleplay, jailbreak_injection, harmful_content, other\n{query}" \
  --example "12345:safe" \
  --example "67890:jailbreak_roleplay" \
  --example "11111:jailbreak_injection" \
  --limit 20000 \
  --use-chroma

[Clusters the classifications]

uv run lmsys cluster kmeans --dataset "lmsys-safety-classified" --n-clusters 6 --use-chroma

[Analyzes cluster distribution with SQL]

Results:
- 82.3% safe queries
- 11.2% jailbreak attempts (roleplay-based)
- 4.1% prompt injection attempts
- 2.4% other safety concerns

Recommendation: Implement roleplay detection as highest-impact safety improvement (11.2% of traffic).
See queries in cluster 3 for specific patterns.

[Provides specific examples and implementation suggestions]
```

**Example 5: Multi-Stage Intent Classification**

```
User: "I want to understand what people use the chatbot for"

Claude:
Let me create a multi-stage classification pipeline:

Stage 1: Broad domain classification
uv run lmsys summarize "lmsys-1m" --output "domains" \
  --prompt "Domain: tech, creative, education, business, personal: {query}" --limit 10000

Stage 2: Tech-specific intent extraction  
uv run lmsys summarize "domains" --output "tech-intent" \
  --prompt "Tech intent: code_gen, debugging, explanation, setup_help: {query}" \
  --where "query_text = 'tech'"

Stage 3: Cluster by programming language
uv run lmsys summarize "tech-intent" --output "tech-languages" \
  --prompt "Primary language: python, javascript, java, cpp, other: {query}" \
  --where "query_text = 'code_gen' OR query_text = 'debugging'"

[Analyzes final distribution]

Findings:
1. 67% tech queries (code_gen=45%, debugging=30%, explanation=15%, setup=10%)
2. Python dominates code_gen (62%), JavaScript leads debugging (48%)  
3. Creative writing is 18% (story generation, roleplay, poetry)
4. Education is 12% (homework help, concept explanations)

[Provides detailed breakdown with SQL queries and examples]
```

## Implementation Checklist

- [ ] Schema changes in `models.py`
  - [ ] Add `Prompt` table
  - [ ] Update `Dataset` with foreign keys (add `root_dataset_id`)
  - [ ] Update `Query` unique constraint to `(dataset_id, conversation_id)`
  - [ ] Create database migration script

- [ ] Core summarization logic
  - [ ] Implement `RowSummarizer` class
  - [ ] Add `serialize_query_to_xml()` function
  - [ ] Add `format_examples_xml()` function
  - [ ] Add prompt template validation (`{query}` required)
  - [ ] Implement async batch processing with Query objects
  - [ ] Add retry logic with exponential backoff

- [ ] CLI command
  - [ ] Implement `summarize` command
  - [ ] Add `--where` clause filtering
  - [ ] Add `--examples` file support (JSONL with query_id)
  - [ ] Add `--example` inline support (query_id:output)
  - [ ] Add auto-timestamp for existing outputs
  - [ ] Integrate with ChromaDB for embeddings
  - [ ] Add progress bars and logging

- [ ] Testing
  - [ ] Unit tests for `RowSummarizer`
  - [ ] Unit tests for XML serialization
  - [ ] Unit tests for schema changes (root_dataset_id, unique constraint)
  - [ ] Integration test for full command with examples
  - [ ] Create `smoketest-summarize.sh`
  - [ ] Update main smoke test

- [ ] Documentation
  - [ ] Update README with summarization workflow
  - [ ] **Add "Phase 4.5: Dataset Transformation" section to CLAUDE.md**
  - [ ] **Add autonomous summarization examples to CLAUDE.md**
  - [ ] Create prompt template guide (XML structure, metadata fields)
  - [ ] Document SQL queries for analysis
  - [ ] Add few-shot learning best practices

- [ ] Future enhancements
  - [ ] Prompt generation command (`lmsys prompts generate`)
  - [ ] Prompt library management (`lmsys prompts list/export/import`)
  - [ ] Dry-run mode for cost estimation
  - [ ] Validation sampling (auto-check N summaries)

---

## Conclusion

This implementation enables **autonomous, hypothesis-driven data analysis** by giving Claude the ability to:

1. **Create custom views** of datasets through LLM summarization
2. **Test behavioral hypotheses** via classification prompts
3. **Iteratively refine** prompts based on clustering results
4. **Extract structured features** for enrichment
5. **Validate findings** with SQL queries and cluster inspection

The key innovation is treating **summarization as a programmable transformation** rather than a fixed preprocessing step. This unlocks a new category of autonomous agent behaviors where Claude can explore data through self-directed prompt engineering.
