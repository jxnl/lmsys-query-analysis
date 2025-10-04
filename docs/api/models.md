# Database Models

The project uses SQLModel for database schema definition and ORM functionality.

## Query

Represents a single user query from the LMSYS-1M dataset.

### Fields

- **id** (int): Auto-incremented primary key
- **conversation_id** (str): Unique identifier from LMSYS dataset
- **model** (str): LLM model used in conversation
- **query_text** (str): First user message in conversation
- **language** (str | None): Detected language of query
- **extra_metadata** (dict | None): Additional JSON metadata
- **created_at** (datetime): Timestamp of record creation

### Example

```python
from lmsys_query_analysis.db.models import Query

query = Query(
    conversation_id="conv_12345",
    model="gpt-3.5-turbo",
    query_text="How do I center a div in CSS?",
    language="en"
)
```

---

## ClusteringRun

Tracks clustering experiments and their parameters.

### Fields

- **run_id** (str): Unique identifier (e.g., "kmeans-100-20251003-123456")
- **algorithm** (str): Clustering algorithm ("kmeans" or "hdbscan")
- **num_clusters** (int | None): Number of clusters (KMeans only)
- **parameters** (dict | None): Algorithm parameters as JSON
- **description** (str | None): Optional user description
- **created_at** (datetime): Timestamp of run creation

### Example

```python
from lmsys_query_analysis.db.models import ClusteringRun

run = ClusteringRun(
    run_id="kmeans-100-20251003-123456",
    algorithm="kmeans",
    num_clusters=100,
    parameters={"batch_size": 4096},
    description="Fine-grained clustering"
)
```

---

## QueryCluster

Maps queries to clusters within a specific run.

### Fields

- **id** (int): Auto-incremented primary key
- **run_id** (str): Foreign key to ClusteringRun
- **query_id** (int): Foreign key to Query
- **cluster_id** (int): Cluster assignment
- **confidence_score** (float | None): Confidence or distance score

### Example

```python
from lmsys_query_analysis.db.models import QueryCluster

assignment = QueryCluster(
    run_id="kmeans-100-20251003-123456",
    query_id=42,
    cluster_id=5,
    confidence_score=0.85
)
```

---

## ClusterSummary

LLM-generated summaries for clusters.

### Fields

- **id** (int): Auto-incremented primary key
- **run_id** (str): Foreign key to ClusteringRun
- **cluster_id** (int): Cluster number
- **title** (str | None): Short cluster title
- **description** (str | None): Detailed description
- **num_queries** (int): Count of queries in cluster
- **representative_queries** (list | None): Sample queries as JSON
- **created_at** (datetime): Timestamp of summary creation

### Example

```python
from lmsys_query_analysis.db.models import ClusterSummary

summary = ClusterSummary(
    run_id="kmeans-100-20251003-123456",
    cluster_id=5,
    title="CSS Layout Questions",
    description="Queries about CSS positioning, flexbox, and grid layouts",
    num_queries=150,
    representative_queries=["How to center a div?", "Flexbox vs Grid?"]
)
```

---

## Database Schema

```mermaid
erDiagram
    Query ||--o{ QueryCluster : "assigned to"
    ClusteringRun ||--o{ QueryCluster : "contains"
    ClusteringRun ||--o{ ClusterSummary : "has"

    Query {
        int id PK
        string conversation_id UK
        string model
        string query_text
        string language
        json extra_metadata
        datetime created_at
    }

    ClusteringRun {
        string run_id PK
        string algorithm
        int num_clusters
        json parameters
        string description
        datetime created_at
    }

    QueryCluster {
        int id PK
        string run_id FK
        int query_id FK
        int cluster_id
        float confidence_score
    }

    ClusterSummary {
        int id PK
        string run_id FK
        int cluster_id
        string title
        string description
        int num_queries
        json representative_queries
        datetime created_at
    }
```

## Usage

### Creating a Database Session

```python
from lmsys_query_analysis.db.connection import DatabaseManager

db = DatabaseManager("~/.lmsys-query-analysis/queries.db")
session = db.get_session()

# Query data
from lmsys_query_analysis.db.models import Query
from sqlmodel import select

statement = select(Query).limit(10)
queries = session.exec(statement).all()
```

### Querying Clusters

```python
from lmsys_query_analysis.db.models import ClusterSummary
from sqlmodel import select

# Get all summaries for a run
statement = select(ClusterSummary).where(
    ClusterSummary.run_id == "kmeans-100-20251003-123456"
)
summaries = session.exec(statement).all()

# Get clusters sorted by size
statement = select(ClusterSummary).where(
    ClusterSummary.run_id == "kmeans-100-20251003-123456"
).order_by(ClusterSummary.num_queries.desc())
top_clusters = session.exec(statement).all()
```

## Next Steps

- [CLI Reference](../cli/overview.md)
- [Clustering API](clustering.md)
