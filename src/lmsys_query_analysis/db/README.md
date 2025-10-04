# Database Module

The database module handles data persistence using SQLModel (SQLAlchemy) with SQLite and provides ChromaDB integration for vector storage and semantic search.

## Overview

This module provides two storage layers:
1. **SQLite**: Relational storage for queries, clustering runs, and metadata
2. **ChromaDB**: Vector database for embeddings and semantic search

## Architecture

```mermaid
graph TB
    A[CLI Commands] --> B[Database Module]
    
    B --> C[Database Class]
    B --> D[ChromaManager Class]
    
    C --> E[(SQLite Database)]
    D --> F[(ChromaDB)]
    
    E --> G[Query Table]
    E --> H[ClusteringRun Table]
    E --> I[QueryCluster Table]
    E --> J[ClusterSummary Table]
    
    F --> K[Queries Collection]
    F --> L[Cluster Summaries Collection]
    
    G --> M[Query Text + Metadata]
    K --> N[Query Embeddings]
    
    H --> O[Run Parameters]
    I --> P[Query-Cluster Mappings]
    J --> Q[LLM Summaries]
    L --> R[Summary Embeddings]
```

## Core Components

### Database Models (models.py)

```mermaid
erDiagram
    Query ||--o{ QueryCluster : "has many"
    ClusteringRun ||--o{ QueryCluster : "has many"
    ClusteringRun ||--o{ ClusterSummary : "has many"
    
    Query {
        int id PK
        string conversation_id UK
        string model
        string query_text
        string language
        datetime timestamp
        json extra_metadata
        datetime created_at
    }
    
    ClusteringRun {
        string run_id PK
        string algorithm
        json parameters
        string description
        datetime created_at
        int num_clusters
    }
    
    QueryCluster {
        int id PK
        string run_id FK
        int query_id FK
        int cluster_id
        float confidence_score
        datetime created_at
    }
    
    ClusterSummary {
        int id PK
        string run_id FK
        int cluster_id
        string title
        string description
        string summary
        int num_queries
        json representative_queries
        datetime generated_at
    }
```

### Database Connection (connection.py)

**Purpose**: Manages SQLite database connections and sessions

**Key Features**:
- Automatic table creation
- Foreign key enforcement
- Session management
- Default path configuration

**Flow**:
```mermaid
graph LR
    A[get_db] --> B[Database Instance]
    B --> C[Create Engine]
    C --> D[Enable Foreign Keys]
    D --> E[Session Management]
    E --> F[CRUD Operations]
```

### Data Loader (loader.py)

**Purpose**: Downloads and loads LMSYS-1M dataset from HuggingFace

**Process Flow**:
```mermaid
graph TB
    A[load_lmsys_dataset] --> B[Download Dataset]
    B --> C[Extract First Query]
    C --> D{Skip Existing?}
    D -->|Yes| E[Check Database]
    D -->|No| F[Create Query Record]
    E --> G{Exists?}
    G -->|Yes| H[Skip]
    G -->|No| F
    F --> I[Batch Commit]
    I --> J{ChromaDB Enabled?}
    J -->|Yes| K[Generate Embeddings]
    J -->|No| L[Complete]
    K --> M[Store in ChromaDB]
    M --> L
```

**Key Functions**:
- `extract_first_query()`: Extracts first user message from conversation
- `load_lmsys_dataset()`: Main loading function with progress tracking
- Batch processing for performance
- Optional ChromaDB integration

### ChromaDB Manager (chroma.py)

**Purpose**: Manages vector storage and semantic search capabilities

**Collections**:
```mermaid
graph TB
    A[ChromaManager] --> B[Queries Collection]
    A --> C[Cluster Summaries Collection]
    
    B --> D[Query Embeddings]
    B --> E[Query Metadata]
    
    C --> F[Summary Embeddings]
    C --> G[Run-Specific Metadata]
    
    D --> H[Semantic Search]
    F --> I[Cluster Search]
```

**Key Features**:
- **Batch Operations**: Efficient bulk inserts
- **Metadata Enrichment**: Adds SQLite IDs and run information
- **Search Capabilities**: Semantic search across queries and summaries
- **Run Isolation**: Cluster summaries filtered by run_id

**Search Methods**:
- `search_queries()`: Find similar queries
- `search_cluster_summaries()`: Find relevant clusters
- `get_query_embeddings_map()`: Retrieve embeddings for reuse

## Data Flow

### Loading Process
```mermaid
sequenceDiagram
    participant CLI
    participant Loader
    participant Database
    participant ChromaDB
    participant HF
    
    CLI->>Loader: load_lmsys_dataset()
    Loader->>HF: Download dataset
    HF-->>Loader: Dataset chunks
    loop For each conversation
        Loader->>Database: Check if exists
        Database-->>Loader: Exists status
        alt Not exists
            Loader->>Database: Insert query
            Loader->>ChromaDB: Store embedding
        end
    end
    Loader-->>CLI: Loading statistics
```

### Clustering Process
```mermaid
sequenceDiagram
    participant CLI
    participant Clustering
    participant Database
    participant ChromaDB
    
    CLI->>Clustering: run_kmeans_clustering()
    Clustering->>Database: Create run record
    Clustering->>ChromaDB: Get existing embeddings
    Clustering->>Clustering: Generate embeddings
    Clustering->>Clustering: Fit clustering model
    Clustering->>Database: Store assignments
    Clustering->>ChromaDB: Store centroids
    Clustering-->>CLI: Run ID
```

### Search Process
```mermaid
sequenceDiagram
    participant CLI
    participant ChromaDB
    participant Database
    
    CLI->>ChromaDB: search_queries()
    ChromaDB->>ChromaDB: Compute query embedding
    ChromaDB->>ChromaDB: Vector similarity search
    ChromaDB-->>CLI: Similar queries
    CLI->>Database: Get full query details
    Database-->>CLI: Query metadata
```

## Key Design Decisions

1. **Dual Storage**: SQLite for structured data, ChromaDB for vectors
2. **Batch Processing**: Efficient handling of large datasets
3. **Embedding Reuse**: ChromaDB caches embeddings to avoid recomputation
4. **Run Isolation**: Each clustering experiment is isolated by run_id
5. **Metadata Preservation**: All original data preserved with clustering results

## Performance Considerations

- **Chunked Processing**: Large datasets processed in chunks
- **Batch Commits**: Database writes batched for efficiency
- **Embedding Caching**: ChromaDB prevents redundant embedding generation
- **Indexing**: Database indexes on frequently queried fields
- **Memory Management**: Streaming approach for large datasets
