# LMSYS Query Analysis

A comprehensive tool for analyzing the LMSYS-1M dataset through clustering and semantic search. This system enables researchers and developers to understand user query patterns, identify common use cases, and explore the conversational AI landscape.

## System Overview

```mermaid
graph TB
    A[LMSYS-1M Dataset] --> B[Data Loading]
    B --> C[SQLite Database]
    B --> D[ChromaDB Vectors]
    
    C --> E[Query Storage]
    C --> F[Clustering Runs]
    C --> G[Cluster Assignments]
    C --> H[LLM Summaries]
    
    D --> I[Query Embeddings]
    D --> J[Cluster Centroids]
    D --> K[Summary Embeddings]
    
    E --> L[Clustering Pipeline]
    I --> L
    L --> M[KMeans Clustering]
    L --> N[HDBSCAN Clustering]
    
    M --> O[Cluster Results]
    N --> O
    
    O --> P[LLM Summarization]
    P --> Q[Cluster Titles & Descriptions]
    
    Q --> R[Analysis & Search]
    I --> R
    K --> R
    
    R --> S[Semantic Query Search]
    R --> T[Cluster Exploration]
    R --> U[Data Export]
```

## Module Architecture

```mermaid
graph TB
    A[CLI Module] --> B[Database Module]
    A --> C[Clustering Module]
    
    B --> D[SQLite Storage]
    B --> E[ChromaDB Vectors]
    
    C --> F[Embedding Generation]
    C --> G[Clustering Algorithms]
    C --> H[LLM Summarization]
    
    D --> I[Query Models]
    D --> J[Run Tracking]
    D --> K[Assignment Storage]
    D --> L[Summary Storage]
    
    E --> M[Query Embeddings]
    E --> N[Cluster Centroids]
    E --> O[Summary Embeddings]
    
    F --> P[Sentence Transformers]
    F --> Q[OpenAI API]
    
    G --> R[KMeans]
    G --> S[HDBSCAN]
    
    H --> T[Anthropic Claude]
    H --> U[OpenAI GPT]
    H --> V[Groq LLaMA]
```

## Data Flow

### 1. Data Loading Phase
```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Loader
    participant HF
    participant SQLite
    participant ChromaDB
    
    User->>CLI: lmsys load --use-chroma
    CLI->>Loader: load_lmsys_dataset()
    Loader->>HF: Download LMSYS-1M
    HF-->>Loader: Dataset stream
    
    loop For each conversation
        Loader->>SQLite: Store query metadata
        Loader->>ChromaDB: Store query embedding
    end
    
    Loader-->>CLI: Loading statistics
    CLI-->>User: Results table
```

### 2. Clustering Phase
```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Clustering
    participant SQLite
    participant ChromaDB
    
    User->>CLI: lmsys cluster kmeans
    CLI->>Clustering: run_kmeans_clustering()
    
    Clustering->>SQLite: Create run record
    Clustering->>ChromaDB: Get existing embeddings
    
    loop Streaming chunks
        Clustering->>Clustering: Generate embeddings
        Clustering->>Clustering: Partial fit KMeans
    end
    
    loop Prediction phase
        Clustering->>Clustering: Predict cluster labels
        Clustering->>SQLite: Store assignments
        Clustering->>ChromaDB: Store centroids
    end
    
    Clustering-->>CLI: Run ID
    CLI-->>User: Cluster statistics
```

### 3. Summarization Phase
```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Summarizer
    participant SQLite
    participant LLM
    participant ChromaDB
    
    User->>CLI: lmsys summarize run_id
    CLI->>Summarizer: generate_batch_summaries()
    
    loop For each cluster
        Summarizer->>SQLite: Get cluster queries
        Summarizer->>LLM: Generate title/description
        LLM-->>Summarizer: Structured response
        Summarizer->>SQLite: Store summary
        Summarizer->>ChromaDB: Update embeddings
    end
    
    Summarizer-->>CLI: Complete
    CLI-->>User: Summary generation complete
```

### 4. Analysis Phase
```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Search
    participant SQLite
    participant ChromaDB
    
    User->>CLI: lmsys search "python help"
    
    alt Query search
        CLI->>ChromaDB: search_queries()
        ChromaDB-->>CLI: Similar queries
        CLI->>SQLite: Get full details
        SQLite-->>CLI: Query metadata
    else Cluster search
        CLI->>ChromaDB: search_cluster_summaries()
        ChromaDB-->>CLI: Relevant clusters
        CLI->>SQLite: Get cluster info
        SQLite-->>CLI: Cluster details
    end
    
    CLI-->>User: Formatted results
```

## Key Features

### Data Management
- **Flexible Loading**: Configurable limits and filtering
- **Incremental Updates**: Skip existing data, resume loading
- **Metadata Preservation**: All original conversation data retained
- **Progress Tracking**: Real-time loading statistics

### Clustering Capabilities
- **Multiple Algorithms**: KMeans and HDBSCAN support
- **Scalable Processing**: Memory-efficient streaming for large datasets
- **Embedding Flexibility**: Local models or cloud APIs
- **Run Tracking**: Complete experiment history

### Analysis Tools
- **Semantic Search**: Find similar queries or clusters
- **Cluster Exploration**: Detailed inspection of query groups
- **Export Options**: CSV and JSON output formats
- **Rich Display**: Beautiful terminal tables and progress bars

### LLM Integration
- **Multi-Provider**: Anthropic, OpenAI, and Groq support
- **Structured Output**: Reliable parsing with Pydantic models
- **Batch Processing**: Efficient summary generation
- **Error Handling**: Graceful fallbacks for API failures

## Usage Patterns

### Research Workflow
1. **Load Dataset**: `lmsys load --limit 50000 --use-chroma`
2. **Cluster Analysis**: `lmsys cluster kmeans --n-clusters 500`
3. **Generate Summaries**: `lmsys summarize run_id --use-chroma`
4. **Explore Results**: `lmsys list-clusters run_id`
5. **Search Patterns**: `lmsys search "programming help"`
6. **Export Data**: `lmsys export run_id --output analysis.csv`

### Development Workflow
1. **Quick Testing**: Small dataset loads for algorithm testing
2. **Parameter Tuning**: Multiple clustering runs with different parameters
3. **Quality Assessment**: Manual inspection of cluster quality
4. **Iterative Improvement**: Refine algorithms based on results

## Performance Characteristics

### Scalability
- **Memory Efficient**: Streaming processing for datasets >1M queries
- **Parallel Processing**: Multi-core utilization for clustering
- **Batch Operations**: Optimized database and API operations
- **Caching**: Embedding reuse prevents redundant computation

### Storage Requirements
- **SQLite**: ~100MB per 10K queries (metadata only)
- **ChromaDB**: ~500MB per 10K queries (with embeddings)
- **Total**: ~600MB per 10K queries for full functionality

### Processing Times
- **Loading**: ~1000 queries/second
- **Embedding**: ~500 queries/second (local models)
- **Clustering**: ~10K queries/minute (KMeans)
- **Summarization**: ~50 clusters/minute (depends on LLM provider)

## Integration Points

### External Dependencies
- **HuggingFace**: Dataset loading and model hosting
- **SQLModel**: Database ORM and migrations
- **ChromaDB**: Vector storage and similarity search
- **Rich**: Terminal UI and progress display
- **Typer**: CLI framework and argument parsing

### API Integrations
- **OpenAI**: Embedding generation and LLM summarization
- **Anthropic**: LLM summarization
- **Groq**: Fast LLM inference for summarization
- **Sentence Transformers**: Local embedding models

This architecture provides a robust foundation for large-scale query analysis while maintaining flexibility for different research and development needs.
