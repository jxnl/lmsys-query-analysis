# Clustering Module

The clustering module provides embedding generation and clustering algorithms for analyzing LMSYS query patterns. It supports both KMeans and HDBSCAN clustering with optional LLM-powered summarization.

## Overview

This module enables semantic analysis of user queries through:
1. **Embedding Generation**: Convert text to vector representations
2. **Clustering Algorithms**: Group similar queries
3. **LLM Summarization**: Generate human-readable cluster descriptions

## Architecture

```mermaid
graph TB
    A[Query Texts] --> B[Embedding Generator]
    B --> C[Vector Embeddings]
    C --> D{Clustering Algorithm}
    
    D --> E[KMeans Clustering]
    D --> F[HDBSCAN Clustering]
    
    E --> G[Fixed Number Clusters]
    F --> H[Dynamic Clusters + Noise]
    
    G --> I[Cluster Assignments]
    H --> I
    
    I --> J[Cluster Summarizer]
    J --> K[LLM Provider]
    K --> L[Cluster Titles & Descriptions]
    
    B --> M[Sentence Transformers]
    B --> N[OpenAI API]
    
    K --> O[Anthropic Claude]
    K --> P[OpenAI GPT]
    K --> Q[Groq LLaMA]
```

## Core Components

### Embedding Generator (embeddings.py)

**Purpose**: Generate vector embeddings from text using different providers

**Supported Providers**:
```mermaid
graph LR
    A[EmbeddingGenerator] --> B[Sentence Transformers]
    A --> C[OpenAI API]
    
    B --> D[all-MiniLM-L6-v2]
    B --> E[all-mpnet-base-v2]
    B --> F[Other Models]
    
    C --> G[text-embedding-3-small]
    C --> H[text-embedding-3-large]
    C --> I[text-embedding-ada-002]
```

**Key Features**:
- **Batch Processing**: Efficient encoding of multiple texts
- **Progress Tracking**: Rich progress bars for long operations
- **Model Caching**: Lazy loading and reuse of models
- **Performance Metrics**: Timing and throughput reporting

**Process Flow**:
```mermaid
graph TB
    A[Text Input] --> B{Provider}
    B -->|sentence-transformers| C[Load Model]
    B -->|openai| D[API Client]
    
    C --> E[Encode Batch]
    D --> F[API Call]
    
    E --> G[NumPy Array]
    F --> G
    
    G --> H[Return Embeddings]
```

### KMeans Clustering (kmeans.py)

**Purpose**: MiniBatchKMeans clustering with streaming embeddings for large datasets

**Algorithm Flow**:
```mermaid
graph TB
    A[Start] --> B[Initialize MiniBatchKMeans]
    B --> C[Create Run Record]
    C --> D[Stream Queries in Chunks]
    
    D --> E{ChromaDB Available?}
    E -->|Yes| F[Get Existing Embeddings]
    E -->|No| G[Generate Embeddings]
    
    F --> H{All Found?}
    H -->|No| I[Generate Missing]
    H -->|Yes| J[Use Existing]
    G --> K[Store in ChromaDB]
    I --> K
    J --> L[Partial Fit]
    K --> L
    
    L --> M{More Chunks?}
    M -->|Yes| D
    M -->|No| N[Predict Labels]
    
    N --> O[Store Assignments]
    O --> P[Compute Statistics]
    P --> Q[Store Centroids]
    Q --> R[Complete]
```

**Key Features**:
- **Streaming Processing**: Handles datasets larger than memory
- **Embedding Reuse**: Leverages ChromaDB to avoid recomputation
- **Incremental Learning**: MiniBatchKMeans for efficiency
- **Statistics Tracking**: Detailed cluster size analysis

### HDBSCAN Clustering (hdbscan_clustering.py)

**Purpose**: Density-based clustering that finds natural clusters and handles noise

**Algorithm Flow**:
```mermaid
graph TB
    A[Start] --> B[Load All Embeddings]
    B --> C[Initialize HDBSCAN]
    C --> D[Set Parameters]
    
    D --> E[min_cluster_size]
    D --> F[min_samples]
    D --> G[cluster_selection_epsilon]
    D --> H[metric]
    
    E --> I[Fit & Predict]
    F --> I
    G --> I
    H --> I
    
    I --> J[Cluster Labels]
    J --> K[Compute Centroids]
    K --> L[Store Assignments]
    L --> M[Store Centroids]
    M --> N[Report Statistics]
    N --> O[Complete]
```

**Key Features**:
- **Noise Detection**: Identifies outliers (label -1)
- **Variable Clusters**: Number of clusters determined by data
- **Persistence Analysis**: Measures cluster stability
- **Centroid Computation**: For cluster representation

**Parameters**:
- `min_cluster_size`: Minimum queries per cluster
- `min_samples`: Minimum neighbors for core points
- `cluster_selection_epsilon`: Distance threshold for merging
- `metric`: Distance metric (euclidean/cosine)

### Cluster Summarizer (summarizer.py)

**Purpose**: Generate human-readable titles and descriptions using LLMs

**LLM Integration**:
```mermaid
graph TB
    A[Cluster Queries] --> B[Sample Queries]
    B --> C[Build Prompt]
    C --> D[LLM Provider]
    
    D --> E[Anthropic Claude]
    D --> F[OpenAI GPT]
    D --> G[Groq LLaMA]
    
    E --> H[Structured Response]
    F --> H
    G --> H
    
    H --> I[Title]
    H --> J[Description]
    H --> K[Sample Queries]
```

**Key Features**:
- **Structured Output**: Uses Pydantic models for reliable parsing
- **Query Sampling**: Intelligent sampling for large clusters
- **Multiple Providers**: Support for various LLM APIs
- **Error Handling**: Fallback summaries for failed requests

**Prompt Structure**:
```
Analyze these queries and provide:
1. A SHORT TITLE (5-10 words) that captures the main topic
2. A DESCRIPTION (2-3 sentences) explaining what types of questions/requests are in this cluster
```

## Data Flow

### Complete Clustering Pipeline
```mermaid
sequenceDiagram
    participant CLI
    participant EmbeddingGen
    participant Clustering
    participant Database
    participant ChromaDB
    participant LLM
    
    CLI->>Clustering: run_clustering()
    Clustering->>Database: Create run record
    Clustering->>EmbeddingGen: Generate embeddings
    EmbeddingGen-->>Clustering: Vector array
    Clustering->>Clustering: Fit clustering model
    Clustering->>Database: Store assignments
    Clustering->>ChromaDB: Store centroids
    Clustering-->>CLI: Run ID
    
    CLI->>LLM: generate_summaries()
    LLM->>Database: Get cluster queries
    LLM->>LLM: Generate titles/descriptions
    LLM->>Database: Store summaries
    LLM->>ChromaDB: Update summaries
    LLM-->>CLI: Complete
```

### Embedding Generation Process
```mermaid
sequenceDiagram
    participant Clustering
    participant ChromaDB
    participant EmbeddingGen
    participant Model
    
    Clustering->>ChromaDB: get_query_embeddings_map()
    ChromaDB-->>Clustering: Existing embeddings
    
    alt Missing embeddings
        Clustering->>EmbeddingGen: generate_embeddings()
        EmbeddingGen->>Model: encode()
        Model-->>EmbeddingGen: Vectors
        EmbeddingGen-->>Clustering: NumPy array
        Clustering->>ChromaDB: Store new embeddings
    end
```

## Performance Optimizations

### Memory Management
- **Chunked Processing**: Large datasets processed in configurable chunks
- **Streaming Embeddings**: KMeans uses partial_fit for memory efficiency
- **Batch Operations**: Embeddings generated in batches

### Computational Efficiency
- **Embedding Caching**: ChromaDB prevents redundant computations
- **Parallel Processing**: HDBSCAN uses all CPU cores
- **MiniBatch Algorithm**: Faster convergence than full KMeans

### I/O Optimization
- **Batch Database Writes**: Reduces transaction overhead
- **Progress Tracking**: User feedback for long operations
- **Error Recovery**: Graceful handling of API failures

## Configuration Options

### Embedding Models
- **Sentence Transformers**: Local models, no API costs
- **OpenAI**: Cloud-based, higher quality embeddings
- **Batch Sizes**: Configurable for memory/performance tradeoffs

### Clustering Parameters
- **KMeans**: Number of clusters, random seed, batch size
- **HDBSCAN**: Minimum cluster size, epsilon, distance metric
- **Memory Limits**: Chunk sizes for large datasets

### LLM Integration
- **Provider Selection**: Multiple API providers supported
- **Model Choice**: Different models for cost/quality tradeoffs
- **Rate Limiting**: Built-in handling of API limits
