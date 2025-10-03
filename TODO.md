# TODO

## High Priority

### Core Features
- [ ] **Implement `lmsys list` command** - List queries with filtering by run_id and cluster_id
- [ ] **Implement `lmsys inspect` command** - Detailed view of specific cluster with all queries
- [ ] **Implement `lmsys export` command** - Export clustering results to CSV/JSON
- [ ] **Add database migration support** - Handle schema changes gracefully (alembic)
- [ ] **Parallel LLM summarization** - Batch API calls with async/threading for faster processing
- [ ] **HDBSCAN clustering support** - Alternative to KMeans for density-based clustering

### Performance
- [ ] **Caching for embeddings** - Store embeddings in DB to avoid regenerating
- [ ] **Incremental data loading** - Resume interrupted loads without reprocessing
- [ ] **Batch size optimization** - Tune batch sizes for embedding generation
- [ ] **Index optimization** - Add composite indexes for common query patterns

### User Experience
- [ ] **Progress bars for LLM calls** - Show progress during batch summarization
- [ ] **Cluster visualization** - Generate t-SNE/UMAP plots for cluster exploration
- [ ] **Interactive TUI mode** - Browse clusters and queries interactively
- [ ] **Query similarity within cluster** - Find most representative/diverse queries

## Medium Priority

### Data Quality
- [ ] **Query deduplication** - Identify and handle duplicate queries
- [ ] **Language filtering** - Filter queries by detected language
- [ ] **Length filtering** - Filter very short/long queries
- [ ] **Quality scoring** - Score queries by coherence/completeness

### Analysis Features
- [ ] **Cluster merging** - Merge similar clusters based on embeddings
- [ ] **Hierarchical clustering** - Build cluster hierarchy for drill-down
- [ ] **Temporal analysis** - Analyze query trends over time
- [ ] **Model comparison** - Compare queries across different LLMs

### Search Enhancements
- [ ] **Hybrid search** - Combine semantic + keyword search
- [ ] **Query expansion** - Suggest related search terms
- [ ] **Filter by metadata** - Search with model/language/date filters
- [ ] **Save searches** - Store common queries for reuse

### LLM Integration
- [ ] **Streaming responses** - Stream LLM outputs for better UX
- [ ] **Custom prompts** - Allow users to customize summarization prompts
- [ ] **Multi-model comparison** - Generate summaries with multiple LLMs
- [ ] **Summary refinement** - Iteratively improve summaries based on feedback

## Low Priority

### Documentation
- [ ] **API documentation** - Generate API docs from docstrings
- [ ] **Tutorial notebooks** - Jupyter notebooks with walkthroughs
- [ ] **Video demo** - Record demo video for README
- [ ] **Blog post** - Write about architecture and use cases

### DevOps
- [ ] **Docker support** - Containerize the application
- [ ] **CI/CD pipeline** - Automated testing and releases
- [ ] **Pre-commit hooks** - Linting and formatting checks
- [ ] **Benchmarking suite** - Performance regression tests

### Web Interface
- [ ] **Web dashboard** - Flask/FastAPI app for browser access
- [ ] **REST API** - HTTP API for programmatic access
- [ ] **Real-time updates** - WebSocket for live progress
- [ ] **User authentication** - Multi-user support

### Advanced Features
- [ ] **Active learning** - Suggest clusters for manual review
- [ ] **Query generation** - Generate synthetic queries for testing
- [ ] **Cluster evolution tracking** - Track how clusters change with new data
- [ ] **Anomaly detection** - Find unusual queries in clusters

## Known Issues

### Bugs
- [ ] **ChromaDB collection size limits** - May hit limits with 1M queries
- [ ] **Memory usage with large datasets** - OOM with full 1M dataset
- [ ] **Schema migration required** - title/description columns need migration
- [ ] **Model name validation** - instructor model format not validated

### Limitations
- [ ] **No incremental clustering** - Must recluster all data
- [ ] **Single-node only** - No distributed processing
- [ ] **English-only summaries** - LLM summaries assume English
- [ ] **API rate limits** - No rate limiting for LLM providers

## Ideas / Future Exploration

- [ ] **Fine-tune embedding model** - Train on LMSYS data for better clusters
- [ ] **Graph-based clustering** - Use query similarity graphs
- [ ] **Topic modeling** - LDA/BERTopic for unsupervised topics
- [ ] **Query intent classification** - Classify queries by intent type
- [ ] **Multi-turn analysis** - Analyze full conversations, not just first query
- [ ] **Cross-dataset comparison** - Compare LMSYS with other datasets
- [ ] **Prompt engineering analysis** - Study effective prompt patterns
- [ ] **Model performance correlation** - Link clusters to model capabilities

## Dependencies to Consider

- [ ] **DuckDB** - For faster analytical queries
- [ ] **Polars** - For faster DataFrame operations
- [ ] **Ray** - For distributed processing
- [ ] **Dagster/Prefect** - For workflow orchestration
- [ ] **Weights & Biases** - For experiment tracking
- [ ] **Streamlit** - For quick prototyping dashboards

## Testing Gaps

- [ ] Integration tests for full workflow
- [ ] Tests for ChromaDB operations
- [ ] Tests for LLM summarization (mock)
- [ ] Performance benchmarks
- [ ] Edge cases (empty clusters, single query, etc.)
- [ ] Large dataset stress tests

---

**Notes:**
- Items marked with [ ] are not yet implemented
- Priority is subjective and may change based on user feedback
- Some items may be split into smaller tasks
- Cross off items as they're completed with [x]
