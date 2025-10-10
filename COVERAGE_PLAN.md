# Test Coverage Improvement Plan

**Current Coverage: 50.27% (1509/3000 statements)**  
**Goal: 65-70% (Focus on testable business logic)**

---

## ✅ Already Well-Covered (>80%)
- ✅ `db/models.py` - 100%
- ✅ All services (94-100%)
- ✅ `semantic/types.py` - 100%
- ✅ `cli/main.py` - 96%
- ✅ `db/connection.py` - 96%
- ✅ `semantic/clusters.py` - 86%
- ✅ `summarizer.py` - 84%

---

## 🎯 High-Value Targets (Significant Impact)

### 1. **embeddings.py** - Currently 56% (89 missing lines)
**Missing Lines:** 70-83, 131, 162-180, 227, 229-238, 265-275, 283-286, 308-356, 365-409, 417-422, 425

#### Missing Coverage Areas:
- **Lines 70-83**: OpenAI client initialization with API key
- **Lines 162-180**: Cohere async embedding generation
- **Lines 229-238**: Cohere batch processing logic
- **Lines 265-275**: Cohere client setup and error handling
- **Lines 283-286**: OpenAI async error handling in nested loops
- **Lines 308-356**: Cohere async batch operations
- **Lines 365-409**: Full OpenAI async batch processing
- **Lines 417-422**: Rate limiting handling

#### Proposed Tests:
```python
# tests/test_embeddings_coverage.py
1. test_openai_embeddings_with_mock_api()
   - Mock OpenAI API responses
   - Test batch processing paths
   - Cover lines 70-83, 365-409

2. test_cohere_embeddings_with_mock_api()
   - Mock Cohere API responses
   - Test async generation path
   - Cover lines 162-180, 229-238, 265-275, 308-356

3. test_embedding_error_handling()
   - Test API failures
   - Test rate limiting scenarios
   - Cover lines 417-422

4. test_embedding_empty_batch_edge_cases()
   - Test with single item
   - Test with exactly batch_size items
   - Cover remaining edge cases
```

**Potential Improvement: 56% → 75% (+19%)**

---

### 2. **kmeans.py** - Currently 63% (61 missing lines)
**Missing Lines:** 59-62, 106, 127-173, 188, 223-261, 264, 275, 290-292, 321-350, 371-395

#### Missing Coverage Areas:
- **Lines 59-62**: MiniBatch KMeans initialization parameters
- **Lines 127-173**: Main clustering execution with embeddings
- **Lines 223-261**: Cluster assignment and confidence calculation
- **Lines 321-350**: Cluster size analysis and statistics
- **Lines 371-395**: Cluster centroids and quality metrics

#### Proposed Tests:
```python
# tests/test_kmeans_clustering.py
1. test_kmeans_full_pipeline_with_real_embeddings()
   - Create sample queries with embeddings
   - Run clustering end-to-end
   - Verify cluster assignments
   - Cover lines 127-173, 223-261

2. test_cluster_quality_metrics()
   - Test silhouette score calculation
   - Test inertia metrics
   - Cover lines 321-350, 371-395

3. test_minibatch_kmeans_parameters()
   - Test different batch sizes
   - Test n_init variations
   - Cover lines 59-62

4. test_cluster_confidence_scores()
   - Verify distance-based confidence
   - Test edge cases (equidistant points)
   - Cover confidence calculation paths
```

**Potential Improvement: 63% → 85% (+22%)**

---

### 3. **chroma.py** - Currently 61% (43 missing lines)
**Missing Lines:** 34, 37, 62, 159-179, 227-228, 270, 303, 319-320, 333-348, 360-361, 370-376, 380, 384-387, 411

#### Missing Coverage Areas:
- **Lines 159-179**: Query embedding updates
- **Lines 333-348**: Cluster summary storage
- **Lines 370-376**: Search with filters
- **Lines 384-387**: Collection metadata operations

#### Proposed Tests:
```python
# tests/test_chroma_integration.py
1. test_chroma_query_embedding_storage()
   - Add queries with embeddings
   - Update embeddings
   - Verify persistence
   - Cover lines 159-179

2. test_chroma_cluster_summary_operations()
   - Store cluster summaries
   - Retrieve by run_id
   - Update summaries
   - Cover lines 333-348

3. test_chroma_search_with_filters()
   - Search with cluster_id filter
   - Search with run_id filter
   - Test combined filters
   - Cover lines 370-376, 384-387

4. test_chroma_collection_metadata()
   - Test metadata retrieval
   - Test collection info
   - Cover lines 37, 62, 411
```

**Potential Improvement: 61% → 85% (+24%)**

---

### 4. **semantic/queries.py** - Currently 79% (33 missing lines)
**Missing Lines:** 39-64, 117-119, 152, 198, 207-214, 268-269, 290, 297, 306-307

#### Missing Coverage Areas:
- **Lines 39-64**: Query search with complex filters
- **Lines 207-214**: Cluster membership queries
- **Lines 268-269**: Language filtering
- **Lines 306-307**: Sample query selection

#### Proposed Tests:
```python
# tests/test_semantic_queries.py
1. test_query_search_with_filters()
   - Search by language
   - Search by model
   - Combine multiple filters
   - Cover lines 39-64, 268-269

2. test_cluster_membership_queries()
   - Get queries in cluster
   - Get queries not in cluster
   - Cover lines 207-214

3. test_sample_query_selection()
   - Sample with different strategies
   - Test edge cases (empty clusters)
   - Cover lines 290, 297, 306-307
```

**Potential Improvement: 79% → 95% (+16%)**

---

### 5. **config.py** - Currently 72% (15 missing lines)
**Missing Lines:** 67-73, 79-84, 113-116, 126-127

#### Missing Coverage Areas:
- **Lines 67-73**: ChromaDB configuration
- **Lines 79-84**: Embedding provider configuration
- **Lines 113-116**: Config validation
- **Lines 126-127**: Environment variable overrides

#### Proposed Tests:
```python
# tests/test_config.py
1. test_chroma_config_settings()
   - Test default ChromaDB paths
   - Test custom paths
   - Cover lines 67-73

2. test_embedding_provider_config()
   - Test OpenAI config
   - Test Cohere config
   - Test sentence-transformers
   - Cover lines 79-84

3. test_config_validation()
   - Test invalid configs
   - Test missing required fields
   - Cover lines 113-116, 126-127
```

**Potential Improvement: 72% → 95% (+23%)**

---

## 📊 Medium-Value Targets

### 6. **cli/formatters/tables.py** - Currently 66% (41 missing lines)
**Missing Lines:** 177-194, 206-220, 262-278, 291-300

#### Proposed Tests:
```python
# tests/test_formatters.py
1. test_cluster_table_formatting()
   - Test various cluster data formats
   - Cover lines 177-194

2. test_hierarchy_table_display()
   - Test nested hierarchy rendering
   - Cover lines 206-220

3. test_table_pagination()
   - Test large result sets
   - Cover lines 262-278, 291-300
```

**Potential Improvement: 66% → 85% (+19%)**

---

### 7. **cli/common.py** - Currently 68% (8 missing lines)
**Missing Lines:** 25-36

#### Proposed Tests:
```python
# tests/test_cli_common.py
1. test_database_path_resolution()
   - Test default paths
   - Test custom paths
   - Test environment overrides
   - Cover lines 25-36
```

**Potential Improvement: 68% → 100% (+32%)**

---

## 🔴 Low Priority (Complex Integration / CLI Commands)

These modules have low coverage but involve complex CLI interactions, external API calls, or are already tested via smoke tests:

- **loader.py** (12%) - Data loading from HuggingFace (tested via CLI)
- **runner.py** (17%) - Main workflow orchestration (tested via CLI)
- **hierarchy.py** (16%) - LLM-driven hierarchy merge (tested via CLI)
- **hdbscan_clustering.py** (17%) - HDBSCAN algorithm (tested via CLI)
- **CLI commands** (14-53%) - User-facing commands (tested via smoke tests)

**Recommendation:** Keep these tested via integration/smoke tests rather than unit tests.

---

## 📈 Implementation Roadmap

### Phase 1: Quick Wins (Est: +10% coverage)
1. ✅ Fix failing Cohere test (DONE)
2. Add `test_config.py` → +23% on config.py
3. Add `test_cli_common.py` → +32% on cli/common.py
4. Add `test_semantic_queries.py` → +16% on queries.py

**Expected Overall: 50% → 54%**

---

### Phase 2: Core Business Logic (Est: +8% coverage)
1. Add `test_embeddings_coverage.py` → +19% on embeddings.py
2. Add `test_kmeans_clustering.py` → +22% on kmeans.py
3. Add `test_chroma_integration.py` → +24% on chroma.py

**Expected Overall: 54% → 62%**

---

### Phase 3: Polish (Est: +3% coverage)
1. Add `test_formatters.py` → +19% on tables.py
2. Expand existing tests for edge cases

**Expected Overall: 62% → 65%**

---

## 🎯 Target Achieved: 65% Coverage

### Summary of New Tests Needed:
- **8 new test files** with ~30-40 new test functions
- **Focus areas:** embeddings, kmeans, chroma, semantic queries, config
- **Expected time:** 4-6 hours of development
- **Expected outcome:** 50% → 65% (+15 percentage points)

---

## 🚀 Beyond 65%: Diminishing Returns

To go beyond 65% would require:
- Mocking complex CLI workflows
- Testing HuggingFace data loading (flaky, slow)
- Testing full hierarchy merge pipeline (expensive LLM calls)
- Deep CLI command integration tests

**Recommendation:** Stop at 65-70% coverage. The remaining untested code is:
1. Already covered by smoke tests
2. Involves expensive external API calls
3. CLI presentation logic with minimal business logic
4. Integration code best tested end-to-end

---

## 📝 Notes

### Lines That Are Hard to Test:
- **API client initialization** (lines like 70-83 in embeddings.py) - Require API keys or mocks
- **Async event loops** (nested asyncio calls) - Require complex async mocking
- **CLI error handling** - Best tested via manual/smoke tests
- **Progress bars / Rich formatting** - Presentation layer, low value to test

### Testing Philosophy:
- ✅ **DO test:** Business logic, data transformations, algorithms
- ✅ **DO test:** Error conditions, edge cases, validation
- ⚠️ **CONSIDER:** Integration paths, API wrappers (with mocks)
- ❌ **DON'T test:** CLI presentation, complex external integrations (use smoke tests)

---

**Generated:** 2025-01-09  
**Next Review:** After Phase 1 implementation

