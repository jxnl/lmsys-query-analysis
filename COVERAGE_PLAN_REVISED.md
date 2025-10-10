# Revised Test Coverage Plan - High-Impact Focus

**Current Coverage: 52.17% (1565/3000)**  
**Target: 65%+ (Focus on modules with most missing lines)**

---

## 🎯 High-Impact Strategy

### The Problem with Phase 1:
- ✅ Added 47 tests
- ✅ Got 3 modules to 100%
- ❌ Only gained 2% overall coverage
- **Why?** We tested small modules (234 statements) instead of large ones with most missing coverage

### New Approach:
**Focus on the 3 modules with most missing, testable lines:**

1. **embeddings.py** - 89 missing lines (most impactful)
2. **kmeans.py** - 61 missing lines (clustering core)
3. **chroma.py** - 42 missing lines (vector DB)

**Combined:** 192 missing lines = **6.4% potential gain**

---

## 📋 Implementation Checklist

### ✅ Phase 1: Quick Wins (COMPLETED)
- [x] test_config.py (21 tests) → config.py to 100%
- [x] test_cli_common.py (12 tests) → cli/common.py to 100%
- [x] test_semantic_queries.py (14 tests) → semantic/queries.py to 96%
- [x] **Result:** 52% coverage, +47 tests

---

### 🔥 Phase 2: High-Impact Modules (IN PROGRESS)

#### Module 1: embeddings.py (59% → 85%, +26%)
**Missing Lines:** 74, 76, 81, 131, 162-180, 227, 229-238, 265-275, 283-286, 308-356, 365-409, 417-422, 425

**Test File:** `tests/test_embeddings_advanced.py`

- [ ] **Test 1:** `test_openai_embeddings_with_mocked_api()` 
  - Mock OpenAI client responses
  - Test batch processing (lines 365-409)
  - Test API initialization (lines 74, 76, 81)
  - **Impact:** ~40 lines

- [ ] **Test 2:** `test_cohere_embeddings_with_mocked_api()`
  - Mock Cohere client responses  
  - Test async generation (lines 162-180)
  - Test batch operations (lines 229-238, 265-275, 308-356)
  - **Impact:** ~50 lines

- [ ] **Test 3:** `test_embedding_rate_limiting()`
  - Test rate limit handling (lines 417-422)
  - Test retry logic
  - **Impact:** ~10 lines

- [ ] **Test 4:** `test_embedding_error_paths()`
  - Test API failures
  - Test invalid inputs (line 131)
  - **Impact:** ~5 lines

**Expected:** embeddings.py 59% → 85% (+26%), ~4 tests, +105 statements

---

#### Module 2: kmeans.py (63% → 90%, +27%)
**Missing Lines:** 59-62, 106, 127-173, 188, 223-261, 264, 275, 290-292, 321-350, 371-395

**Test File:** `tests/test_kmeans_pipeline.py`

- [ ] **Test 1:** `test_kmeans_full_clustering_pipeline()`
  - Create queries with real embeddings
  - Run full KMeans clustering (lines 127-173)
  - Verify assignments (lines 223-261)
  - **Impact:** ~70 lines

- [ ] **Test 2:** `test_kmeans_cluster_quality_metrics()`
  - Test silhouette scores (lines 321-350)
  - Test inertia calculations (lines 371-395)
  - **Impact:** ~50 lines

- [ ] **Test 3:** `test_minibatch_kmeans_initialization()`
  - Test MiniBatch parameters (lines 59-62)
  - Test batch size variations
  - **Impact:** ~10 lines

**Expected:** kmeans.py 63% → 90% (+27%), ~3 tests, +130 statements

---

#### Module 3: chroma.py (62% → 90%, +28%)
**Missing Lines:** 34, 37, 159-179, 227-228, 270, 303, 319-320, 333-348, 360-361, 370-376, 380, 384-387, 411

**Test File:** `tests/test_chroma_operations.py`

- [ ] **Test 1:** `test_chroma_query_embedding_updates()`
  - Add queries with embeddings
  - Update embeddings (lines 159-179)
  - Verify persistence
  - **Impact:** ~30 lines

- [ ] **Test 2:** `test_chroma_cluster_summary_storage()`
  - Store cluster summaries (lines 333-348)
  - Retrieve by run_id
  - Test metadata (lines 227-228, 270, 303)
  - **Impact:** ~40 lines

- [ ] **Test 3:** `test_chroma_search_with_filters()`
  - Search with cluster filters (lines 370-376)
  - Search with metadata filters (lines 380, 384-387)
  - Test collection operations (lines 34, 37, 411)
  - **Impact:** ~25 lines

**Expected:** chroma.py 62% → 90% (+28%), ~3 tests, +95 statements

---

## 📊 Expected Phase 2 Results

| Module | Before | After | New Lines | Tests Added |
|--------|--------|-------|-----------|-------------|
| embeddings.py | 59% (114) | 85% (173) | +59 | 4 |
| kmeans.py | 63% (104) | 90% (149) | +45 | 3 |
| chroma.py | 62% (68) | 90% (99) | +31 | 3 |
| **TOTAL** | - | - | **+135** | **10** |

**Overall Coverage:** 52% → **56.5%** (+4.5%)

---

## 🚀 Phase 3: Additional High-Value Targets

After Phase 2, if we want to push further:

### Option A: Medium-Size Modules
- [ ] **cli/formatters/tables.py** (66% → 90%, +24%)
  - Table rendering tests
  - Pagination tests
  - **Impact:** +30 lines

### Option B: Integration Tests
- [ ] **Full clustering workflow integration test**
  - Load → Embed → Cluster → Summarize
  - **Impact:** +50 lines across multiple modules

### Option C: Smoke Test Expansion
- [ ] Add more end-to-end smoke tests
- [ ] Test error recovery paths
- [ ] **Impact:** +40 lines

---

## 🎯 Final Target: 60-65% Coverage

**Realistic Goal:**
- Phase 1 (✅ Done): 50% → 52%
- Phase 2 (🔥 Focus): 52% → 56.5%
- Phase 3 (Optional): 56.5% → 60-62%

**Total New Tests:** ~60-70 tests  
**Total New Coverage:** +10-12%

---

## 🚫 What We're NOT Testing

These modules are better tested via CLI/smoke tests:
- **runner.py** (17%) - Complex orchestration, external dependencies
- **hierarchy.py** (16%) - Expensive LLM calls, integration testing
- **loader.py** (12%) - HuggingFace API, flaky network calls
- **CLI commands** (14-53%) - User-facing, smoke tested

---

## ✅ Success Criteria

1. **Coverage:** 60%+ overall
2. **Core modules:** embeddings, kmeans, chroma at 85%+
3. **Tests:** 220+ total tests
4. **Quality:** All tests pass, no skipped tests without good reason
5. **Maintainability:** Tests use proper mocking, no API keys required

---

**Status:** 🔥 Phase 2 In Progress  
**Last Updated:** 2025-01-09  
**Next Action:** Implement embeddings.py tests

