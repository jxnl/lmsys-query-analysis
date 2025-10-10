# Test File Combination Plan

## ❌ Files with Issues

### test_chroma_integration.py
**Problem:** Mixes pure functions (11 tests) with integration tests (12 tests)

**Solution:**
1. Move all `test_sanitize_*` tests → `test_chroma_unit.py`
2. Keep only `test_chroma_manager_*` tests that use real ChromaDB
3. Result: Pure integration test file

**Why:** Pure functions don't need real ChromaDB, shouldn't be integration tests

## ✅ Files That Should Stay Separate

### Semantic Tests
- `test_semantic_queries.py` - QueriesClient unit functionality
- `test_semantic_sdk.py` - Full SDK integration
**Reason:** Different components being tested

### Embedding Tests  
- `test_embeddings.py` (unit) - Mocked, fast
- `test_embeddings_advanced.py` (integration) - Real components
- `test_embedding_smoke.py` (smoke) - Real APIs
**Reason:** Different test levels (unit/integration/smoke)

### CLI Tests
- `test_cli_common.py` (unit) - Error handlers
- `test_cli.py` (integration) - Full CLI commands  
- `test_cli_smoke.py` (smoke) - Real workflows
**Reason:** Different test levels

## Summary

**Combine:** test_chroma files (move pure functions from integration → unit)
**Keep Separate:** Everything else (properly organized by test level)
