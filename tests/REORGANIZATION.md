# Test Reorganization Plan

## Unit Tests (to tests/unit/)
- test_embeddings.py - Mocked embedding tests
- test_hdbscan.py - Algorithm tests with numpy
- test_config.py - Pydantic model validation
- test_cli_common.py - CLI error handling
- test_hierarchy.py - Hierarchy algorithm
- test_summarizer.py - Prompt generation
- test_loader.py - Pure extract functions
- test_models.py - SQLModel tests

## Integration Tests (to tests/integration/)
- test_semantic_queries.py - Full DB+Chroma workflow
- test_semantic_sdk.py - SDK integration
- test_cli.py - CLI with real commands
- test_kmeans_detailed.py - Full clustering
- test_embeddings_advanced.py - Complex embedding workflows

## Already in correct location
- tests/smoke/test_embeddings_smoke.py ✓
- tests/unit/* ✓
- tests/integration/test_service_integration.py ✓
