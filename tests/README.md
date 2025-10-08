# Testing Guide

This directory contains comprehensive tests for the LMSYS Query Analysis project.

## Test Structure

```
tests/
├── conftest.py                # Shared fixtures
├── unit/                      # Unit tests (no external dependencies)
│   ├── services/              # Business logic tests (94-100% coverage)
│   ├── formatters/            # Output formatting tests
│   └── helpers/               # Utility function tests
├── integration/               # Integration tests (services working together)
├── smoke/                     # Smoke tests with real API calls
│   ├── test_embedding_smoke.py      # Real embedding API calls
│   ├── test_clustering_smoke.py     # End-to-end clustering
│   ├── test_summarization_smoke.py  # Real LLM calls
│   ├── test_search_smoke.py         # Real ChromaDB operations
│   └── test_cli_smoke.py            # CLI command verification
└── fixtures/                  # Shared test data

```

## Running Tests

### Run All Unit Tests (Fast, No API Calls)
```bash
uv run pytest tests/unit/ -v
```

### Run Integration Tests
```bash
uv run pytest tests/integration/ -v
```

### Run Smoke Tests (Calls Real APIs - Use Sparingly)
```bash
# Run all smoke tests
uv run pytest tests/smoke/ -v -m smoke

# Run specific smoke test
uv run pytest tests/smoke/test_embedding_smoke.py::test_openai_embeddings -v
```

### Run All Tests with Coverage
```bash
uv run pytest tests/unit/ tests/integration/ --cov=src/lmsys_query_analysis --cov-report=term-missing
```

### Skip Smoke Tests (Default for CI)
```bash
uv run pytest -v -m "not smoke"
```

## Test Categories

### Unit Tests
- Test individual functions and services in isolation
- Use in-memory SQLite database
- No external API calls
- Fast execution (< 5 seconds)
- **Coverage: 94-100% for services layer**

### Integration Tests
- Test multiple services working together
- Verify data flows correctly through the system
- Use in-memory database with populated fixtures
- No external API calls

### Smoke Tests
- Verify real API integrations work
- Call actual embedding APIs (OpenAI, Cohere)
- Call actual LLMs for summarization
- Test ChromaDB operations
- Use real CLI commands
- **Note:** These tests cost money and should be run manually

## Key Fixtures

### Database Fixtures
- `temp_db` - Clean in-memory SQLite database
- `db_session` - Database session for tests
- `populated_db` - Database with sample queries, runs, and summaries

### Data Fixtures
- `sample_queries` - 5 sample queries for testing
- `sample_clustering_run` - Sample KMeans run
- `sample_query_clusters` - Query-to-cluster assignments
- `sample_cluster_summaries` - Sample cluster summaries

### Utility Fixtures
- `temp_dir` - Temporary directory for file operations
- `mock_chroma` - Mock ChromaDB client (for unit tests)

## Test Philosophy

1. **Services are fully tested** - All business logic has comprehensive unit tests
2. **Commands are lightly tested** - CLI commands are thin wrappers, smoke tested
3. **Smoke tests verify integrations** - Real API calls ensure system works end-to-end
4. **Integration tests verify workflows** - Services working together correctly

## Adding New Tests

### For New Services
1. Add unit tests in `tests/unit/services/`
2. Aim for 90%+ coverage
3. Test both success and error cases

### For New CLI Commands
1. Add to appropriate command module
2. Add smoke test in `tests/smoke/test_cli_smoke.py`

### For New API Integrations
1. Add smoke test with `@pytest.mark.smoke`
2. Document API key requirements
3. Keep tests minimal to avoid costs

## CI Configuration

For CI/CD, run:
```bash
uv run pytest -v -m "not smoke" --cov=src/lmsys_query_analysis --cov-fail-under=80
```

This skips expensive smoke tests while ensuring code quality.

