# Testing Strategy Guide

## When to Mock vs When to Use Real Objects

### ✅ ALWAYS Mock These

1. **External APIs** - OpenAI, Cohere, Anthropic clients
2. **File System Operations** - Disk reads/writes, temp files
3. **Database Connections** - ChromaDB, external databases
4. **Network Calls** - HTTP requests, downloads
5. **Time-based Operations** - `time.sleep()`, `datetime.now()`
6. **Environment Variables** - When testing configuration
7. **Expensive Computations** - Model loading, large data processing

### ❌ DON'T Mock These

1. **Pure Functions** - `sanitize_collection_name()`, data transformations
2. **Simple Data Classes** - Pydantic models, dataclasses
3. **The Code You're Testing** - Mock dependencies, not the unit under test
4. **Standard Library** - Lists, dicts (unless performance critical)

## Test Types

### Unit Tests (`tests/unit/`)
- **Goal**: Test single functions/classes in isolation
- **Speed**: < 100ms each
- **Mocking**: Heavy - mock all external dependencies
- **Example**:
```python
@patch('module.external_api_client')
def test_process_data(mock_client):
    mock_client.fetch.return_value = {"data": "test"}
    result = process_data()
    assert result == expected
```

### Integration Tests (`tests/integration/`)
- **Goal**: Test multiple components together
- **Speed**: < 1 second each
- **Mocking**: Moderate - use in-memory databases, mock external APIs
- **Example**:
```python
def test_full_workflow(temp_db):
    # Use real DB but in-memory
    # Mock only external APIs
    result = workflow(temp_db)
    assert result.success
```

### Smoke Tests (`tests/smoke/`)
- **Goal**: Test critical paths end-to-end
- **Speed**: Can be slow (seconds)
- **Mocking**: Minimal - can hit real APIs (with skip if no keys)
- **Example**:
```python
@pytest.mark.smoke
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_real_embedding():
    embeddings = generate_real_embeddings()
    assert embeddings.shape == (10, 1536)
```

## Mocking Patterns

### Pattern 1: Mock External API Clients

```python
from unittest.mock import Mock, patch

@patch('module.OpenAI')
def test_with_openai(mock_openai_class):
    # Create mock instance
    mock_client = Mock()
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    mock_openai_class.return_value = mock_client
    
    # Test your code
    result = your_function()
    assert result is not None
```

### Pattern 2: Mock File Operations

```python
from unittest.mock import mock_open, patch

@patch('builtins.open', mock_open(read_data='test data'))
def test_file_reading():
    result = read_config_file()
    assert result == expected
```

### Pattern 3: Mock Database Operations

```python
@patch('module.ChromaManager')
def test_with_chroma(mock_chroma_class):
    mock_manager = Mock()
    mock_manager.count_queries.return_value = 100
    mock_chroma_class.return_value = mock_manager
    
    result = your_function()
    assert result == 100
```

### Pattern 4: Use In-Memory Databases for Integration

```python
def test_with_real_db():
    # Use SQLite in-memory - fast and isolated
    db = Database(":memory:")
    db.create_tables()
    
    # Test with real DB operations
    query = Query(conversation_id="test", ...)
    db.session.add(query)
    db.session.commit()
    
    # Verify
    result = db.session.query(Query).first()
    assert result.conversation_id == "test"
```

## Current Test Organization

```
tests/
├── unit/                    # Fast, heavily mocked
│   ├── formatters/         # Test table/JSON formatting (pure functions)
│   ├── helpers/            # Test helper utilities (mock external deps)
│   └── services/           # Test service layer (mock DB + APIs)
├── integration/            # Multiple components, in-memory DB
│   └── test_service_integration.py
├── smoke/                  # End-to-end, can hit real APIs
│   ├── test_cli_smoke.py
│   ├── test_embedding_smoke.py
│   └── test_search_smoke.py
└── test_*.py              # Main test files (various levels)
```

## Refactoring Checklist

For each test file, ask:

1. **Does this test hit the network?** → Mock it
2. **Does this test write to disk?** → Use temp dir or mock
3. **Does this test load a large model?** → Mock it
4. **Does this test take > 100ms?** → It's not a unit test
5. **Does this test require API keys?** → Mock or mark as smoke

## Example Refactoring

### Before (Slow, brittle):
```python
def test_embedding_generation():
    # Actually loads model, requires API key
    generator = EmbeddingGenerator(provider="openai", model="gpt-4")
    embeddings = generator.generate(["test"])
    assert embeddings.shape[0] == 1
```

### After (Fast, reliable):
```python
@patch('module.OpenAI')
def test_embedding_generation(mock_openai_class):
    mock_client = Mock()
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    mock_openai_class.return_value = mock_client
    
    generator = EmbeddingGenerator(provider="openai", model="gpt-4")
    embeddings = generator.generate(["test"])
    
    assert embeddings.shape == (1, 1536)
    mock_client.embeddings.create.assert_called_once()
```

## Benefits of Proper Mocking

1. **Speed**: Test suite runs in seconds, not minutes
2. **Reliability**: No flaky tests due to network issues
3. **Cost**: No API charges during testing
4. **Isolation**: Tests don't interfere with each other
5. **CI/CD**: Tests run anywhere without configuration

## Running Different Test Levels

```bash
# Fast unit tests only (< 1 second total)
pytest tests/unit/ -v

# Integration tests (few seconds)
pytest tests/integration/ -v

# Smoke tests (requires API keys)
pytest tests/smoke/ -m smoke -v

# All tests
pytest -v

# Skip slow tests
pytest -v -m "not smoke"
```

